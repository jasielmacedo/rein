"""Soft tokens: new special tokens living in the embedding table's spare rows, the only weights a prompt-tuned decider
trains. They appear only in the decision suffix, so the chat never meets them and the generator stays untouched."""

from collections.abc import Sequence
from typing import Any, cast

import torch
from torch import nn
from transformers import PreTrainedTokenizerBase
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from rein.model import constants as c


def input_embeddings(backbone: Qwen3_5TextModel) -> nn.Embedding:
    embedding = backbone.get_input_embeddings()
    if not isinstance(embedding, nn.Embedding):
        raise TypeError(f"the backbone's input embeddings are a {type(embedding).__name__}, not an Embedding")
    return embedding


def soft_token_texts(front: int, tail: int) -> tuple[list[str], list[str]]:
    texts = [c.SOFT_TOKEN_FORMAT.format(index=index) for index in range(front + tail)]
    return texts[:front], texts[front:]


def add_soft_tokens(
    tokenizer: PreTrainedTokenizerBase, front: int, tail: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Registers the soft tokens as special tokens, once, and returns their ids: the front group and the tail group."""
    front_texts, tail_texts = soft_token_texts(front, tail)
    texts = front_texts + tail_texts
    missing = [text for text in texts if text not in tokenizer.get_added_vocab()]
    if missing:
        tokenizer.add_tokens(missing, special_tokens=True)
    ids = [int(token) for token in cast(list[int], tokenizer.convert_tokens_to_ids(texts))]
    return tuple(ids[:front]), tuple(ids[front:])


def project_off(rows: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """`rows` without their components along the orthonormal `basis` rows; `rows` itself when the basis is empty."""
    if basis.shape[0] == 0:
        return rows
    return rows - (rows @ basis.T) @ basis


class SoftPrompt(nn.Module):
    """The soft tokens' embeddings, served in place of the table's rows through a hook on the embedding module.

    With a basis set, the rows served and merged are the trained weight projected off the basis: the tied output
    head then reads nothing from those directions, whatever the context."""

    ids: torch.Tensor
    basis: torch.Tensor

    def __init__(
        self, ids: Sequence[int], embedding: nn.Embedding, init_vocab: int = c.SOFT_INIT_VOCAB, seed: int = 0
    ) -> None:
        super().__init__()
        if not ids:
            raise ValueError("soft tokens need at least one id")
        if max(ids) >= embedding.num_embeddings:
            raise ValueError(
                f"soft token id {max(ids)} is beyond the embedding table ({embedding.num_embeddings} rows)"
            )
        self.register_buffer("ids", torch.tensor(list(ids), dtype=torch.long), persistent=False)
        self.register_buffer("basis", torch.empty(0, embedding.embedding_dim), persistent=False)
        generator = torch.Generator().manual_seed(seed)
        source = torch.randint(0, init_vocab, (len(ids),), generator=generator).to(embedding.weight.device)
        self.weight = nn.Parameter(embedding.weight.detach()[source].clone().float())

    def set_basis(self, basis: torch.Tensor) -> None:
        if basis.shape[1] != self.weight.shape[1]:
            raise ValueError(f"basis vectors have {basis.shape[1]} dimensions, the soft rows {self.weight.shape[1]}")
        self.basis = basis.detach().to(self.weight.device, torch.float32).contiguous()
        with torch.no_grad():
            self.weight.copy_(self.rows)

    @property
    def rows(self) -> torch.Tensor:
        return project_off(self.weight, self.basis)

    def attach(self, embedding: nn.Embedding) -> torch.utils.hooks.RemovableHandle:
        return embedding.register_forward_hook(self._swap)

    def _swap(self, module: nn.Module, inputs: tuple[Any, ...], output: torch.Tensor) -> torch.Tensor:
        token_ids = inputs[0]
        index = torch.full_like(token_ids, -1)
        for slot, token in enumerate(self.ids.tolist()):
            index[token_ids == token] = slot
        mask = index >= 0
        if not bool(mask.any()):
            return output
        swapped = output.clone()
        swapped[mask] = self.rows[index[mask]].to(output.dtype)
        return swapped

    def merge_into(self, embedding: nn.Embedding) -> None:
        """Writes the trained rows into the table, so a plain checkpoint carries them."""
        with torch.no_grad():
            embedding.weight[self.ids] = self.rows.to(embedding.weight.dtype)


def stored_rows(tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    """The rows a saved soft prompt serves: its weight projected off its basis when it has one."""
    return project_off(tensors["weight"].float(), tensors.get("basis", torch.empty(0, tensors["weight"].shape[1])))
