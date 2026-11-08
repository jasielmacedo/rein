"""The pieces every training loss of the module shares: batches of prompts and replies, the forward pass with the
module bypassed (which computes what Qwen computes), next-token log-probabilities through the output head tied to the
embeddings, the KL to Qwen and the decision read at the reply opening."""

import random
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Protocol

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from rein.module.targets import LabelTable
from rein.module.train import readout_loss
from rein.neurons.extract import MlpInputCapture, padded_ids

PROMPT_KL_POSITIONS = 64

Teacher = Mapping[str, tuple[float, ...]]


class Read(Protocol):
    """Anything the decision read learns from at its reply opening."""

    @property
    def record_id(self) -> str: ...

    @property
    def view_id(self) -> str: ...


@dataclass(frozen=True)
class Stacked:
    """Prompts and replies in one padded batch, with every position that predicts a reply token."""

    ids: torch.Tensor
    mask: torch.Tensor
    rows: torch.Tensor
    columns: torch.Tensor
    targets: torch.Tensor
    openings: torch.Tensor
    prompt_rows: torch.Tensor
    prompt_columns: torch.Tensor


def stacked(
    pieces: Sequence[tuple[Sequence[int], Sequence[int]]], pad_id: int, rng: random.Random, prompted: int
) -> Stacked:
    """The batch, with up to PROMPT_KL_POSITIONS positions drawn inside each of the first `prompted` prompts."""
    ids, mask = padded_ids([(*prompt, *reply) for prompt, reply in pieces], pad_id)
    rows, columns, targets, prompt_rows, prompt_columns = [], [], [], [], []
    for row, (prompt, reply) in enumerate(pieces):
        for offset, token in enumerate(reply):
            rows.append(row)
            columns.append(len(prompt) - 1 + offset)
            targets.append(token)
        if row < prompted:
            inside = range(len(prompt) - 1)
            drawn = sorted(rng.sample(inside, min(PROMPT_KL_POSITIONS, len(inside))))
            prompt_rows += [row] * len(drawn)
            prompt_columns += drawn
    openings = torch.tensor([len(prompt) - 1 for prompt, _ in pieces])
    return Stacked(
        ids,
        mask,
        torch.tensor(rows),
        torch.tensor(columns),
        torch.tensor(targets),
        openings,
        torch.tensor(prompt_rows, dtype=torch.long),
        torch.tensor(prompt_columns, dtype=torch.long),
    )


@contextmanager
def bypassed(backbone: Qwen3_5TextModel, layers: Sequence[int]) -> Iterator[None]:
    """The checkpoint as Qwen: each module layer hands its input on unchanged."""

    def skip(module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> Any:
        return args[0] if args else kwargs["hidden_states"]

    handles = [backbone.layers[layer].register_forward_hook(skip, with_kwargs=True) for layer in layers]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def last_hidden(backbone: Qwen3_5TextModel, batch: Stacked) -> torch.Tensor:
    device = backbone.embed_tokens.weight.device
    output = backbone(input_ids=batch.ids.to(device), attention_mask=batch.mask.to(device), use_cache=False)
    return output.last_hidden_state


def log_probs_at(
    backbone: Qwen3_5TextModel, hidden: torch.Tensor, rows: torch.Tensor, columns: torch.Tensor
) -> torch.Tensor:
    """The next-token log-probabilities at the given positions, through the output head tied to the embeddings."""
    picked = hidden[rows.to(hidden.device), columns.to(hidden.device)]
    return torch.log_softmax(F.linear(picked, backbone.embed_tokens.weight).float(), dim=-1)


def reply_log_probs(backbone: Qwen3_5TextModel, batch: Stacked) -> torch.Tensor:
    """The next-token log-probabilities at every reply position."""
    return log_probs_at(backbone, last_hidden(backbone, batch), batch.rows, batch.columns)


def divergence(reference: torch.Tensor, policy: torch.Tensor) -> torch.Tensor:
    """KL from Qwen to the checkpoint, averaged over positions (zero when there are none)."""
    if not len(reference):
        return policy.sum() * 0.0
    return (reference.exp() * (reference - policy)).sum(-1).mean()


def sequence_sums(log_probs: torch.Tensor, batch: Stacked, count: int) -> torch.Tensor:
    targets = batch.targets.to(log_probs.device)
    picked = log_probs.gather(-1, targets[:, None])[:, 0]
    return torch.zeros(count, device=log_probs.device).scatter_add(0, batch.rows.to(log_probs.device), picked)


def decision_rows(
    examples: Sequence[Read], table: LabelTable, teacher: Teacher | None
) -> tuple[list[int], torch.Tensor]:
    """The examples the decision read learns from and their targets: every example's own labels, or, with a teacher,
    its soft labels where it has some."""
    if teacher is None:
        return list(range(len(examples))), torch.tensor(
            [table.by_record.get(example.record_id, table.quiet) for example in examples]
        )
    kept = [index for index, example in enumerate(examples) if example.view_id in teacher]
    return kept, torch.tensor([teacher[examples[index].view_id] for index in kept], dtype=torch.float)


def decision_loss_of(
    head: nn.Linear,
    capture: MlpInputCapture,
    examples: Sequence[Read],
    table: LabelTable,
    teacher: Teacher | None,
    brier_weight: float,
) -> torch.Tensor:
    kept, labels = decision_rows(examples, table, teacher)
    if not kept:
        return head.weight.sum() * 0.0
    states = capture.stacked()[:, 0][torch.tensor(kept, device=head.weight.device)]
    return readout_loss(head, states, labels, brier_weight).total
