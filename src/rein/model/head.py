from typing import cast

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from rein.model import constants as c


class SetAttention(nn.Module):
    """Multi-head self-attention over a set with a key padding mask, built on SDPA so it never leaves the device."""

    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("head dim must divide evenly across heads")
        self.heads = heads
        self.dropout = dropout
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, tokens: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        batch, length, dim = tokens.shape
        qkv = self.qkv(tokens).view(batch, length, 3, self.heads, dim // self.heads).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        allow = valid[:, None, None, :]
        attended = F.scaled_dot_product_attention(
            query, key, value, attn_mask=allow, dropout_p=self.dropout if self.training else 0.0
        )
        return self.out(attended.transpose(1, 2).reshape(batch, length, dim))


class SetBlock(nn.Module):
    """Pre-norm transformer block over a set: no positional encoding, so it is permutation-equivariant."""

    def __init__(self, dim: int, heads: int, ffn_mult: int, dropout: float) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = SetAttention(dim, heads, dropout)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_mult * dim), nn.GELU(), nn.Linear(ffn_mult * dim, dim))

    def forward(self, tokens: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        tokens = tokens + self.attn(self.attn_norm(tokens), valid)
        return tokens + self.ffn(self.ffn_norm(tokens))


def layout_for(readout: str) -> str:
    return c.LAYOUT_LABELS if readout == c.READOUT_LABELS else c.LAYOUT_ROWS


def build_head(
    readout: str, backbone: Qwen3_5TextModel, dim: int, heads: int, set_block: bool
) -> "SetPointerHead | LabelHead":
    if readout == c.READOUT_LABELS:
        return LabelHead(cast(nn.Embedding, backbone.get_input_embeddings()))
    return SetPointerHead(hidden_size=backbone.config.hidden_size, dim=dim, heads=heads, set_block=set_block)


class LabelHead(nn.Module):
    """Scores each option by the logit of its label token at the row's last position, through the tied embeddings.

    It owns no weights: the embeddings belong to the backbone, so the decision is the one a runtime reads from the
    logits of the label tokens.
    """

    embeddings: nn.Embedding

    def __init__(self, embeddings: nn.Embedding) -> None:
        super().__init__()
        object.__setattr__(self, "embeddings", embeddings)

    def forward(self, h_query: torch.Tensor, label_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        rows = F.embedding(label_ids, self.embeddings.weight).float()
        logits = torch.einsum("qd,qod->qo", h_query.float(), rows)
        return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)


class SetPointerHead(nn.Module):
    """Scores each option of a question from the query and option hidden states, in any option order."""

    def __init__(
        self,
        hidden_size: int,
        dim: int = c.HEAD_DIM,
        heads: int = c.HEAD_HEADS,
        ffn_mult: int = c.HEAD_FFN_MULT,
        dropout: float = 0.0,
        set_block: bool = True,
    ) -> None:
        super().__init__()
        self.query_in = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, dim))
        self.option_in = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, dim))
        self.block = SetBlock(dim, heads, ffn_mult, dropout) if set_block else None
        self.query_out = nn.Linear(dim, dim)
        self.option_out = nn.Linear(dim, dim)
        self.scale = dim**-0.5

    def forward(self, h_query: torch.Tensor, h_options: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        query = self.query_in(h_query.float())
        options = self.option_in(h_options.float())
        if self.block is not None:
            tokens = torch.cat([query[:, None], options], dim=1)
            valid = torch.cat([torch.ones_like(mask[:, :1]), mask], dim=1)
            tokens = self.block(tokens, valid)
            query, options = tokens[:, 0], tokens[:, 1:]
        logits = (self.option_out(options) * self.query_out(query)[:, None]).sum(-1) * self.scale
        return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
