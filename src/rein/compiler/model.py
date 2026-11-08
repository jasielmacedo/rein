from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from rein.compiler import constants as c
from rein.compiler.reader import RuleReading
from rein.model.head import SetBlock


@dataclass(frozen=True)
class CompiledRules:
    """Per rule, option, target block and facet: a detector over the MLP input and an offset along the bias channel.

    `detectors` is (rules, options, blocks, facets, hidden) and `offsets` is (rules, options, blocks, facets).
    """

    detectors: torch.Tensor
    offsets: torch.Tensor
    mask: torch.Tensor
    blocks: tuple[int, ...]

    def take(self, index: torch.Tensor) -> "CompiledRules":
        return CompiledRules(self.detectors[index], self.offsets[index], self.mask[index], self.blocks)


class RuleEncoder(nn.Module):
    """A transformer without positional encoding over {query, options}: options compare to each other in any order."""

    def __init__(self, hidden_size: int, dim: int, heads: int, layers: int, ffn_mult: int, dropout: float) -> None:
        super().__init__()
        self.query_in = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, dim))
        self.option_in = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, dim))
        self.blocks = nn.ModuleList(SetBlock(dim, heads, ffn_mult, dropout) for _ in range(layers))
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, reading: RuleReading) -> torch.Tensor:
        query = self.query_in(reading.query.float())
        options = self.option_in(reading.options.float())
        tokens = torch.cat([query[:, None], options], dim=1)
        valid = torch.cat([torch.ones_like(reading.mask[:, :1]), reading.mask], dim=1)
        for block in self.blocks:
            tokens = block(tokens, valid)
        return self.out_norm(tokens[:, 1:])


class Compiler(nn.Module):
    """Turns the frozen reading of a rule into neuron detectors: per option, target block and facet.

    Each option's encoding picks its detectors in a learned basis of the block's MLP input space. With several facets
    an option fires when any facet does, which several neurons express exactly.
    """

    def __init__(
        self,
        hidden_size: int,
        blocks: Sequence[int],
        dim: int = c.MODEL_DIM,
        heads: int = c.MODEL_HEADS,
        layers: int = c.MODEL_LAYERS,
        rank: int = c.DETECTOR_RANK,
        ffn_mult: int = c.MODEL_FFN_MULT,
        dropout: float = c.DROPOUT,
        facets: int = 1,
    ) -> None:
        super().__init__()
        if not blocks or facets < 1:
            raise ValueError("the compiler needs at least one target block and one facet")
        self.blocks = tuple(blocks)
        self.rank = rank
        self.facets = facets
        self.encoder = RuleEncoder(hidden_size, dim, heads, layers, ffn_mult, dropout)
        self.codes = nn.Linear(dim, rank * len(self.blocks) * facets)
        self.offsets = nn.Linear(dim, len(self.blocks) * facets)
        self.bases = nn.Parameter(c.BASIS_INIT_STD * torch.randn(len(self.blocks), rank, hidden_size))

    @property
    def output_blocks(self) -> tuple[int, ...]:
        return self.blocks

    def forward(self, reading: RuleReading) -> CompiledRules:
        outputs = self.encoder(reading)
        rules, width = outputs.shape[:2]
        codes = self.codes(outputs).view(rules, width, len(self.blocks), self.facets, self.rank)
        detectors = torch.einsum("nkbfr,brh->nkbfh", codes, self.bases)
        offsets = self.offsets(outputs).view(rules, width, len(self.blocks), self.facets)
        return CompiledRules(detectors, offsets, reading.mask, self.blocks)

    def item_logits(
        self,
        reading: RuleReading,
        inverse: torch.Tensor,
        states: torch.Tensor,
        channels: torch.Tensor,
        nulls: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return decision_logits(self(reading).take(inverse), states, channels, nulls)


def decision_logits(
    compiled: CompiledRules, states: torch.Tensor, channels: torch.Tensor, nulls: torch.Tensor | None = None
) -> torch.Tensor:
    """Option logits of each rule from the MLP inputs at its decision position: the neurons' own pre-activations.

    `states` is (rules, blocks, hidden) and `channels` is (blocks, hidden); the result is (rules, blocks, options).
    Facets of an option pool with logsumexp; the null option (`nulls`, -1 for none) reads its first facet only, so
    each facet of a firing option against the null option stays a single linear comparison, as a neuron needs.
    """
    bias = torch.einsum("nbh,bh->nb", states.float(), channels.float())
    facet_logits = torch.einsum("nkbfh,nbh->nbkf", compiled.detectors, states.float())
    facet_logits = facet_logits + compiled.offsets.permute(0, 2, 1, 3) * bias[..., None, None]
    logits = torch.logsumexp(facet_logits, dim=-1)
    if nulls is not None and facet_logits.shape[-1] > 1:
        width = logits.shape[-1]
        is_null = F.one_hot(nulls.clamp(min=0), width).bool() & (nulls >= 0)[:, None]
        logits = torch.where(is_null[:, None, :], facet_logits[..., 0], logits)
    return logits.masked_fill(~compiled.mask[:, None, :], torch.finfo(logits.dtype).min)


def null_option_index(option_keys: Sequence[str]) -> int:
    """The option meaning "the rule does not fire", or NO_NULL when the rule has none (score rules)."""
    for key in c.NULL_OPTIONS:
        if key in option_keys:
            return list(option_keys).index(key)
    return c.NO_NULL
