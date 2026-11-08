from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from rein.compiler import constants as c
from rein.compiler.model import RuleEncoder
from rein.compiler.reader import RuleReading

CIRCUIT_BLOCKS = 2


@dataclass(frozen=True)
class CompiledCircuits:
    """Per rule and option: feature neurons at the first block and the gate neuron at the second.

    `features` is (rules, options, units, hidden) with `feature_offsets` and `readout` (rules, options, units);
    `detectors` is (rules, options, hidden) with `offsets` (rules, options).
    """

    features: torch.Tensor
    feature_offsets: torch.Tensor
    readout: torch.Tensor
    detectors: torch.Tensor
    offsets: torch.Tensor
    mask: torch.Tensor

    def take(self, index: torch.Tensor) -> "CompiledCircuits":
        return CompiledCircuits(
            self.features[index],
            self.feature_offsets[index],
            self.readout[index],
            self.detectors[index],
            self.offsets[index],
            self.mask[index],
        )


class CircuitCompiler(nn.Module):
    """A two-block circuit per option, made only of FFN neurons.

    Units at the first block turn on for features of the rule and write them along a private residual direction; a
    neuron at the second block reads that direction with its own detector and applies the gate. This logit is the
    circuit's upper bound: it ignores how the second block's RMSNorm rescales the written features.
    """

    def __init__(
        self,
        hidden_size: int,
        blocks: tuple[int, int],
        units: int,
        dim: int = c.MODEL_DIM,
        heads: int = c.MODEL_HEADS,
        layers: int = c.MODEL_LAYERS,
        rank: int = c.DETECTOR_RANK,
        ffn_mult: int = c.MODEL_FFN_MULT,
        dropout: float = c.DROPOUT,
    ) -> None:
        super().__init__()
        if len(blocks) != CIRCUIT_BLOCKS or blocks[0] >= blocks[1] or units < 1:
            raise ValueError("a circuit needs an earlier and a later block and at least one unit")
        self.blocks = tuple(blocks)
        self.units = units
        self.rank = rank
        self.encoder = RuleEncoder(hidden_size, dim, heads, layers, ffn_mult, dropout)
        self.feature_codes = nn.Linear(dim, units * rank)
        self.feature_offsets = nn.Linear(dim, units)
        self.readout = nn.Linear(dim, units)
        self.detector_codes = nn.Linear(dim, rank)
        self.offsets = nn.Linear(dim, 1)
        self.first_basis = nn.Parameter(c.BASIS_INIT_STD * torch.randn(rank, hidden_size))
        self.second_basis = nn.Parameter(c.BASIS_INIT_STD * torch.randn(rank, hidden_size))

    @property
    def output_blocks(self) -> tuple[int, ...]:
        return (self.blocks[1],)

    def forward(self, reading: RuleReading) -> CompiledCircuits:
        outputs = self.encoder(reading)
        rules, width = outputs.shape[:2]
        codes = self.feature_codes(outputs).view(rules, width, self.units, self.rank)
        return CompiledCircuits(
            features=torch.einsum("nkur,rh->nkuh", codes, self.first_basis),
            feature_offsets=self.feature_offsets(outputs),
            readout=self.readout(outputs),
            detectors=torch.einsum("nkr,rh->nkh", self.detector_codes(outputs), self.second_basis),
            offsets=self.offsets(outputs).squeeze(-1),
            mask=reading.mask,
        )

    def item_logits(
        self,
        reading: RuleReading,
        inverse: torch.Tensor,
        states: torch.Tensor,
        channels: torch.Tensor,
        nulls: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return circuit_logits(self(reading).take(inverse), states, channels)


def circuit_logits(compiled: CompiledCircuits, states: torch.Tensor, channels: torch.Tensor) -> torch.Tensor:
    """(rules, 1, options): the second block's gate pre-activation, fed by the first block's feature neurons.

    `states` is (rules, 2, hidden), the MLP inputs of the two blocks; `channels` is (2, hidden).
    """
    first, second = states[:, 0].float(), states[:, 1].float()
    first_bias = first @ channels[0].float()
    second_bias = second @ channels[1].float()
    features = torch.einsum("nkuh,nh->nku", compiled.features, first)
    features = F.silu(features + compiled.feature_offsets * first_bias[:, None, None])
    logits = (features * compiled.readout).sum(-1)
    logits = logits + torch.einsum("nkh,nh->nk", compiled.detectors, second) + compiled.offsets * second_bias[:, None]
    return logits.masked_fill(~compiled.mask, torch.finfo(logits.dtype).min)[:, None, :]
