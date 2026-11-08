import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Neuron:
    """One FFN neuron: the gate row detects, the up row scales, the down column writes back to the residual stream."""

    gate: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor


def logit(probability: float) -> float:
    if not 0.0 < probability < 1.0:
        raise ValueError(f"a gate probability must sit strictly between 0 and 1, got {probability}")
    return math.log(probability / (1.0 - probability))


def threshold_gate(
    detector: torch.Tensor, offset: float, channel: torch.Tensor, temperature: float, gate_probability: float
) -> torch.Tensor:
    """Gate row whose pre-activation is the calibrated logit minus the sheet's gate, in logit units.

    With `c . x = 1`, `row . x = (detector . x + offset) / temperature - logit(gate_probability)`.
    """
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    return (detector + (offset - temperature * logit(gate_probability)) * channel) / temperature


def step_pair(
    gate_row: torch.Tensor,
    channel: torch.Tensor,
    gain: float,
    width: float,
    up_scale: float,
    steer: torch.Tensor,
) -> tuple[Neuron, Neuron]:
    """Two neurons whose sum is a bounded soft step: about 0 below the threshold, `width * up_scale * steer` above."""
    if gain <= 0.0 or width <= 0.0:
        raise ValueError("gain and width must be positive")
    first = gain * gate_row
    up = up_scale * channel
    return Neuron(first, up, steer), Neuron(first - width * channel, up, -steer)


def token_direction(embedding_row: torch.Tensor, final_norm_weight: torch.Tensor) -> torch.Tensor:
    """Residual direction that raises one token's logit: Qwen3.5 ties the lm_head to the embeddings and its final
    RMSNorm scales by `1 + weight`."""
    return (1.0 + final_norm_weight.float()) * embedding_row.float()


def mlp_contribution(inputs: torch.Tensor, neurons: Sequence[Neuron]) -> torch.Tensor:
    """What a set of SwiGLU neurons adds to the residual stream, for MLP inputs of shape (..., hidden)."""
    gates = torch.stack([neuron.gate for neuron in neurons]).to(inputs.dtype)
    ups = torch.stack([neuron.up for neuron in neurons]).to(inputs.dtype)
    downs = torch.stack([neuron.down for neuron in neurons]).to(inputs.dtype)
    return (F.silu(inputs @ gates.T) * (inputs @ ups.T)) @ downs
