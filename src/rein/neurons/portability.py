from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from rein.neurons import constants as c
from rein.neurons.bias import bias_channel
from rein.neurons.checkpoint import Expansion, PlacedNeuron, expand_checkpoint
from rein.neurons.extract import MlpInputCapture
from rein.neurons.units import Neuron, token_direction

FORCE_GAIN = 8.0
FORCE_RESIDUAL_RATIO = 10.0
TOOL_CALL_TOKEN = "<tool_call>"
SIGNAL_TOKEN = c.SIGNAL_TOKENS[0]
ZERO = "zero"
FORCE_TOOL_CALL = "force-tool-call"
FORCE_SIGNAL = "force-signal"
KL_EPS = 1e-12


@dataclass(frozen=True)
class LastLayerStats:
    """MLP inputs of the last layer and the typical size of its output, measured on real prompts."""

    inputs: torch.Tensor
    residual_norm: float


@torch.no_grad()
def last_layer_stats(backbone: Qwen3_5TextModel, sequences: Sequence[Sequence[int]]) -> LastLayerStats:
    layer = len(backbone.layers) - 1
    device = next(backbone.parameters()).device
    capture = MlpInputCapture(backbone, [layer])
    norms: list[torch.Tensor] = []

    def keep_norm(module: nn.Module, args: Any, output: Any) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        norms.append(hidden.float().norm(dim=-1).flatten().cpu())

    handle = cast(nn.Module, backbone.layers[layer]).register_forward_hook(keep_norm)
    inputs = []
    try:
        for ids in sequences:
            tokens = torch.tensor([list(ids)], device=device)
            capture.aim(torch.zeros(len(ids), dtype=torch.long, device=device), torch.arange(len(ids), device=device))
            backbone(input_ids=tokens, use_cache=False)
            inputs.append(capture.stacked()[:, 0].float().cpu())
    finally:
        capture.close()
        handle.remove()
    return LastLayerStats(torch.cat(inputs), float(torch.cat(norms).median()))


def always_on_neuron(
    channel: torch.Tensor,
    direction: torch.Tensor,
    residual_norm: float,
    gain: float = FORCE_GAIN,
    ratio: float = FORCE_RESIDUAL_RATIO,
) -> Neuron:
    """Fires on every token, because its gate reads the bias channel, and writes `direction` at `ratio` times the
    residual stream's size. Used only to prove that a runtime computes the added neurons."""
    unit = direction.float() / direction.float().norm()
    return Neuron(gate=gain * channel.float(), up=channel.float(), down=(ratio * residual_norm / gain) * unit)


def forcing_checkpoint(
    source: Path,
    target: Path,
    token: str,
    token_id: int,
    embedding: torch.Tensor,
    final_norm_weight: torch.Tensor,
    stats: LastLayerStats,
    layer: int,
    added: int = c.NEURON_ALIGN,
) -> Expansion:
    channel = bias_channel(stats.inputs)
    neuron = always_on_neuron(channel, token_direction(embedding[token_id], final_norm_weight), stats.residual_norm)
    placed = [PlacedNeuron(block=layer, slot=0, neuron=neuron, role=f"force:{token}")]
    return expand_checkpoint(source, target, added, placed, manifest={"purpose": f"force {token} at every position"})


@torch.no_grad()
def kl_stats(
    model_a: nn.Module, model_b: nn.Module, sequences: Sequence[Sequence[int]], starts: Sequence[int]
) -> dict[str, float]:
    """KL(a || b) of the next-token distributions over the continuation of each sequence, one sequence at a time."""
    device = next(model_a.parameters()).device
    values, agree = [], []
    for ids, start in zip(sequences, starts, strict=True):
        tokens = torch.tensor([list(ids)], device=device)
        log_a = torch.log_softmax(model_a(input_ids=tokens).logits[0, start - 1 : -1].float(), dim=-1)
        log_b = torch.log_softmax(
            model_b(input_ids=tokens.to(next(model_b.parameters()).device))
            .logits[0, start - 1 : -1]
            .float()
            .to(device),
            dim=-1,
        )
        values.append((log_a.exp() * (log_a - log_b)).sum(-1).clamp(min=0.0).cpu())
        agree.append((log_a.argmax(-1) == log_b.argmax(-1)).float().cpu())
    kl = torch.cat(values)
    return {
        "tokens": float(kl.numel()),
        "kl_mean": float(kl.mean()),
        "kl_p99": float(kl.quantile(0.99)),
        "kl_max": float(kl.max()),
        "top1_agreement": float(torch.cat(agree).mean()),
    }
