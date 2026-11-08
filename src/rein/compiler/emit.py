import torch

from rein.compiler.model import CompiledRules
from rein.neurons.units import Neuron, step_pair, threshold_gate


def firing_gate(
    compiled: CompiledRules,
    rule: int,
    option: int,
    null: int,
    block: int,
    channel: torch.Tensor,
    temperature: float,
    gate_probability: float,
    facet: int = 0,
) -> torch.Tensor:
    """Gate row for "option beats the null option": its pre-activation is the calibrated logit of that comparison
    minus the sheet's gate, so the neuron turns on exactly where the compiler's probability crosses the gate.

    For yes/no rules the comparison is the whole decision; for choice rules it is the option against "none". Each
    facet of the option compiles to its own gate, against the null option's single facet.
    """
    position = compiled.blocks.index(block)
    detector = compiled.detectors[rule, option, position, facet] - compiled.detectors[rule, null, position, 0]
    offset = float(compiled.offsets[rule, option, position, facet] - compiled.offsets[rule, null, position, 0])
    return threshold_gate(detector.detach().float(), offset, channel.float(), temperature, gate_probability)


def rule_neurons(
    gate_row: torch.Tensor, channel: torch.Tensor, steer: torch.Tensor, gain: float, width: float, up_scale: float
) -> tuple[Neuron, Neuron]:
    """The two neurons a firing option compiles to: a bounded step that writes `steer` once the gate is crossed."""
    return step_pair(gate_row, channel.float(), gain, width, up_scale, steer.float())
