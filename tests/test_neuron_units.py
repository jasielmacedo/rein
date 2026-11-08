import math

import pytest
import torch

from rein.neurons.bias import bias_channel, channel_stats
from rein.neurons.units import logit, mlp_contribution, step_pair, threshold_gate, token_direction

HIDDEN = 8
GAIN = 8.0
WIDTH = 4.0
UP_SCALE = 1.0
STEP_ON = 3.0
STEP_OFF = -3.0
STEP_TOLERANCE = 1e-3
OFF_TOLERANCE = 1e-6
SAMPLES = 4096
CONSTANT_LEVEL = 3.0
NOISE = 0.5
TOKENS = 50
FORCE_SCALE = 1e3
RMS_EPS = 1e-6


def _unit(index: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(torch.tensor(index), HIDDEN).float()


def _inputs(detector_values: torch.Tensor) -> torch.Tensor:
    """MLP inputs whose bias channel (dimension 0) reads exactly 1 and whose detector (dimension 1) varies."""
    inputs = torch.zeros(len(detector_values), HIDDEN)
    inputs[:, 0] = 1.0
    inputs[:, 1] = detector_values
    return inputs


def test_step_pair_is_a_bounded_soft_step() -> None:
    channel, steer = _unit(0), _unit(2)
    gate = threshold_gate(_unit(1), offset=0.0, channel=channel, temperature=1.0, gate_probability=0.5)
    neurons = step_pair(gate, channel, GAIN, WIDTH, UP_SCALE, steer)
    out = mlp_contribution(_inputs(torch.tensor([STEP_OFF, STEP_ON])), neurons)
    assert out[0].abs().max() < OFF_TOLERANCE
    assert out[1, 2].item() == pytest.approx(WIDTH * UP_SCALE, abs=STEP_TOLERANCE)
    assert out[1, [0, 1, 3]].abs().max() < OFF_TOLERANCE


@pytest.mark.parametrize("gate_probability", [0.5, 0.85, 0.95])
def test_threshold_gate_crosses_zero_at_the_gate_probability(gate_probability: float) -> None:
    temperature, offset = 1.3, 0.4
    channel = _unit(0)
    row = threshold_gate(_unit(1), offset, channel, temperature, gate_probability)
    crossing = temperature * logit(gate_probability) - offset
    values = torch.tensor([crossing - 0.1, crossing, crossing + 0.1])
    pre_activation = _inputs(values) @ row
    assert pre_activation[0] < 0 < pre_activation[2]
    assert pre_activation[1].item() == pytest.approx(0.0, abs=1e-6)
    probability = 1 / (1 + math.exp(-(crossing + offset) / temperature))
    assert probability == pytest.approx(gate_probability)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_logit_rejects_probabilities_outside_the_open_interval(bad: float) -> None:
    with pytest.raises(ValueError):
        logit(bad)


def test_token_direction_raises_that_token_through_the_final_norm() -> None:
    generator = torch.Generator().manual_seed(0)
    embedding = torch.randn(TOKENS, HIDDEN, generator=generator)
    norm_weight = 0.1 * torch.randn(HIDDEN, generator=generator)
    hidden = torch.randn(HIDDEN, generator=generator)

    def logits(state: torch.Tensor) -> torch.Tensor:
        normed = state * torch.rsqrt(state.pow(2).mean() + RMS_EPS) * (1.0 + norm_weight)
        return embedding @ normed

    for token in (3, 17, 41):
        pushed = hidden + FORCE_SCALE * token_direction(embedding[token], norm_weight)
        assert int(logits(pushed).argmax()) == token


def test_bias_channel_reads_one_with_less_variation_than_the_mean_direction() -> None:
    generator = torch.Generator().manual_seed(0)
    constant = torch.randn(HIDDEN, generator=generator)
    constant = CONSTANT_LEVEL * constant / constant.norm()
    samples = constant + NOISE * torch.randn(SAMPLES, HIDDEN, generator=generator)
    samples[:, -1] = 1.0 + 0.01 * torch.randn(SAMPLES, generator=generator)
    channel = bias_channel(samples)
    stats = channel_stats(channel, samples)
    assert stats.mean == pytest.approx(1.0, abs=1e-4)
    mean = samples.mean(0)
    naive = channel_stats(mean / (mean @ mean), samples)
    assert stats.variation < naive.variation


def test_bias_channel_needs_at_least_two_samples() -> None:
    with pytest.raises(ValueError):
        bias_channel(torch.ones(1, HIDDEN))
