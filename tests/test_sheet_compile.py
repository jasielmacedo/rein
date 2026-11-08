import torch

from rein.neurons.units import mlp_contribution
from rein.sheets.compile import FittedQuestion, Strengths, question_neurons

HIDDEN = 8
TEMPERATURE = 1.0
CONFIDENT, UNSURE, DOUBTFUL = 5.0, 0.5, -5.0
ATOL = 0.05


def _input(logit: float) -> torch.Tensor:
    """A state on the bias channel (first coordinate 1) whose yes-versus-no logit is `logit`."""
    x = torch.zeros(HIDDEN)
    x[0], x[1] = 1.0, logit
    return x


def _question(action: str) -> FittedQuestion:
    detectors = torch.zeros(2, HIDDEN)
    detectors[1, 1] = 1.0
    return FittedQuestion(detectors, torch.zeros(2), ("no", "yes"), action, "rule")


def _write(action: str, logit: float) -> torch.Tensor:
    channel = torch.zeros(HIDDEN)
    channel[0] = 1.0
    tool = torch.zeros(HIDDEN)
    tool[2] = 3.0
    vectors = {"hold": torch.zeros(HIDDEN).index_fill(0, torch.tensor([3]), 2.0)}
    strengths = Strengths(force=4.0, veto=6.0, steer={"hold": 1.5})
    neurons = question_neurons(_question(action), channel, TEMPERATURE, tool, strengths, vectors, {"rule": "hold"})
    return mlp_contribution(_input(logit)[None], [neuron for _, neuron in neurons])[0]


def test_a_steer_rule_writes_its_vector_only_when_the_rule_fires() -> None:
    assert torch.allclose(_write("steer", CONFIDENT)[3], torch.tensor(3.0), atol=ATOL)
    assert torch.allclose(_write("steer", DOUBTFUL), torch.zeros(HIDDEN), atol=ATOL)


def test_a_tool_rule_forces_the_call_when_sure_and_vetoes_it_when_sure_there_is_none() -> None:
    assert torch.allclose(_write("tool", CONFIDENT)[2], torch.tensor(4.0), atol=ATOL)
    assert torch.allclose(_write("tool", DOUBTFUL)[2], torch.tensor(-6.0), atol=ATOL)
    assert torch.allclose(_write("tool", UNSURE), torch.zeros(HIDDEN), atol=ATOL)
