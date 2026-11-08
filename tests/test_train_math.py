import math

import pytest
import torch

from rein.eval.metrics import PredictionRow, expected_calibration_error
from rein.train.calibrate import fit_temperature
from rein.train.loss import decision_loss

TRUE_TEMPERATURE = 2.0
CALIBRATION_SAMPLES = 4000
OPTIONS = 4


def test_loss_ignores_padded_options_and_weights_records_equally() -> None:
    logits = torch.tensor([[2.0, 0.0, 99.0], [0.0, 1.0, -99.0], [0.5, 0.5, 0.0]])
    mask = torch.tensor([[True, True, False], [True, True, False], [True, True, True]])
    targets = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    owner = torch.tensor([0, 0, 1])
    parts = decision_loss(logits, mask, targets, owner, brier_weight=0.0)
    first = -torch.log_softmax(torch.tensor([2.0, 0.0]), -1)[0]
    second = -torch.log_softmax(torch.tensor([0.0, 1.0]), -1)[1]
    third = -torch.log_softmax(torch.tensor([0.5, 0.5, 0.0]), -1)[2]
    expected = ((first + second) / 2 + third) / 2
    assert parts.ce.item() == pytest.approx(expected.item(), rel=1e-5)
    assert parts.total.item() == pytest.approx(parts.ce.item())


def test_brier_term_is_added_with_its_weight() -> None:
    logits = torch.tensor([[0.0, 0.0]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    targets = torch.tensor([[1.0, 0.0]])
    parts = decision_loss(logits, mask, targets, torch.tensor([0]), brier_weight=0.5)
    assert parts.brier.item() == pytest.approx(0.5)
    assert parts.total.item() == pytest.approx(parts.ce.item() + 0.25)


def test_fit_temperature_recovers_a_known_temperature() -> None:
    generator = torch.Generator().manual_seed(0)
    logits = torch.randn(CALIBRATION_SAMPLES, OPTIONS, generator=generator) * 3
    labels = torch.multinomial(torch.softmax(logits / TRUE_TEMPERATURE, -1), 1, generator=generator).squeeze(-1)
    mask = torch.ones_like(logits, dtype=torch.bool)
    assert fit_temperature(logits, mask, labels) == pytest.approx(TRUE_TEMPERATURE, rel=0.1)


def test_ece_matches_a_hand_computed_case() -> None:
    rows = [
        PredictionRow("a", "q", "noul", ("no", "yes"), (0.1, 0.9), 1, {}),
        PredictionRow("b", "q", "noul", ("no", "yes"), (0.1, 0.9), 0, {}),
        PredictionRow("c", "q", "noul", ("no", "yes"), (0.4, 0.6), 1, {}),
        PredictionRow("d", "q", "noul", ("no", "yes"), (0.4, 0.6), 1, {}),
    ]
    ece = expected_calibration_error(rows, bins=10)
    assert ece == pytest.approx((2 * abs(0.5 - 0.9) + 2 * abs(1.0 - 0.6)) / 4)
    assert math.isfinite(ece)
