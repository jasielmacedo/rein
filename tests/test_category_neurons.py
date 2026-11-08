from pathlib import Path

import torch

from rein.compiler.category_neurons import (
    FORCE,
    STEER,
    VETO,
    action_label,
    channel_of,
    labelled,
    load_detectors,
    save,
)
from rein.compiler.category_probe import fit_labelled
from rein.neurons.units import threshold_gate

HIDDEN = 8
PER_SPLIT = 60
SPLITS = ("train", "calibration", "development")
SHIFT = 5.0
NOISE = 1e-4
FLOAT_TOLERANCE = 1e-3


def test_steer_lasts_the_reply_while_force_and_veto_act_only_where_it_opens() -> None:
    assert action_label(STEER, "reply_start", 1) == 1 and action_label(STEER, "reply", 1) == 1
    assert action_label(STEER, "reply", None) == 0 and action_label(STEER, "reasoning", 1) == 0
    assert action_label(FORCE, "answer_start", 1) == 1 and action_label(FORCE, "reply", 1) == 0
    assert action_label(VETO, "reply_start", 0) == 1 and action_label(VETO, "answer_start", 1) == 0
    assert action_label(VETO, "reply", 0) == 0
    assert action_label(VETO, "reply_start", None) == 0 and action_label(VETO, "silence", 0) == 0


def _rows() -> tuple[torch.Tensor, list[dict], list[dict]]:
    generator = torch.Generator().manual_seed(0)
    count = len(SPLITS) * PER_SPLIT
    states = torch.randn(2 * count, HIDDEN, generator=generator)
    states[:, 0] = 1.0 + NOISE * torch.randn(2 * count, generator=generator)
    rows = []
    for row in range(count):
        fires = row % 2
        states[row, 1] += SHIFT * fires
        kind = "reply_start" if row % 3 else "reply"
        rows.append(
            {
                "row": row,
                "split": SPLITS[row // PER_SPLIT],
                "record_id": f"r{row}",
                "position": f"nothink:{kind}",
                "kind": kind,
                "labels": {"locked_attribute": fires},
            }
        )
    quiet = [
        {"row": row, "split": SPLITS[(row - count) // PER_SPLIT], "record_id": f"q{row}", "position": "nothink:silence"}
        for row in range(count, 2 * count)
    ]
    return states, rows, quiet


def test_the_fitted_read_is_exactly_the_neurons_decision() -> None:
    states, rows, quiet = _rows()
    channel = channel_of(states, [row for row in [*rows, *quiet] if row["split"] == "train"])
    train = labelled(rows, quiet, "locked_attribute", STEER, "train")
    calibration = labelled(rows, quiet, "locked_attribute", STEER, "calibration")
    detector = fit_labelled(states, train, calibration, channel)
    assert detector.gate is not None
    direction, offset = detector.neuron_form()
    gate_row = threshold_gate(direction, offset, channel, detector.temperature, detector.gate)
    development, labels = labelled(rows, quiet, "locked_attribute", STEER, "development")
    picked = states[[row["row"] for row in development]]
    by_probe = detector.probabilities(picked) >= detector.gate
    by_neuron = picked @ gate_row >= 0
    assert torch.equal(by_probe, by_neuron)
    assert float((by_probe.long() == labels).float().mean()) > 0.9


def test_saved_neurons_read_back_as_the_same_detector(tmp_path: Path) -> None:
    states, rows, quiet = _rows()
    channel = channel_of(states, [row for row in [*rows, *quiet] if row["split"] == "train"])
    train = labelled(rows, quiet, "locked_attribute", STEER, "train")
    calibration = labelled(rows, quiet, "locked_attribute", STEER, "calibration")
    detector = fit_labelled(states, train, calibration, channel)
    save(tmp_path, 20, channel, {"locked_attribute:steer": detector}, {})
    loaded = load_detectors(tmp_path)["locked_attribute:steer"]
    assert loaded.gate == detector.gate
    assert detector.gate is not None
    assert torch.equal(loaded.probabilities(states) >= loaded.gate, detector.probabilities(states) >= detector.gate)
    assert torch.allclose(loaded.probabilities(states), detector.probabilities(states), atol=FLOAT_TOLERANCE)
