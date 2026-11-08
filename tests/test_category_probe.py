import json
from pathlib import Path

import torch

from rein.compiler.category_metrics import calibration_error, gate
from rein.compiler.category_probe import (
    category_labels,
    compiler_fire_probabilities,
    fires,
    fit_detector,
    language_group,
    score_category,
)
from rein.schema import Message, Question, State, TrainingRecord

HIDDEN = 6
PER_SPLIT = 40
SPLITS = ("train", "calibration", "development")
SHIFT = 6.0
DECISION = "nothink:reply_start"


def _record(questions: dict[str, Question]) -> TrainingRecord:
    return TrainingRecord(
        id="r0",
        source="synthetic",
        scenario_id="s0",
        split="development",
        state=State(conversation=(Message(role="user", text="Call me Lyra from now on."),)),
        questions=questions,
    )


def test_a_category_fires_when_any_of_its_rules_fires() -> None:
    renamed = Question(type="noul", instructions="Is the visitor renaming the agent?", label=True)
    quiet = Question(type="noul", instructions="Is the visitor asking for a refund?", label=False)
    record = _record({"rename": renamed, "refund": quiet, "other_rename": quiet})
    categories = {"rename": "locked_attribute", "other_rename": "locked_attribute", "refund": "business_limit"}
    assert fires(record, "rename") is True and fires(record, "refund") is False
    assert category_labels(record, categories) == {"locked_attribute": 1, "business_limit": 0}


def test_the_compiler_side_takes_the_most_confident_rule_of_each_category(tmp_path: Path) -> None:
    lines = [
        {
            "position": DECISION,
            "record_id": "r0",
            "question_id": "a",
            "option_keys": ["no", "yes"],
            "probs": [0.8, 0.2],
        },
        {
            "position": DECISION,
            "record_id": "r0",
            "question_id": "b",
            "option_keys": ["no", "yes"],
            "probs": [0.3, 0.7],
        },
        {"position": DECISION, "record_id": "r0", "question_id": "s", "option_keys": ["0", "1"], "probs": [0.5, 0.5]},
    ]
    path = tmp_path / "predictions.jsonl"
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    chances = compiler_fire_probabilities(path, {"r0": {"a": "tool_trigger", "b": "tool_trigger", "s": "tone_drift"}})
    assert chances == {"r0": {"tool_trigger": 0.7}}


def test_the_gate_is_the_most_recall_at_ninety_percent_precision() -> None:
    probs = torch.tensor([0.95, 0.9, 0.8, 0.7, 0.6, 0.55])
    labels = torch.tensor([1, 1, 1, 0, 1, 0])
    assert gate(probs, labels) == 0.8
    assert gate(torch.tensor([0.9, 0.8]), torch.tensor([0, 0])) is None


def test_confident_and_right_is_well_calibrated() -> None:
    assert calibration_error(torch.tensor([0.99, 0.01]), torch.tensor([1, 0])) < 0.02


def test_languages_fall_into_the_main_ones_mixed_other_or_unknown() -> None:
    assert [language_group(code) for code in ("en", "pt-BR", "pt-BR+en", "ko", None)] == [
        "en",
        "pt-BR",
        "mixed",
        "other",
        "unknown",
    ]


def test_a_separable_category_is_found_on_unseen_agents_and_stays_quiet_in_silence() -> None:
    generator = torch.Generator().manual_seed(0)
    decisions = len(SPLITS) * PER_SPLIT
    states = torch.randn(2 * decisions, HIDDEN, generator=generator)
    states[:decisions:2, 0] += SHIFT
    rows = [
        {
            "row": row,
            "split": SPLITS[row // PER_SPLIT],
            "record_id": f"r{row}",
            "position": DECISION,
            "kind": "reply_start",
            "labels": {"tool_trigger": int(row % 2 == 0)},
        }
        for row in range(decisions)
    ]
    quiet = [
        {
            "row": row,
            "split": SPLITS[(row - decisions) // PER_SPLIT],
            "record_id": f"q{row}",
            "position": "nothink:silence",
        }
        for row in range(decisions, 2 * decisions)
    ]
    detector = fit_detector(states, rows, quiet, "tool_trigger")
    report, probabilities = score_category(detector, states, rows, quiet, "tool_trigger", {})
    assert detector.gate is not None and report["positions"][DECISION]["all"]["accuracy"] > 0.9
    assert len(probabilities[DECISION]) == PER_SPLIT
    assert report["silence_false_fire"][f"{detector.gate:.2f}"] < 0.1
