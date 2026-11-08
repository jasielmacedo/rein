from dataclasses import replace

import pytest

from rein.compiler.evaluate import diagnose
from rein.compiler.firing import firing, firing_curve, recall_at_precision
from rein.compiler.tool_binary import ANY_SUFFIX
from rein.eval.metrics import PredictionRow


def _row(question: str, yes_probability: float, label_yes: bool, target: bool, form: str = "direct") -> PredictionRow:
    return PredictionRow(
        record_id=f"r-{question}-{yes_probability}",
        question_id=question,
        question_type="noul",
        option_keys=("no", "yes"),
        probs=(1 - yes_probability, yes_probability),
        label_index=int(label_yes),
        tags={"target_rule": question if target else "other", "category": "npc_secret", "case_form": form},
    )


def test_firing_at_the_gate_counts_precision_and_recall() -> None:
    rows = [_row("a", 0.95, True, True), _row("b", 0.9, False, True), _row("c", 0.4, True, True)]
    at_gate = firing(rows, 0.85)
    assert at_gate["fired"] == 2 and at_gate["positives"] == 2
    assert at_gate["precision"] == pytest.approx(0.5) and at_gate["recall"] == pytest.approx(0.5)
    assert recall_at_precision(firing_curve(rows), floor=0.9) is None


def test_diagnosis_separates_the_rule_in_play_from_the_others() -> None:
    rows = [_row(f"t{i}", 0.9, True, True) for i in range(25)] + [_row(f"o{i}", 0.1, False, False) for i in range(25)]
    report = diagnose(rows)
    assert report["target_rule"]["n"] == 25 and report["other_rules"]["n"] == 25
    assert report["target_category:npc_secret"]["accuracy"] == pytest.approx(1.0)
    assert "target_case_form:direct" in report


def test_a_choice_rule_fires_on_its_best_option_other_than_none() -> None:
    row = PredictionRow("r", "tool", "choice", ("ring_bell", "give_item", "none"), (0.7, 0.2, 0.1), 0)
    wrong = PredictionRow("s", "tool", "choice", ("ring_bell", "give_item", "none"), (0.1, 0.8, 0.1), 0)
    quiet = PredictionRow("t", "tool", "choice", ("ring_bell", "give_item", "none"), (0.05, 0.05, 0.9), 2)
    at = firing([row, wrong, quiet], 0.6)
    assert at["fired"] == 2 and at["positives"] == 2
    assert at["precision"] == pytest.approx(0.5) and at["recall"] == pytest.approx(0.5)
    curve = firing_curve([row, quiet])
    assert recall_at_precision(curve)["recall"] == pytest.approx(1.0)


def test_yes_no_versions_of_tool_rules_stay_out_of_the_original_slices() -> None:
    original = [_row(f"t{i}", 0.9, True, True) for i in range(25)]
    derived = [
        PredictionRow(f"d{i}", f"t{i}{ANY_SUFFIX}", "noul", ("no", "yes"), (0.2, 0.8), 1, {"target_rule": f"t{i}"})
        for i in range(25)
    ]
    report = diagnose(original + derived)
    assert report["all"]["n"] == 25 and report["type:noul"]["n"] == 25
    assert report["tool:any_tool"]["n"] == 25 and "tool:each_tool" not in report
    elsewhere = [replace(row, tags={"target_rule": "lore"}) for row in derived]
    assert diagnose(original + elsewhere)["tool:any_tool_elsewhere"]["n"] == 25
