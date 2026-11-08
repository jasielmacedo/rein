from dataclasses import dataclass

from rein.control.plan import Probabilities
from rein.eval.soft_agreement import compare, summarize
from rein.schema import DecisionRequest


@dataclass
class FixedBackend:
    answers: Probabilities

    def decide(self, request: DecisionRequest) -> Probabilities:
        return self.answers


def test_compare_reports_each_question_gap_and_agreement(roleplay_request: DecisionRequest) -> None:
    reference = FixedBackend({"q1": {"A": 0.9, "B": 0.1}, "q2": {"A": 0.4, "B": 0.6}})
    candidate = FixedBackend({"q1": {"A": 0.85, "B": 0.15}, "q2": {"A": 0.55, "B": 0.45}})
    rows = compare(reference, candidate, [roleplay_request])
    assert [row["question"] for row in rows] == ["q1", "q2"]
    assert rows[0]["max_abs_diff"] == 0.05
    assert rows[0]["same_argmax"] is True
    assert rows[1]["max_abs_diff"] == 0.15
    assert rows[1]["same_argmax"] is False
    summary = summarize(rows)
    assert summary["questions"] == 2
    assert summary["same_argmax"] == 1
    assert summary["worst_max_abs_diff"] == 0.15
    assert summary["mean_max_abs_diff"] == 0.1
