import random

import pytest
import torch
from transformers import GenerationMixin

from rein.eval.report import PredictionRow, build_report
from rein.model.decider import Decider
from rein.schema import DecisionRequest, Question
from tests.devices import PROB_ATOL

SHUFFLE_SEEDS = (1, 2, 3)


def test_1_forward_returns_questions_by_max_options_with_mask(
    decider: Decider, roleplay_request: DecisionRequest
) -> None:
    out = decider.decide(roleplay_request)
    option_counts = [len(question.option_keys()) for question in roleplay_request.questions.values()]
    assert out.probs.shape == (len(option_counts), max(option_counts))
    assert out.mask.shape == out.probs.shape
    for row, count in enumerate(option_counts):
        assert out.mask[row, :count].all()
        assert not out.mask[row, count:].any()
    assert out.question_ids == list(roleplay_request.questions)


def test_2_probabilities_of_each_question_sum_to_one(decider: Decider, roleplay_request: DecisionRequest) -> None:
    out = decider.decide(roleplay_request)
    torch.testing.assert_close(out.probs.sum(-1), torch.ones(out.probs.shape[0]), atol=PROB_ATOL, rtol=0)
    assert torch.all(out.probs[~out.mask] == 0)


def test_3_decision_path_never_uses_lm_head_or_generate(
    decider: Decider, roleplay_request: DecisionRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert not any("lm_head" in name for name, _ in decider.named_modules())

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("generate() must never run in the decision path")

    monkeypatch.setattr(GenerationMixin, "generate", forbidden)
    decider.decide(roleplay_request)


@pytest.mark.parametrize("seed", SHUFFLE_SEEDS)
def test_4_option_order_does_not_change_probabilities(
    decider: Decider, roleplay_request: DecisionRequest, seed: int
) -> None:
    rng = random.Random(seed)
    shuffled: dict[str, Question] = {}
    for qid, question in roleplay_request.questions.items():
        if question.type == "choice":
            assert isinstance(question.criteria, dict)
            items = list(question.criteria.items())
            rng.shuffle(items)
            shuffled[qid] = question.model_copy(update={"criteria": dict(items)})
        else:
            shuffled[qid] = question
    base = decider.decide(roleplay_request).by_key()
    other = decider.decide(roleplay_request.model_copy(update={"questions": shuffled})).by_key()
    for qid, probs in base.items():
        for key, value in probs.items():
            assert other[qid][key] == pytest.approx(value, abs=PROB_ATOL)


@pytest.mark.parametrize("seed", SHUFFLE_SEEDS)
def test_4_row_order_is_irrelevant_for_every_question_type(
    decider: Decider, roleplay_request: DecisionRequest, seed: int
) -> None:
    encoded = decider.encoder.encode(roleplay_request)
    permuted = encoded.with_shuffled_options(random.Random(seed))
    base = decider.decide_encoded([encoded])
    other = decider.decide_encoded([permuted])
    for question_a, question_b, row_a, row_b in zip(
        encoded.questions, permuted.questions, base.probs, other.probs, strict=True
    ):
        lookup = dict(zip(question_b.option_keys, row_b.tolist(), strict=False))
        for key, value in zip(question_a.option_keys, row_a.tolist(), strict=False):
            assert lookup[key] == pytest.approx(value, abs=PROB_ATOL)


def _rows(model_bias: float, count: int = 40) -> list[PredictionRow]:
    rows = []
    for i in range(count):
        correct = i % 4 != 0
        probs = (0.8 + model_bias, 0.2 - model_bias) if correct else (0.3, 0.7)
        rows.append(
            PredictionRow(
                record_id=f"r{i}",
                question_id="lock_name",
                question_type="noul",
                option_keys=("yes", "no"),
                probs=probs,
                label_index=0,
                tags={"split": "test", "unseen_rule": True},
            )
        )
    return rows


def test_5_report_has_unseen_rule_metrics_and_ece() -> None:
    report = build_report({"rein": _rows(0.0)})
    cell = report.cell("unseen_rules", "rein")
    assert {"n", "accuracy", "ece", "brier", "nll"} <= set(cell)
    assert cell["n"] == 40


def test_6_report_compares_tinyjev_on_the_same_items() -> None:
    report = build_report({"rein": _rows(0.0), "tinyjev-4b": _rows(0.05)})
    assert report.models == ("rein", "tinyjev-4b")
    delta = report.paired_delta("unseen_rules", "rein", "tinyjev-4b", metric="accuracy")
    assert delta.low <= delta.estimate <= delta.high
    with pytest.raises(ValueError, match="same items"):
        build_report({"rein": _rows(0.0), "tinyjev-4b": _rows(0.0, count=39)})
