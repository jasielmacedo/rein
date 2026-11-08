import random

from rein.public.common import NONE_OPTION, class_names_from_card, intent_question, render_value

CARD = """---
dataset_info:
  features:
  - name: text
    dtype: string
  - name: label
    dtype:
      class_label:
        names:
          '0': activate_my_card
          '1': age_limit
          '2': atm_support
---
# card
"""
INTENTS = [f"intent_{i}" for i in range(40)]


def test_render_value_matches_kev_layout() -> None:
    assert render_value({"policy": "max 10%", "case": {"units": 3, "tags": ["a", "b"]}}) == (
        "policy: max 10%\ncase:\n  units: 3\n  tags:\n    - a\n    - b"
    )


def test_class_names_follow_label_ids() -> None:
    assert class_names_from_card(CARD) == ["activate_my_card", "age_limit", "atm_support"]


def test_intent_question_keeps_the_answer_or_falls_back_to_none() -> None:
    rng = random.Random(0)
    for _ in range(200):
        question = intent_question("intent_7", INTENTS, rng)
        assert isinstance(question.criteria, dict)
        assert NONE_OPTION in question.criteria
        assert question.label in question.criteria
        assert question.label in {"intent_7", NONE_OPTION}
        assert (question.label == NONE_OPTION) == ("intent_7" not in question.criteria)
    assert intent_question(None, INTENTS, rng).label == NONE_OPTION
