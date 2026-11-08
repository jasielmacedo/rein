import pytest
from pydantic import ValidationError

from rein.schema import DecisionRequest, Message, Question, State, TrainingRecord
from rein.schema import constants as c


def test_noul_options_and_labels() -> None:
    question = Question(
        type="noul",
        instructions="Is the user renaming the character?",
        criteria={"true": "they assert a new name", "false": "anything else"},
        label=True,
    )
    assert question.option_keys() == c.NOUL_OPTION_KEYS
    assert question.option_texts() == ("no: anything else", "yes: they assert a new name")
    assert question.label_index() == 1


def test_choice_label_must_be_an_option() -> None:
    criteria = {"send_photo": "share an image", "none": None}
    assert Question(type="choice", instructions="Which tool?", criteria=criteria, label="none").label_index() == 1
    with pytest.raises(ValidationError):
        Question(type="choice", instructions="Which tool?", criteria=criteria, label="web_search")


def test_score_levels_and_label_range() -> None:
    question = Question(type="score", instructions="How off-tone?", criteria=["on tone", "slightly", "far"], label=2)
    assert question.option_keys() == ("0", "1", "2")
    assert question.option_texts()[0] == "0: on tone"
    with pytest.raises(ValidationError):
        Question(type="score", instructions="How off-tone?", criteria=["a", "b"], label=2)
    with pytest.raises(ValidationError):
        Question(type="score", instructions="How off-tone?", criteria=["a", "b"], label=True)


def test_target_must_be_a_distribution_over_options() -> None:
    question = Question(type="noul", instructions="Ambiguous?", target={"yes": 0.6, "no": 0.4})
    assert question.target_vector() == (0.4, 0.6)
    with pytest.raises(ValidationError):
        Question(type="noul", instructions="Ambiguous?", target={"yes": 0.6, "no": 0.6})
    with pytest.raises(ValidationError):
        Question(type="noul", instructions="Ambiguous?", target={"maybe": 1.0})


def test_noul_criteria_keys_are_restricted() -> None:
    with pytest.raises(ValidationError):
        Question(type="noul", instructions="x?", criteria={"maybe": "?"})


def test_request_rejects_bad_ids_and_extra_fields() -> None:
    state = State(conversation=(Message(role="user", text="*waves* hi"),))
    question = Question(type="noul", instructions="Greeting?")
    assert DecisionRequest(state=state, questions={"greet": question})
    with pytest.raises(ValidationError):
        DecisionRequest(state=state, questions={"bad id!": question})
    with pytest.raises(ValidationError):
        Message(role="user", text="hi", extra="nope")  # type: ignore[call-arg]


def test_training_record_requires_supervision() -> None:
    state = State(conversation=(Message(role="user", text="hi"),))
    with pytest.raises(ValidationError):
        TrainingRecord(
            id="r1",
            source="synthetic",
            scenario_id="s1",
            split="train",
            state=state,
            questions={"greet": Question(type="noul", instructions="Greeting?")},
        )
