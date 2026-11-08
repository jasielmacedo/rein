from rein.module.firstturn import first_turn, is_first
from rein.schema import Message, Question, State, TrainingRecord

QUESTION = Question(type="noul", instructions="Did the user ask for a photo?", label=True)


def _record(*roles: str) -> TrainingRecord:
    conversation = tuple(Message(role=role, text=f"{role} {n}") for n, role in enumerate(roles))
    return TrainingRecord(
        id="r1",
        source="synthetic",
        scenario_id="s1",
        split="train",
        state=State(conversation=conversation),
        questions={"photo": QUESTION},
    )


def test_a_conversation_is_cut_after_its_first_user_message() -> None:
    cut = first_turn(_record("narrator", "user", "assistant", "user"))
    assert cut is not None
    assert [message.role for message in cut.state.conversation] == ["narrator", "user"]
    assert cut.id == "r1"


def test_a_conversation_that_already_ends_at_its_first_user_message_is_left_out() -> None:
    assert first_turn(_record("user")) is None
    assert first_turn(_record("narrator")) is None


def test_first_turn_views_are_told_apart_by_their_suffix() -> None:
    assert is_first("r1|nothink|first")
    assert not is_first("r1|nothink")
