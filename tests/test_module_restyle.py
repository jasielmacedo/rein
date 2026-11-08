from rein.module.restyle import conversation_text, restyled_record
from rein.schema import Message, Question, State, TrainingRecord

QUESTION = Question(type="noul", instructions="Did the user ask for a photo?", label=True)


def _record() -> TrainingRecord:
    conversation = (
        Message(role="user", text="hello there"),
        Message(role="assistant", text="hi"),
        Message(role="narrator", text="*the bell rings*"),
        Message(role="user", text="send me a photo please"),
    )
    return TrainingRecord(
        id="r1",
        source="synthetic",
        scenario_id="s1",
        split="train",
        state=State(conversation=conversation),
        questions={"photo": QUESTION},
    )


def test_the_user_messages_are_replaced_in_order_and_the_rest_kept() -> None:
    restyled = restyled_record(_record(), ["hey there friend", "could you send me a picture please"])
    assert restyled is not None
    texts = [message.text for message in restyled.state.conversation]
    assert texts == ["hey there friend", "hi", "*the bell rings*", "could you send me a picture please"]
    assert restyled.id == "r1"


def test_a_rewrite_that_does_not_line_up_is_dropped() -> None:
    assert restyled_record(_record(), ["only one"]) is None
    assert restyled_record(_record(), ["yo", "  "]) is None


def test_the_teacher_sees_every_turn_with_its_role() -> None:
    assert conversation_text(_record()).splitlines()[2] == "narrator: *the bell rings*"
