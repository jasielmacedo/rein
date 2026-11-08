from rein.compiler.tool_binary import ANY_SUFFIX, EACH_SEPARATOR, derived_bases, derived_questions, with_tool_questions
from rein.genview.keys import rule_key
from rein.schema import Message, Question, State, TrainingRecord
from rein.schema.constants import MAX_OPTION_TEXT_CHARS

TOOLS = Question(
    type="choice",
    instructions="Which tool does the request need?",
    criteria={"ring_bell": "ring the signal bell", "give_item": None, "none": None},
    label="ring_bell",
)


def _record() -> TrainingRecord:
    return TrainingRecord(
        id="r",
        source="synthetic",
        scenario_id="s",
        split="train",
        state=State(conversation=(Message(role="user", text="ring it"),)),
        questions={"tool": TOOLS},
    )


def test_a_tool_choice_becomes_yes_no_questions_with_the_same_answer() -> None:
    derived = derived_questions("tool", TOOLS)
    assert derived[f"tool{ANY_SUFFIX}"].label is True
    assert derived[f"tool{EACH_SEPARATOR}ring_bell"].label is True
    assert derived[f"tool{EACH_SEPARATOR}give_item"].label is False
    assert f"tool{EACH_SEPARATOR}none" not in derived
    quiet = derived_questions("tool", TOOLS.model_copy(update={"label": "none"}))
    assert quiet[f"tool{ANY_SUFFIX}"].label is False


def test_records_gain_the_derived_questions_and_their_bases() -> None:
    records = with_tool_questions({"r": _record()}, {"r": "tool"})
    assert len(records["r"].questions) == 1 + 1 + 2
    bases = derived_bases(records, {rule_key(TOOLS): "You can ring the bell and give items."})
    assert len(bases) == 3 and set(bases.values()) == {"You can ring the bell and give items."}
    untouched = with_tool_questions({"r": _record()}, {})
    assert list(untouched["r"].questions) == ["tool"]


def test_each_sheet_asks_its_own_any_tool_question() -> None:
    gate = TOOLS.model_copy(update={"criteria": {"open_gate": "open the gate", "none": "just talk"}, "label": "none"})
    bell = derived_questions("tool", TOOLS)[f"tool{ANY_SUFFIX}"]
    other = derived_questions("tool", gate)[f"tool{ANY_SUFFIX}"]
    assert rule_key(bell) != rule_key(other)
    assert "ring_bell: ring the signal bell" in bell.option_texts()[1]
    assert "give_item" in bell.option_texts()[1]
    assert "just talk" in other.option_texts()[0]


def test_a_long_any_tool_question_still_fits_one_option_text() -> None:
    wordy = {f"tool_{index}": "x" * MAX_OPTION_TEXT_CHARS for index in range(3)} | {"none": None}
    question = derived_questions("tool", TOOLS.model_copy(update={"criteria": wordy, "label": "none"}))
    described = question[f"tool{ANY_SUFFIX}"].option_texts()[1]
    assert all(f"tool_{index}" in described for index in range(3))
