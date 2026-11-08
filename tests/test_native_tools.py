import pytest

from rein.compiler.native_tools import summarize, tool_question
from rein.schema import Message, Question, State, TrainingRecord

PROMPT = '<tools>{"name": "ring_bell"} {"name": "give_item"}</tools> ...'


def _record(label: str) -> TrainingRecord:
    tools = Question(
        type="choice",
        instructions="Which tool?",
        criteria={"ring_bell": None, "give_item": None, "none": None},
        label=label,
    )
    other = Question(type="choice", instructions="Which team?", criteria={"billing": None, "none": None}, label="none")
    return TrainingRecord(
        id="r",
        source="synthetic",
        scenario_id="s",
        split="development",
        state=State(conversation=(Message(role="user", text="ring it"),)),
        questions={"team": other, "tool": tools},
    )


def test_the_tool_rule_is_the_choice_whose_options_are_offered_tools() -> None:
    found = tool_question(_record("ring_bell"), PROMPT)
    assert found is not None and found[0] == "tool"
    assert tool_question(_record("ring_bell"), "<tools></tools>") is None


def test_summary_counts_calls_names_and_unneeded_calls() -> None:
    cases = [
        {"label": "ring_bell", "natural_called": True, "natural_name": "ring_bell", "forced_name": "ring_bell"},
        {"label": "give_item", "natural_called": False, "natural_name": None, "forced_name": "ring_bell"},
        {"label": "none", "natural_called": True, "natural_name": "give_item", "forced_name": "give_item"},
        {"label": "none", "natural_called": False, "natural_name": None, "forced_name": "ring_bell"},
    ]
    summary = summarize(cases)
    assert summary["called_when_needed"] == pytest.approx(0.5)
    assert summary["right_tool_when_called"] == pytest.approx(1.0)
    assert summary["called_when_not_needed"] == pytest.approx(0.5)
    assert summary["right_tool_after_forced_opening"] == pytest.approx(0.5)
