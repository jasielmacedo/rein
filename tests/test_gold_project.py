from typing import Any

import pytest
from pydantic import ValidationError

from rein.genview.keys import rule_key
from rein.gold.prepare import tool_rules
from rein.gold.project import GoldProject, build_records, gold_index, sheet_bases

SHEET: dict[str, Any] = {
    "id": "maren",
    "npc": "Maren Vell",
    "persona": "You are Maren Vell, the lighthouse keeper.",
    "scene": "Dusk, fog rising.",
    "language": "en",
    "tools": [{"name": "ring_signal_bell", "description": "Ring the warning bell."}],
    "rules": [
        {
            "id": "name_change",
            "question": "Is the visitor renaming the keeper?",
            "kind": "yes_no",
            "base": "You answer only to Maren Vell.",
        },
        {
            "id": "keeper_action",
            "question": "Which action does the message call for?",
            "kind": "choice",
            "category": "tool_trigger",
            "options": [
                {"key": "ring_signal_bell", "when": "a ship is in danger"},
                {"key": "none", "when": "no action"},
            ],
        },
        {
            "id": "narrated_action",
            "question": "Does the reply narrate an action without its tool call?",
            "kind": "choice",
            "looks_at": "npc_reply",
            "options": [{"key": "ring_signal_bell"}, {"key": "none"}],
        },
    ],
}


def _message(speaker: str, text: str, **extra: Any) -> dict[str, Any]:
    return {"from": speaker, "text": text, **extra}


def _project(*messages: dict[str, Any], sheet: dict[str, Any] = SHEET) -> GoldProject:
    conversation = {"id": "c1", "sheet": sheet["id"], "language": "pt-BR", "messages": list(messages)}
    return GoldProject.model_validate(
        {"format": "rein-gold-project", "version": 1, "sheets": [sheet], "conversations": [conversation]}
    )


def test_each_player_turn_and_npc_reply_becomes_a_record_with_unmarked_rules_off() -> None:
    project = _project(
        _message("player", "Evening, keeper."),
        _message("npc", "*She rings the bell.*", answers={"narrated_action": "ring_signal_bell"}),
        _message("narration", "*Everyone now calls her Lyra.*", answers={"name_change": "yes"}),
    )
    built = build_records(project)
    assert not built.problems and len(built.records) == 3
    quiet, reply, renamed = built.records
    assert quiet.questions["name_change"].label is False and quiet.questions["keeper_action"].label == "none"
    assert quiet.meta["target_rule"] is None and quiet.meta["case_form"] == "neutral"
    assert reply.state.draft is not None and reply.state.draft.text == "*She rings the bell.*"
    assert [m.role for m in reply.state.conversation] == ["user"] and list(reply.questions) == ["narrated_action"]
    assert renamed.state.conversation[-1].role == "narrator" and renamed.meta["target_rule"] == "name_change"
    assert renamed.state.context == "You are Maren Vell, the lighthouse keeper.\n\nDusk, fog rising."


def test_a_hard_negative_names_the_rule_it_tests_and_npc_tool_calls_stay_in_the_history() -> None:
    project = _project(
        _message("npc", "Stand back.", tool_calls=["ring_signal_bell"]),
        _message("player", "Who are you, really?", tests="name_change", form="hard_negative"),
    )
    record = build_records(project).records[-1]
    assert record.meta["target_rule"] == "name_change" and record.meta["case_form"] == "hard_negative"
    assert record.state.conversation[0].tool_calls[0].name == "ring_signal_bell"


def test_a_rule_without_a_null_option_must_be_marked() -> None:
    sheet = {
        **SHEET,
        "rules": [
            {
                "id": "route",
                "question": "Where does the turn go?",
                "kind": "choice",
                "options": [{"key": "sales"}, {"key": "support"}],
            }
        ],
    }
    built = build_records(_project(_message("player", "Hi"), sheet=sheet))
    assert not built.records and "no answer for route" in built.problems[0]


def test_the_index_finds_each_sheet_and_the_author_bases_are_kept() -> None:
    project = _project(
        _message("player", "Ring it!", answers={"keeper_action": "ring_signal_bell"}), _message("npc", "Aye.")
    )
    built = build_records(project)
    index = gold_index(project)
    assert all(index.lookup(record) is not None for record in built.records)
    info = index.lookup(built.records[0])
    assert (
        info is not None and info.tools == ("ring_signal_bell",) and info.categories["keeper_action"] == "tool_trigger"
    )
    bases = sheet_bases(project)
    assert bases == {rule_key(project.sheets[0].rules[0].as_question()): "You answer only to Maren Vell."}
    assert tool_rules(project, built.records) == {built.records[0].id: "keeper_action"}


def test_an_invalid_rule_is_refused_when_the_project_loads() -> None:
    sheet = {**SHEET, "rules": [{"id": "empty", "question": "Which one?", "kind": "choice", "options": []}]}
    with pytest.raises(ValidationError):
        _project(sheet=sheet)
