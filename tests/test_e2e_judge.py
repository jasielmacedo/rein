from pathlib import Path

import pytest

from rein.sheets.e2e_judge import attacks, summarize
from rein.sheets.model import expected_tool, load_sheet


def test_the_script_says_which_tool_a_turn_needs_and_which_rules_it_attacks() -> None:
    sheet = load_sheet(Path("configs/sheets/ilsa.yaml"))
    both = {"answers": {"name_lock": "yes", "quartermaster_action": "issue_item"}}
    quiet = {"answers": {}}
    assert expected_tool(sheet, both) == "issue_item" and expected_tool(sheet, quiet) is None
    assert attacks(sheet, both) == ["name_lock"] and attacks(sheet, quiet) == []


def test_each_rate_counts_only_the_turns_where_it_applies() -> None:
    turns = [
        {"needed_called": True, "right_tool": False, "anything_wrong": True},
        {"unnecessary_call": False, "anything_wrong": False},
    ]
    summary = summarize(turns)
    assert summary["needed_called"] == 1.0 and summary["right_tool"] == 0.0
    assert summary["unnecessary_call"] == 0.0 and summary["anything_wrong"] == pytest.approx(0.5)
