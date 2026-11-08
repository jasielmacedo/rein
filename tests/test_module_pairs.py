import argparse
import json
from pathlib import Path

from rein.module.material import ANCHORS_FILE, PAIRS_FILE, SUMMARY_FILE
from rein.module.pairs import (
    BROKE,
    KEPT,
    NEEDED_CALL,
    RULE,
    UNCLEAR,
    UNNEEDED_CALL,
    Situation,
    build,
    combined,
    pair_kind,
    tool_verdict,
)
from rein.sheets.steer import Verdict

TOOLS = ("check_inventory", "web_search")
RULE_TYPES = {"tool_trigger", "lore_consistency"}
CALL = "<tool_call>\n<function={name}>\n</function>\n</tool_call>"


def _place(needed: str | None, tools: tuple[str, ...] = TOOLS, quiet: bool = True) -> Situation:
    return Situation("r1", "rule", "tool_trigger", False, tools, needed, quiet)


def test_a_call_is_right_only_when_the_turn_needs_that_tool() -> None:
    assert tool_verdict("Sure, let me tell you.", _place(None), RULE_TYPES) == KEPT
    assert tool_verdict(CALL.format(name="web_search"), _place(None), RULE_TYPES) == BROKE
    assert tool_verdict(CALL.format(name="web_search"), _place("web_search"), RULE_TYPES) == KEPT
    assert tool_verdict(CALL.format(name="check_inventory"), _place("web_search"), RULE_TYPES) == BROKE
    assert tool_verdict("I looked it up for you.", _place("web_search"), RULE_TYPES) == BROKE
    assert tool_verdict("Hello there.", _place(None, ()), RULE_TYPES) is None
    assert tool_verdict(CALL.format(name="web_search"), _place(None, quiet=False), RULE_TYPES) == UNCLEAR
    assert tool_verdict("I will hand you over.", _place(None, quiet=False), RULE_TYPES) == KEPT


def test_either_side_wrong_breaks_the_reply() -> None:
    kept, broke = Verdict(verdict="kept", natural=True), Verdict(verdict="broke", natural=True)
    assert combined(None, None, False) == KEPT
    assert combined(KEPT, broke, True) == BROKE
    assert combined(BROKE, kept, True) == BROKE
    assert combined(KEPT, kept, True) == KEPT
    assert combined(None, Verdict(verdict="kept", natural=False), True) == UNCLEAR
    assert combined(None, None, True) == UNCLEAR
    assert combined(UNCLEAR, None, True) == UNCLEAR


def test_the_wrong_reply_names_the_kind_of_mistake() -> None:
    assert pair_kind(BROKE, None) == UNNEEDED_CALL
    assert pair_kind(BROKE, "web_search") == NEEDED_CALL
    assert pair_kind(KEPT, "web_search") == RULE
    assert pair_kind(None, None) == RULE


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_pairs_come_from_views_with_both_outcomes_and_anchors_from_clean_ones(tmp_path: Path) -> None:
    replies = [[1], [2], [3]]
    samples = [
        {"view_id": f"v{n}", "record_id": f"r{n}", "split": split, "prompt": [9, n], "replies": replies}
        for n, split in enumerate(("train", "train", "train", "development"))
    ]
    situation = {"category": "lore_consistency", "needed_tool": None, "rule_fires": False, "quiet": True}
    calls = [BROKE, KEPT, KEPT]
    verdicts = [
        {"view_id": "v0", "split": "train", "situation": situation, "verdicts": [BROKE, KEPT, KEPT], "tools": calls},
        {"view_id": "v1", "split": "train", "situation": situation, "verdicts": [KEPT, KEPT, KEPT], "tools": calls},
        {"view_id": "v2", "split": "train", "situation": situation, "verdicts": [KEPT, UNCLEAR, KEPT], "tools": calls},
        {
            "view_id": "v3",
            "split": "development",
            "situation": situation,
            "verdicts": [KEPT, BROKE, KEPT],
            "tools": calls,
        },
    ]
    out = tmp_path / "built"
    build(
        argparse.Namespace(
            samples=_write(tmp_path / "samples.jsonl", samples),
            verdicts=_write(tmp_path / "verdicts.jsonl", verdicts),
            out=out,
        )
    )
    pairs = [json.loads(line) for line in (out / PAIRS_FILE).read_text(encoding="utf-8").splitlines()]
    anchors = [json.loads(line) for line in (out / ANCHORS_FILE).read_text(encoding="utf-8").splitlines()]
    assert [(pair["view_id"], pair["chosen"], pair["rejected"]) for pair in pairs] == [("v0", [2], [1])]
    assert pairs[0]["kind"] == UNNEEDED_CALL
    assert [(anchor["view_id"], anchor["reply"]) for anchor in anchors] == [("v1", [1])]
    summary = json.loads((out / SUMMARY_FILE).read_text(encoding="utf-8"))
    assert summary["right_share"]["development:lore_consistency"] == 2 / 3
    assert summary["by_situation"]["train:quiet"] == {BROKE: 1, KEPT: 7, UNCLEAR: 1}
