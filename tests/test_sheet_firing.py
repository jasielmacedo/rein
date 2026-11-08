from pathlib import Path

from rein.sheets.firing import decisions, wrong_fires
from rein.sheets.model import load_sheet

SHEET = load_sheet(Path("configs/sheets/maren.yaml"))
NEEDS_BELL = {"answers": {"keeper_action": "ring_signal_bell"}}
RENAMES = {"answers": {"name_change": "yes"}}
QUIET = {"answers": {}}


def test_a_force_without_need_and_a_veto_against_need_are_wrong() -> None:
    sure, doubtful = decisions(0.95, "tool"), decisions(0.1, "tool")
    assert wrong_fires({"keeper_action::any": sure}, SHEET, QUIET) == ["force:keeper_action"]
    assert wrong_fires({"keeper_action::any": doubtful}, SHEET, NEEDS_BELL) == ["veto:keeper_action"]
    assert wrong_fires({"keeper_action::any": sure}, SHEET, NEEDS_BELL) == []


def test_a_steer_is_wrong_when_it_disagrees_with_the_script() -> None:
    assert wrong_fires({"name_change": decisions(0.2, "steer")}, SHEET, RENAMES) == ["steer:name_change"]
    assert wrong_fires({"name_change": decisions(0.9, "steer")}, SHEET, QUIET) == ["steer:name_change"]
    assert wrong_fires({"narrated_action": decisions(0.9, "call")}, SHEET, QUIET) == ["call:narrated_action"]


def test_each_action_reads_its_own_positions() -> None:
    from rein.compiler.table import Item
    from rein.sheets.fit import labelled_items

    def item(question: str, kind: str, label: int) -> Item:
        keys = (
            ("no", "yes")
            if "::" in question or question == "name_change"
            else ("ring_signal_bell", "give_item", "none")
        )
        null = 0 if keys[0] == "no" else 2
        return Item(0, 0, question, "choice", keys, label, (0.0,) * len(keys), null, "r", "train", "nothink", kind)

    actions = {"name_change": "steer", "keeper_action::any": "tool", "narrated_action": "call"}
    items = [
        item("name_change", "reply", 1),
        item("keeper_action::any", "continuation_end", 1),
        item("narrated_action", "reply_start", 0),
        item("narrated_action", "reply", 1),
    ]
    kept = {(i.question_id, i.kind): i.label for i in labelled_items(items, actions)}
    assert kept == {
        ("name_change", "reply"): 1,
        ("keeper_action::any", "continuation_end"): 0,
        ("narrated_action", "reply"): 2,
    }


def test_another_sheets_neurons_must_stay_quiet_unless_a_veto_meets_no_need() -> None:
    from rein.sheets.firing import foreign_fires

    other = {"ilsa:quartermaster_action::any": {"force": False, "veto": True}}
    assert foreign_fires(other, SHEET, QUIET) == []
    assert foreign_fires(other, SHEET, NEEDS_BELL) == ["foreign-veto:ilsa:quartermaster_action::any"]
    assert foreign_fires({"ilsa:lore_guard": {"steer": True}}, SHEET, QUIET) == ["foreign-steer:ilsa:lore_guard"]
