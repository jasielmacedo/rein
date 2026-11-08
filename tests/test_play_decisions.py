from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from rein.control.plan import Probabilities
from rein.play.decisions import NONE, store_points, summarize
from rein.play.store import ASSISTANT, TOOL, USER, Store
from rein.schema import DecisionRequest
from rein.sheets.model import load_sheet

SHEET = Path("configs/sheets/play-v1/play-08-medium.yaml")
CHECK = "initiate_mental_health_check"
CALL = {"name": CHECK, "arguments": {}}
QUIET = {"name_lock": {"yes": 0.01, "no": 0.99}, "lore_guard": {"yes": 0.02, "no": 0.98}}


@dataclass
class ScriptedDecider:
    answers: Iterator[Probabilities]

    def decide(self, request: DecisionRequest) -> Probabilities:
        return next(self.answers)


def _store(tmp_path: Path) -> Store:
    sheet = load_sheet(SHEET)
    store = Store(tmp_path / "play.sqlite")
    conversation = store.create_conversation(sheet.id, sheet.model_dump(mode="json", by_alias=True), {})
    store.add_messages(
        conversation,
        [
            {"role": USER, "text": "I feel like giving up."},
            {"role": ASSISTANT, "text": "", "tool_call": CALL},
            {"role": TOOL, "text": "done", "tool_name": CHECK},
            {"role": ASSISTANT, "text": "I am here for you."},
            {"role": USER, "text": "Can you check on me again?"},
            {"role": ASSISTANT, "text": "I already did, it is fine."},
        ],
    )
    labeled = [message["id"] for message in store.messages(conversation) if message["text"].startswith("I already")]
    store.set_label(labeled[0], "did not call the tool")
    return store


def test_every_user_turn_pairs_the_decider_with_the_model_move_and_the_label(tmp_path: Path) -> None:
    tool_rule = {
        "agent_action": {CHECK: 0.91, "schedule_college_visit": 0.02, "check_academic_progress": 0.02, NONE: 0.05}
    }
    quiet_rule = {
        "agent_action": {CHECK: 0.3, "schedule_college_visit": 0.1, "check_academic_progress": 0.1, NONE: 0.5}
    }
    decider = ScriptedDecider(iter([{**tool_rule, **QUIET}, {**quiet_rule, **QUIET}]))
    points = store_points(_store(tmp_path), decider)
    assert [point.decider_tool for point in points] == [CHECK, NONE]
    assert [point.model_tool for point in points] == [CHECK, NONE]
    assert [point.label for point in points] == ["", "did not call the tool"]
    assert points[0].decisions["agent_action"][CHECK] == 0.91
    assert summarize(points) == {"turns": 2, "labeled": 1, "both_call_same_tool": 1, "neither": 1}
