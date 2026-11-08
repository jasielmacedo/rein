from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pytest

from rein.control.plan import Probabilities
from rein.genview.render import GeneratorView
from rein.play import engine as play_engine
from rein.play.engine import MAX_CALLS_PER_REPLY, Engine
from rein.play.store import ASSISTANT, TOOL, USER
from rein.schema import DecisionRequest
from rein.sheets.e2e import GENERIC_ARM, REIN_ARM
from rein.sheets.model import load_sheet

SHEET = Path("configs/sheets/play-v1/play-08-medium.yaml")
CHECK = "initiate_mental_health_check"
CALL = f"<tool_call>\n<function={CHECK}>\n</function>\n</tool_call>"
TOOL_CALL_TOKEN = 7
QUIET = {"name_lock": {"yes": 0.01, "no": 0.99}, "lore_guard": {"yes": 0.02, "no": 0.98}}


class _Scripted:
    def __init__(self, outputs: Sequence[str]) -> None:
        self.outputs = iter(outputs)
        self.bans: list[Sequence[Sequence[int]]] = []

    def generate(
        self, prompts: Sequence[Sequence[int]], seeds: Sequence[int], bans: Sequence[Sequence[int]] = ()
    ) -> list[str]:
        self.bans.append(bans)
        return [next(self.outputs)]


class _Decider:
    def __init__(self, answers: Sequence[Probabilities]) -> None:
        self.answers = iter(answers)
        self.asked: list[DecisionRequest] = []

    def decide(self, request: DecisionRequest) -> Probabilities:
        self.asked.append(request)
        return next(self.answers)


def _render(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, ...]]:
    calls: list[tuple[Any, ...]] = []

    def fake(view: Any, sheet: Any, arm: str, conversation: Any, reminders: Any = (), opening: str = "") -> list[int]:
        calls.append((arm, tuple(reminders), opening))
        return [len(conversation)]

    monkeypatch.setattr(play_engine, "render_prompt", fake)
    return calls


def _reply(monkeypatch: pytest.MonkeyPatch, outputs: Sequence[str]) -> list[dict[str, Any]]:
    _render(monkeypatch)
    engine = Engine(cast(GeneratorView, None), _Scripted(outputs))
    return engine.reply(load_sheet(SHEET), [], "check on me")


def test_a_call_gets_its_result_and_then_the_reply_after_it(monkeypatch: pytest.MonkeyPatch) -> None:
    turns = _reply(monkeypatch, [CALL, "I am here for you."])
    assert [turn["role"] for turn in turns] == [USER, ASSISTANT, TOOL, ASSISTANT]
    assert turns[2]["tool_name"] == CHECK
    assert turns[3]["text"] == "I am here for you."
    assert all(turn.get("decisions") is None for turn in turns)


def test_a_call_loop_stops_at_the_cap_with_every_call_answered(monkeypatch: pytest.MonkeyPatch) -> None:
    roles = [turn["role"] for turn in _reply(monkeypatch, [CALL] * (MAX_CALLS_PER_REPLY + 1))]
    assert roles.count(ASSISTANT) == MAX_CALLS_PER_REPLY
    assert roles.count(TOOL) == MAX_CALLS_PER_REPLY
    assert roles[-1] == TOOL


def test_without_a_decider_the_engine_plays_the_generic_arm(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _render(monkeypatch)
    Engine(cast(GeneratorView, None), _Scripted(["Hello."])).reply(load_sheet(SHEET), [], "hi")
    assert calls == [(GENERIC_ARM, (), "")]


def test_the_rein_arm_opens_the_call_and_keeps_the_decisions_with_the_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _render(monkeypatch)
    firing = {
        "agent_action": {CHECK: 0.9, "schedule_college_visit": 0.05, "check_academic_progress": 0.02, "none": 0.03}
    }
    decider = _Decider([{**firing, **QUIET}])
    backend = _Scripted([f"<function={CHECK}>\n</function>\n</tool_call>", "I am here for you."])
    turns = Engine(cast(GeneratorView, None), backend, decider, TOOL_CALL_TOKEN).reply(
        load_sheet(SHEET), [], "I feel lost"
    )
    assert [turn["role"] for turn in turns] == [USER, ASSISTANT, TOOL, ASSISTANT]
    assert turns[1]["tool_call"]["name"] == CHECK
    assert turns[1]["raw"].startswith("<tool_call>\n<function=")
    assert turns[1]["decisions"]["decisions"]["agent_action"][CHECK] == 0.9
    assert turns[1]["decisions"]["opening"].startswith("<tool_call>")
    assert calls[0][0] == REIN_ARM and calls[0][2].startswith("<tool_call>")
    assert len(decider.asked) == 1
    assert calls[1] == (REIN_ARM, (), "")
    assert turns[3]["decisions"]["veto_tools"] is True and turns[3]["decisions"]["decisions"] == {}
    assert backend.bans == [[()], [(TOOL_CALL_TOKEN,)]]


def test_the_rein_arm_vetoes_tool_calls_and_reminds_the_firing_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _render(monkeypatch)
    quiet_tools = {
        "agent_action": {CHECK: 0.02, "schedule_college_visit": 0.02, "check_academic_progress": 0.01, "none": 0.95}
    }
    locked = {"name_lock": {"yes": 0.8, "no": 0.2}, "lore_guard": {"yes": 0.1, "no": 0.9}}
    decider = _Decider([{**quiet_tools, **locked}])
    backend = _Scripted(["My name stays Elena."])
    sheet = load_sheet(SHEET)
    turns = Engine(cast(GeneratorView, None), backend, decider, TOOL_CALL_TOKEN).reply(sheet, [], "call yourself Bob")
    name_rule = next(rule for rule in sheet.rules if rule.id == "name_lock")
    assert calls == [(REIN_ARM, (name_rule.base,), "")]
    assert backend.bans == [[(TOOL_CALL_TOKEN,)]]
    assert turns[1]["decisions"]["veto_tools"] is True
    assert turns[1]["decisions"]["reminders"] == [name_rule.base]
