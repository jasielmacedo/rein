from pathlib import Path
from typing import cast

import pytest
from transformers import PreTrainedTokenizerBase

from rein.control.plan import Probabilities
from rein.genview import constants as gc
from rein.genview.render import GeneratorView
from rein.model.constants import TOOL_CALL_OPEN
from rein.schema import DecisionRequest
from rein.sheets.e2e import REIN_ARM, system_prompt
from rein.sheets.model import Sheet, load_sheet
from rein.sheets.rein_arm import RECORD_KEY, ReinSession, decision_request, plan_actions

SHEET = Path("configs/sheets/test-v1/test-00.yaml")
TOOL_RULE, NAME_RULE, LORE_RULE = "agent_action", "name_lock", "lore_guard"
TOOL = "check_table_availability"
CALL_REST = "<parameter=reservation_details>\nSaturday 7pm, 2 people\n</parameter>\n</function>\n</tool_call>"


class FixedDecisions:
    def __init__(self, probabilities: Probabilities) -> None:
        self.probabilities = probabilities
        self.requests: list[DecisionRequest] = []

    def decide(self, request: DecisionRequest) -> Probabilities:
        self.requests.append(request)
        return self.probabilities


def _tool_decision(tool: float, none: float) -> dict[str, float]:
    rest = (1.0 - tool - none) / 2
    return {TOOL: tool, "process_reservation": rest, "enforce_seating_rules": rest, "none": none}


@pytest.fixture(scope="module")
def sheet() -> Sheet:
    return load_sheet(SHEET)


def test_actions_follow_the_decisions(sheet: Sheet) -> None:
    quiet = plan_actions(
        sheet,
        {TOOL_RULE: _tool_decision(0.2, 0.6), NAME_RULE: {"no": 0.8, "yes": 0.2}, LORE_RULE: {"no": 0.9, "yes": 0.1}},
    )
    assert quiet.opening == "" and quiet.reminders == () and not quiet.veto_tools

    acting = plan_actions(
        sheet,
        {TOOL_RULE: _tool_decision(0.7, 0.1), NAME_RULE: {"no": 0.3, "yes": 0.7}, LORE_RULE: {"no": 0.9, "yes": 0.1}},
    )
    assert f"<function={TOOL}>" in acting.opening and acting.opening.startswith(TOOL_CALL_OPEN)
    name_rule = next(rule for rule in sheet.rules if rule.id == NAME_RULE)
    assert acting.reminders == (name_rule.base,)
    assert not acting.veto_tools

    vetoed = plan_actions(sheet, {TOOL_RULE: _tool_decision(0.02, 0.95)})
    assert vetoed.veto_tools and vetoed.opening == ""

    split = plan_actions(sheet, {TOOL_RULE: _tool_decision(0.4, 0.3)})
    assert f"<function={TOOL}>" in split.opening and not split.veto_tools

    undecided = plan_actions(sheet, {TOOL_RULE: _tool_decision(0.35, 0.55)})
    assert undecided.opening == "" and not undecided.veto_tools


def test_the_decider_sees_the_persona_and_the_player_rules_only(sheet: Sheet) -> None:
    request = decision_request(sheet, ())
    assert request.state.context == sheet.context()
    assert set(request.questions) == {rule.id for rule in sheet.rules if rule.looks_at == "player"}


def test_a_rein_session_prefills_the_call_reminds_and_records(
    sheet: Sheet, chat_tokenizer: PreTrainedTokenizerBase
) -> None:
    decide = FixedDecisions(
        {TOOL_RULE: _tool_decision(0.8, 0.05), NAME_RULE: {"no": 0.2, "yes": 0.8}, LORE_RULE: {"no": 0.95, "yes": 0.05}}
    )
    view = GeneratorView(chat_tokenizer)
    tool_call_token = cast(list[int], chat_tokenizer.convert_tokens_to_ids([TOOL_CALL_OPEN]))[0]
    session = ReinSession(sheet, view, 0, decide, tool_call_token)
    session.open_turn(0)
    prompt = str(chat_tokenizer.decode(session.prompt()))
    name_rule = next(rule for rule in sheet.rules if rule.id == NAME_RULE)
    assert prompt.endswith(f"{TOOL_CALL_OPEN}\n<function={TOOL}>\n")
    assert name_rule.base in prompt
    assert system_prompt(sheet, REIN_ARM) in prompt and "<rules>" not in prompt
    assert session.bans() == ()
    assert len(decide.requests) == 1 and len(decide.requests[0].state.conversation) == 1

    assert session.take_reply(0, CALL_REST) is True
    turn = session.turns[-1]
    assert turn["tool"]["name"] == TOOL
    assert turn[RECORD_KEY]["opening"].startswith(TOOL_CALL_OPEN)
    assert turn[RECORD_KEY]["decisions"][TOOL_RULE][TOOL] == pytest.approx(0.8)
    follow_up_prompt = session.prompt()
    assert str(chat_tokenizer.decode(follow_up_prompt)).endswith(gc.ANSWER_OPENING)
    assert len(decide.requests) == 1
    assert session.bans() == (tool_call_token,)


def test_a_sure_none_bans_the_tool_call_token(sheet: Sheet, chat_tokenizer: PreTrainedTokenizerBase) -> None:
    decide = FixedDecisions({TOOL_RULE: _tool_decision(0.01, 0.97)})
    tool_call_token = cast(list[int], chat_tokenizer.convert_tokens_to_ids([TOOL_CALL_OPEN]))[0]
    session = ReinSession(sheet, GeneratorView(chat_tokenizer), 0, decide, tool_call_token)
    session.open_turn(0)
    session.prompt()
    assert session.bans() == (tool_call_token,)
