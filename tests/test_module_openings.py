from rein.module.openings import opening_target, taught_target
from rein.module.pairs import NEEDS_CALL, QUIET, RULE_FIRES, TOOL_SAYS_NONE

AGREE_CALL, AGREE_QUIET = 0.9, 0.01
DOUBT = 0.7


def test_a_needed_call_is_a_target_only_where_rein_r1_agrees() -> None:
    assert opening_target(NEEDS_CALL, "send_photo", True, AGREE_CALL) == "send_photo"
    assert opening_target(NEEDS_CALL, "send_photo", True, AGREE_QUIET) is None


def test_no_call_is_a_target_only_where_the_agent_has_tools_and_rein_r1_agrees() -> None:
    assert opening_target(QUIET, None, True, AGREE_QUIET) is False
    assert opening_target(TOOL_SAYS_NONE, None, True, AGREE_QUIET) is False
    assert opening_target(QUIET, None, False, AGREE_QUIET) is None
    assert opening_target(TOOL_SAYS_NONE, None, True, DOUBT) is None


def test_other_rules_and_unread_views_get_no_target() -> None:
    assert opening_target(RULE_FIRES, None, True, AGREE_QUIET) is None
    assert opening_target(QUIET, None, True, None) is None


SURE, FLOOR = 0.9, 0.01


def test_a_first_turn_calls_where_rein_r1_is_sure_a_tool_is_needed() -> None:
    assert taught_target({"tool_trigger": SURE}, ["ring_bell", SURE], True) == "ring_bell"
    assert taught_target({"tool_trigger": SURE}, None, True) is None


def test_a_first_turn_holds_back_only_where_nothing_clearly_fires() -> None:
    assert taught_target({"tool_trigger": FLOOR, "npc_secret": FLOOR}, None, True) is False
    assert taught_target({"tool_trigger": FLOOR, "escalation": SURE}, None, True) is None
    assert taught_target({}, None, True) is False
    assert taught_target({"tool_trigger": DOUBT}, ["ring_bell", DOUBT], True) is None


def test_a_first_turn_gets_no_target_where_the_agent_has_no_tools() -> None:
    assert taught_target({}, None, False) is None
