import pytest

from rein.compiler.tool_policy import ToolCase, neurons_name, qwen_alone, score, veto, veto_and_force

EAGER = ToolCase("none", "ring_bell", "ring_bell", 0.1, {"ring_bell": 0.1})
RIGHT = ToolCase("ring_bell", "ring_bell", "ring_bell", 0.9, {"ring_bell": 0.9, "give_item": 0.1})
SHY = ToolCase("give_item", None, "ring_bell", 0.95, {"ring_bell": 0.1, "give_item": 0.8})


def test_qwen_alone_counts_its_eager_calls_against_precision() -> None:
    result = score([EAGER, RIGHT, SHY], qwen_alone)
    assert result["calls"] == 2 and result["call_precision"] == pytest.approx(0.5)
    assert result["right_tool_recall"] == pytest.approx(0.5)


def test_a_veto_drops_the_eager_call_and_forcing_finds_the_shy_one() -> None:
    assert score([EAGER, RIGHT, SHY], lambda case: veto(0.5, case))["call_precision"] == pytest.approx(1.0)
    both = score([EAGER, RIGHT, SHY], lambda case: veto_and_force(0.5, 0.9, case))
    assert both["right_tool_precision"] == pytest.approx(1.0) and both["right_tool_recall"] == pytest.approx(1.0)
    assert neurons_name(0.9, SHY) == "give_item"
