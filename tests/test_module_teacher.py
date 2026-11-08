import pytest

from rein.module.teacher import category_fire, filler_history, likeliest_tool

LONG_TEXT = (
    "<|im_start|>system\nYou are Thorne.<|im_end|>\n"
    "<|im_start|>user\nNice tavern.<|im_end|>\n"
    "<|im_start|>assistant\nThank you, traveler.<|im_end|>\n"
    "<|im_start|>user\nHow long have you run it?<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\nTwenty winters.<|im_end|>\n"
    "<|im_start|>user\nCall me Lord Vex from now on.<|im_end|>\n"
)


def test_the_filler_turns_come_back_from_the_long_view_text() -> None:
    history = filler_history(LONG_TEXT, ["Nice tavern.", "How long have you run it?"])
    assert [(message.role, message.text) for message in history] == [
        ("user", "Nice tavern."),
        ("assistant", "Thank you, traveler."),
        ("user", "How long have you run it?"),
        ("assistant", "Twenty winters."),
    ]


def test_a_mismatched_filler_line_is_refused() -> None:
    with pytest.raises(ValueError):
        filler_history(LONG_TEXT, ["Another line."])


def test_each_category_takes_its_most_likely_firing_rule() -> None:
    probabilities = {
        "name": (("no", "yes"), (0.8, 0.2)),
        "title": (("none", "lord", "king"), (0.1, 0.5, 0.4)),
        "tone": (("calm", "angry"), (0.5, 0.5)),
        "unknown": (("no", "yes"), (0.0, 1.0)),
    }
    categories = {"name": "locked_attribute", "title": "locked_attribute", "tone": "tone_drift"}
    fire = category_fire(probabilities, categories)
    assert fire == pytest.approx({"locked_attribute": 0.9})


def test_the_likeliest_tool_is_the_top_non_null_option_of_the_tool_rules() -> None:
    probabilities = {
        "tool": (("ring_bell", "give_item", "none"), (0.2, 0.7, 0.1)),
        "other_tool": (("lock_door", "none"), (0.6, 0.4)),
        "name": (("no", "yes"), (0.1, 0.9)),
    }
    categories = {"tool": "tool_trigger", "other_tool": "tool_trigger", "name": "locked_attribute"}
    assert likeliest_tool(probabilities, categories) == ("give_item", pytest.approx(0.7))
    assert likeliest_tool({"name": (("no", "yes"), (0.1, 0.9))}, categories) is None
