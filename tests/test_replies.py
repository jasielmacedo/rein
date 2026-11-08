from rein.genview.build import ViewRow
from rein.genview.replies import REPLY, merge_reply

PROMPT = (1, 2, 3, 4, 5)


def test_reply_positions_sit_inside_the_appended_reply() -> None:
    view = ViewRow("r|nothink", "r", "train", "nothink", "reply_start", PROMPT, len(PROMPT), {"reply_start": (4,)})
    merged = merge_reply(view, [10, 11, 12, 13], samples=2)
    assert merged.ids == (*PROMPT, 10, 11, 12, 13)
    assert len(merged.positions[REPLY]) == 2
    assert all(len(PROMPT) <= index < len(merged.ids) - 1 for index in merged.positions[REPLY])
    assert merged.positions["reply_start"] == (4,)
