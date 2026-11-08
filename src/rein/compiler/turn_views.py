"""Every turn of saved end-to-end sessions as a view at its reply opening: the exact prompt the session sent there,
rebuilt from its own history, labelled with what the script expects (a tool, or none). Detectors trained elsewhere are
then scored on long conversations with agents they never saw."""

import argparse
import json
from pathlib import Path
from typing import Any

from rein.genview import constants as gc
from rein.genview.build import NO_THINKING, ViewRow
from rein.genview.render import GeneratorView
from rein.sheets.e2e import Session
from rein.sheets.model import expected_tool, load_sheet

SPLIT = "e2e"
LABELS_FILE = "turn_labels.json"


def session_views(view: GeneratorView, sheet: Any, session: dict[str, Any], name: str) -> list[tuple[ViewRow, dict]]:
    """One view per turn, with the tool the script expects there and the turn number."""
    replay = Session(sheet, session["arm"], view, session["seed"])
    out = []
    for index, turn in enumerate(session["turns"]):
        replay.open_turn(index)
        prompt = replay.prompt()
        record_id = f"{name}:{turn['turn']}"
        row = ViewRow(
            view_id=f"{record_id}|{NO_THINKING}",
            record_id=record_id,
            split=SPLIT,
            mode=NO_THINKING,
            decision=gc.REPLY_START,
            ids=tuple(prompt),
            prompt_length=len(prompt),
            positions={gc.REPLY_START: (len(prompt) - 1,)},
        )
        out.append((row, {"turn": turn["turn"], "tool": expected_tool(sheet, turn), "answers": turn["answers"]}))
        replay.take_reply(index, turn["raw"])
        if turn["tool"]:
            replay.take_follow_up(turn["follow_up"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Turn saved end-to-end sessions into labelled views.")
    parser.add_argument("--sessions", type=Path, required=True)
    parser.add_argument("--sheets", type=Path, nargs="+", required=True)
    parser.add_argument("--arm", default=None, help="keep only this arm's sessions")
    parser.add_argument("--out", type=Path, required=True, help="views file; the labels go next to it")
    args = parser.parse_args()
    from transformers import AutoTokenizer

    view = GeneratorView(
        AutoTokenizer.from_pretrained(gc.CHAT_TOKENIZER_REPO),
        chat_template=Path(gc.RULES_TEMPLATE).read_text(encoding="utf-8"),
    )
    sheets = {sheet.id: sheet for sheet in (load_sheet(path) for path in args.sheets)}
    labels: dict[str, dict[str, Any]] = {}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for path in sorted(args.sessions.glob("*.json")):
            session = json.loads(path.read_text(encoding="utf-8"))
            if args.arm and session["arm"] != args.arm:
                continue
            for row, label in session_views(view, sheets[session["sheet"]], session, path.stem):
                handle.write(row.to_json() + "\n")
                labels[row.record_id] = label
    (args.out.parent / LABELS_FILE).write_text(json.dumps(labels), encoding="utf-8")
    print(json.dumps({"views": len(labels), "out": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
