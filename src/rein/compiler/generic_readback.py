"""Where the generic neurons acted in end-to-end sessions (DR-026): each turn's prompt rebuilt as the session saw it,
read with its own reply through the checkpoint, and every detector's decision taken at the reply opening and on each
reply token. Joined with the judge's verdicts, this separates a rule that was never detected from one that was
detected and still broken."""

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from rein.compiler.category_neurons import MANIFEST_FILE
from rein.compiler.sameness import MlpInput, gate_rows
from rein.genview import constants as gc
from rein.genview.render import GeneratorView
from rein.sheets.e2e import Session
from rein.sheets.model import load_sheet

FILE = "readback.jsonl"


@torch.no_grad()
def session_firing(
    model: Any, view: GeneratorView, sheet: Any, session: dict[str, Any], rows: dict[str, torch.Tensor], block: int
) -> list[dict[str, Any]]:
    """Per turn of one saved session: which detectors fire where the reply opens, and on what share of its tokens."""
    replay = Session(sheet, session["arm"], view, session["seed"])
    capture = MlpInput(model.model.layers[block])
    out = []
    try:
        for index, turn in enumerate(session["turns"]):
            replay.open_turn(index)
            prompt = replay.prompt()
            reply = view.text_ids(turn["raw"])
            model(input_ids=torch.tensor([[*prompt, *reply]], device=model.device))
            assert capture.value is not None
            inputs = capture.value[0].float()
            opening = len(prompt) - 1
            decided: dict[str, Any] = {}
            for name, row in rows.items():
                fired = inputs @ row.to(inputs.device) >= 0
                decided[name] = {
                    "opening": bool(fired[opening]),
                    "reply_share": float(fired[opening + 1 :].float().mean()) if reply else 0.0,
                }
            out.append({"turn": turn["turn"], "answers": turn["answers"], "fired": decided})
            replay.take_reply(index, turn["raw"])
            if turn["tool"]:
                replay.take_follow_up(turn["follow_up"])
    finally:
        capture.handle.remove()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Read back where the generic neurons fired in end-to-end sessions.")
    parser.add_argument("--sessions", type=Path, required=True)
    parser.add_argument("--sheets", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--neurons", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, dtype=torch.bfloat16, device_map="cuda").eval()
    view = GeneratorView(
        AutoTokenizer.from_pretrained(gc.CHAT_TOKENIZER_REPO),
        chat_template=Path(gc.RULES_TEMPLATE).read_text(encoding="utf-8"),
    )
    sheets = {sheet.id: sheet for sheet in (load_sheet(path) for path in args.sheets)}
    rows = gate_rows(args.neurons)
    block = json.loads((args.neurons / MANIFEST_FILE).read_text(encoding="utf-8"))["block"]
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / FILE).open("w", encoding="utf-8") as handle:
        for path in sorted(args.sessions.glob("*.json")):
            session = json.loads(path.read_text(encoding="utf-8"))
            for turn in session_firing(model, view, sheets[session["sheet"]], session, rows, block):
                handle.write(json.dumps({"session": path.name, **turn}) + "\n")
            print(json.dumps({"session": path.name}), flush=True)


if __name__ == "__main__":
    main()
