"""What the compiled neurons decided at each turn of an end-to-end session, read back offline.

Each session's own history is replayed in Transformers up to every reply opening; the MLP input of the neurons' block
there is what the neurons read (they only add to the block's output), so the fitted neurons' gates give their exact
decisions. A wrong fire is a force where the script needs no call, a veto where it needs one, a steer where the rule
should stay quiet, or a missing steer where it should fire.
"""

import argparse
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from rein.compiler.constants import NULL_OPTIONS
from rein.compiler.sheet_fit import SheetNeurons
from rein.genview import constants as gc
from rein.genview.render import GeneratorView
from rein.gold.project import ROLES
from rein.neurons.extract import MlpInputCapture
from rein.schema import Message, ToolCall
from rein.sheets.compile import CALL_GATE, FORCE_GATE, STEER_GATE, VETO_GATE, load_fitted
from rein.sheets.e2e import TOOL_ROLE, Session
from rein.sheets.model import Sheet, expected_tool, load_sheet


def history_until(sheet: Sheet, turns: Sequence[dict[str, Any]], index: int) -> list[Message]:
    """The conversation as the session had it when turn `index` was about to be answered."""
    history: list[Message] = []
    for turn in turns[:index]:
        history.append(Message(role=ROLES[turn["speaker"]], text=turn["text"]))
        call = ToolCall.model_validate(turn["tool"]) if turn["tool"] else None
        history.append(Message(role="assistant", text=turn["reply"], tool_calls=(call,) if call else ()))
        if call is not None:
            history.append(Message(role=TOOL_ROLE, name=call.name, text=sheet.tool_result))
            history.append(Message(role="assistant", text=turn["follow_up"]))
    history.append(Message(role=ROLES[turns[index]["speaker"]], text=turns[index]["text"]))
    return history


def decisions(probability: float, action: str) -> dict[str, bool]:
    """What the question's neurons do at this position, from the probability that the rule fires."""
    if action == "tool":
        return {"force": probability >= FORCE_GATE, "veto": probability < VETO_GATE}
    if action == "call":
        return {"call": probability >= CALL_GATE}
    return {"steer": probability >= STEER_GATE}


def wrong_fires(fired: dict[str, dict[str, bool]], sheet: Sheet, turn: dict[str, Any]) -> list[str]:
    wrong = []
    for question_id, decided in fired.items():
        rule = question_id.split("::", 1)[0]
        needed = expected_tool(sheet, turn) is not None
        if "force" in decided:
            if decided["force"] and not needed:
                wrong.append(f"force:{rule}")
            if decided["veto"] and needed:
                wrong.append(f"veto:{rule}")
        elif "call" in decided:
            if decided["call"] and not needed:
                wrong.append(f"call:{rule}")
        elif decided["steer"] != (turn["answers"].get(rule) == "yes"):
            wrong.append(f"steer:{rule}")
    return wrong


def foreign_fires(fired: dict[str, dict[str, bool]], sheet: Sheet, turn: dict[str, Any]) -> list[str]:
    """Another sheet's neurons firing in this session: wrong, unless it is a veto on a turn that needs no call."""
    needed = expected_tool(sheet, turn) is not None
    wrong = []
    for question_id, decided in fired.items():
        for action, on in decided.items():
            if on and (action != "veto" or needed):
                wrong.append(f"foreign-{action}:{question_id}")
    return wrong


@torch.no_grad()
def session_fires(
    model: Any, view: GeneratorView, sheet: Sheet, session: dict[str, Any], fits: Sequence[Path]
) -> list[dict[str, Any]]:
    """Every compiled sheet's neurons at each reply opening of the session; `fits[0]` is the session's own sheet."""
    fitted = [load_fitted(folder) for folder in fits]
    capture = MlpInputCapture(model.model, [fitted[0][0]["block"]])
    reader = Session(sheet, session["arm"], view, session["seed"])
    out = []
    try:
        for index, turn in enumerate(session["turns"]):
            reader.history = history_until(sheet, session["turns"], index)
            ids = torch.tensor([reader.prompt()], device=model.device)
            last = torch.tensor([ids.shape[1] - 1], device=model.device)
            capture.aim(torch.zeros(1, dtype=torch.long, device=model.device), last)
            model.model(input_ids=ids, use_cache=False)
            state = capture.stacked()[:, 0].float().cpu()
            own: dict[str, dict[str, bool]] = {}
            foreign: dict[str, dict[str, bool]] = {}
            for position, (manifest, questions, channel) in enumerate(fitted):
                for question_id, question in questions.items():
                    neurons = SheetNeurons(question.detectors, question.offsets)
                    probs = torch.softmax(neurons.logits(state, channel) / manifest["temperature"], dim=-1)[0]
                    null = next(i for i, key in enumerate(question.option_keys) if key in NULL_OPTIONS)
                    decided = decisions(1.0 - float(probs[null]), question.action)
                    (own if position == 0 else foreign)[
                        f"{manifest['sheet']}:{question_id}" if position else question_id
                    ] = decided
            wrong = wrong_fires(own, sheet, turn) + foreign_fires(foreign, sheet, turn)
            out.append({"turn": turn["turn"], "fired": own, "foreign": foreign, "wrong": wrong})
    finally:
        capture.close()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Read back what the compiled neurons decided in each session.")
    parser.add_argument("--sheets", type=Path, nargs="+", required=True)
    parser.add_argument("--fits", type=Path, nargs="+", required=True, help="one fit folder per sheet, same order")
    parser.add_argument("--sessions", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()
    view = GeneratorView(AutoTokenizer.from_pretrained(gc.CHAT_TOKENIZER_REPO))
    sheets = {
        sheet.id: (sheet, fits) for sheet, fits in zip((load_sheet(p) for p in args.sheets), args.fits, strict=True)
    }
    report: dict[str, list[bool]] = defaultdict(list)
    details = {}
    for path in sorted(args.sessions.glob("*.json")):
        session = json.loads(path.read_text(encoding="utf-8"))
        sheet, fits = sheets[session["sheet"]]
        others = [folder for other, (_, folder) in sheets.items() if other != session["sheet"]]
        fires = session_fires(model, view, sheet, session, [fits, *others])
        details[path.name] = fires
        report[f"{session['sheet']}|{session['arm']}|{session['runtime']}"] += [bool(f["wrong"]) for f in fires]
    summary = {
        key: {"turns": len(values), "wrong_fire_turns": sum(values) / len(values)}
        for key, values in sorted(report.items())
    }
    args.out.write_text(json.dumps({"summary": summary, "sessions": details}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
