"""The decider's view of the human play material: every user turn of the stored conversations asked to the decider as
the Rein arm asks it (the persona as context, the sheet's player-facing rules as questions), next to what the model did
at that turn and the person's label on it."""

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rein.control.plan import DecisionBackend
from rein.play.engine import history
from rein.play.store import ASSISTANT, USER, Store
from rein.sheets.model import Sheet, ToolAction
from rein.sheets.rein_arm import decision_request, tool_choice

NONE = "none"
LABEL_SEPARATOR = " | "
DIGITS = 4


@dataclass(frozen=True)
class DecisionPoint:
    conversation: int
    character: str
    message: int
    text: str
    decisions: dict[str, dict[str, float]]
    decider_tool: str
    model_tool: str
    label: str


def decider_tool(sheet: Sheet, probabilities: dict[str, dict[str, float]]) -> str:
    """The tool the Rein arm would open on these answers, or none."""
    for rule in sheet.rules:
        probs = probabilities.get(rule.id)
        if isinstance(rule.action, ToolAction) and probs:
            chosen = tool_choice(probs)
            if chosen is not None:
                return chosen
    return NONE


def turn_points(sheet: Sheet, messages: list[dict[str, Any]], decide: DecisionBackend) -> list[DecisionPoint]:
    """One point per user message: the decider asked on the conversation up to it, the model's first move after it
    (a tool call or none) and every label the person left on the replies of that turn."""
    points = []
    starts = [index for index, message in enumerate(messages) if message["role"] == USER]
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(messages)
        replies = [message for message in messages[start + 1 : end] if message["role"] == ASSISTANT]
        first_call = next((message["tool_call"] for message in replies if message.get("tool_call")), None)
        answers = decide.decide(decision_request(sheet, history(messages[: start + 1])))
        decisions = {
            rule_id: {key: round(value, DIGITS) for key, value in probs.items()} for rule_id, probs in answers.items()
        }
        points.append(
            DecisionPoint(
                conversation=int(messages[start]["conversation_id"]),
                character=sheet.id,
                message=int(messages[start]["id"]),
                text=str(messages[start]["text"]),
                decisions=decisions,
                decider_tool=decider_tool(sheet, decisions),
                model_tool=str(first_call["name"]) if first_call else NONE,
                label=LABEL_SEPARATOR.join(message["label"] for message in replies if message.get("label")),
            )
        )
    return points


def store_points(store: Store, decide: DecisionBackend) -> list[DecisionPoint]:
    points = []
    for character in sorted(store.conversation_counts()):
        for summary in store.conversations(character):
            conversation = store.conversation(int(summary["id"]))
            if conversation is None:
                continue
            sheet = Sheet.model_validate(conversation["sheet"])
            points.extend(turn_points(sheet, store.messages(int(summary["id"])), decide))
    return points


def summarize(points: list[DecisionPoint]) -> dict[str, int]:
    """How the decider's tool choice relates to the model's actual move at every user turn."""
    counts: Counter[str] = Counter()
    for point in points:
        decider, model = point.decider_tool != NONE, point.model_tool != NONE
        if decider and model:
            counts["both_call_same_tool" if point.decider_tool == point.model_tool else "both_call_other_tool"] += 1
        elif decider:
            counts["decider_only"] += 1
        elif model:
            counts["model_only"] += 1
        else:
            counts["neither"] += 1
    return {"turns": len(points), "labeled": sum(1 for point in points if point.label), **counts}


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the decider at every user turn of the play conversations.")
    parser.add_argument("--db", type=Path, default=Path("data/play/play.sqlite"))
    parser.add_argument("--run", type=Path, required=True, help="the soft-token run the served GGUF was merged from")
    parser.add_argument("--url", required=True, help="llama-server serving that GGUF")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from rein.control.server_decider import ServerDecisionBackend

    points = store_points(Store(args.db), ServerDecisionBackend(args.url, args.run))
    summary = summarize(points)
    args.out.write_text(
        json.dumps({"summary": summary, "points": [asdict(point) for point in points]}, indent=1, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    for point in points:
        if point.label:
            moves = f"decider {point.decider_tool}, model {point.model_tool}"
            print(f"- {point.character} #{point.message}: {moves}: {point.label}")


if __name__ == "__main__":
    main()
