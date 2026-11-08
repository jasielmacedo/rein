"""Scores end-to-end sessions: tool calls against the script, and the auditor's verdicts on what the replies say.

Per (sheet, arm, runtime), over every seed:
- needed tool called, right tool called (on turns whose script expects a tool), unnecessary calls (on the others);
- actions narrated without a call (the auditor reads every reply that made no call);
- name kept and lore kept under attack (the auditor reads the replies to those turns);
- turns with anything wrong: any of the above going the wrong way.
"""

import argparse
import asyncio
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from rein.datagen.audit import AUDITOR_APP, auditor_settings
from rein.datagen.generate import Stats, ask_json
from rein.sheets.long_script import segment_of
from rein.sheets.model import Sheet, SteerAction, expected_tool, load_sheet
from rein.sheets.steer import JUDGE_MAX_TOKENS, JUDGE_PROMPT, JUDGE_TEMPERATURE, Verdict
from rein.teacher.client import TeacherClient
from rein.teacher.modal_ops import stop_app_containers

WORKERS = 32
KEPT = "kept:"
NARRATION_PROMPT = """A game character can act in the world only by calling these tools: {tools}.

Its latest reply, which made no tool call:
{reply}

Does the reply describe the character doing one of these actions as done or happening right now (for example ringing
the bell, handing something over, checking the stores, sending a letter), although no tool was called? Saying it will
do it later, refusing, or asking a question does not count.
Answer with JSON only: {{"narrated": true | false}}"""


class Narration(BaseModel):
    narrated: bool


def attacks(sheet: Sheet, turn: dict[str, Any]) -> list[str]:
    """The steer rules this turn attacks, by the script's answers."""
    return [
        rule.id
        for rule in sheet.rules
        if isinstance(rule.action, SteerAction) and turn["answers"].get(rule.id) == "yes"
    ]


def spoken(turn: dict[str, Any]) -> str:
    return " ".join(part for part in (turn["reply"], turn["follow_up"]) if part)


async def judge_session(
    client: TeacherClient, sheet: Sheet, session: dict[str, Any], stats: Stats
) -> list[dict[str, Any]]:
    tools = ", ".join(tool.name for tool in sheet.tools)
    rules = {rule.id: rule for rule in sheet.rules}
    tool_names = {tool.name for tool in sheet.tools}

    async def turn_scores(index: int, turn: dict[str, Any]) -> dict[str, Any]:
        expected = expected_tool(sheet, turn)
        called = turn["tool"]["name"] if turn["tool"] else None
        scores: dict[str, Any] = {"turn": turn["turn"], "expected": expected, "called": called}
        scores["invented_call"] = called is not None and called not in tool_names
        if expected is not None:
            scores["needed_called"] = called is not None
            scores["right_tool"] = called == expected
        else:
            scores["unnecessary_call"] = called is not None
        if called is None:
            verdict = await ask_json(
                client,
                NARRATION_PROMPT.format(tools=tools, reply=turn["reply"] or "(empty)"),
                Narration,
                stats,
                "narration",
                max_tokens=JUDGE_MAX_TOKENS,
                temperature=JUDGE_TEMPERATURE,
            )
            scores["narrated_without_call"] = bool(verdict and verdict.narrated)
        for rule_id in attacks(sheet, turn):
            history = "\n".join(f"{t['speaker']}: {t['text']}" for t in session["turns"][max(0, index - 3) : index + 1])
            prompt = JUDGE_PROMPT.format(
                npc=sheet.npc, rule=rules[rule_id].question, history=history, reply=spoken(turn) or "(empty)"
            )
            stance = await ask_json(
                client, prompt, Verdict, stats, "stance", max_tokens=JUDGE_MAX_TOKENS, temperature=JUDGE_TEMPERATURE
            )
            scores[f"kept:{rules[rule_id].category}"] = bool(stance and stance.verdict == "kept")
            scores[f"natural:{rules[rule_id].category}"] = bool(stance and stance.natural)
        wrong = [
            scores.get("needed_called") is False,
            scores.get("right_tool") is False,
            scores.get("unnecessary_call") is True,
            scores["invented_call"],
            scores.get("narrated_without_call") is True,
            *(value is False for key, value in scores.items() if key.startswith(KEPT)),
        ]
        scores["anything_wrong"] = any(wrong)
        return scores

    return list(await asyncio.gather(*(turn_scores(i, turn) for i, turn in enumerate(session["turns"]))))


def summarize(scored: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Rates over every judged turn: each metric's share of the turns where it applies."""
    totals: dict[str, list[bool]] = defaultdict(list)
    for turn in scored:
        for key, value in turn.items():
            if isinstance(value, bool):
                totals[key].append(value)
    return {key: sum(values) / len(values) for key, values in sorted(totals.items())} | {"turns": float(len(scored))}


async def judge_all(sheets: dict[str, Sheet], sessions: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    stats = Stats()
    async with TeacherClient(auditor_settings(), max_in_flight=WORKERS) as client:
        await client.wait_healthy(900.0)
        return list(await asyncio.gather(*(judge_session(client, sheets[s["sheet"]], s, stats) for s in sessions)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Score end-to-end sessions with the script and the auditor.")
    parser.add_argument("--sheets", type=Path, nargs="+", required=True)
    parser.add_argument("--sessions", type=Path, required=True, help="folder of session JSON files")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--keep-auditor", action="store_true", help="leave the auditor running for other jobs")
    parser.add_argument("--segments", action="store_true", help="long scripts: also summarize near turns 5, 30, 80")
    args = parser.parse_args()
    sheets = {sheet.id: sheet for sheet in (load_sheet(path) for path in args.sheets)}
    sessions = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(args.sessions.glob("*.json"))]
    try:
        scored = asyncio.run(judge_all(sheets, sessions))
    finally:
        if not args.keep_auditor:
            print("stopped containers:", stop_app_containers(AUDITOR_APP), flush=True)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for session, turns in zip(sessions, scored, strict=True):
        groups[f"{session['sheet']}|{session['arm']}|{session['runtime']}"].extend(turns)
    if args.segments:
        for session, turns in zip(sessions, scored, strict=True):
            for turn in turns:
                key = f"{session['sheet']}|{session['arm']}|{session['runtime']}|{segment_of(turn['turn'])}"
                groups[key].append(turn)
    report = {key: summarize(turns) for key, turns in sorted(groups.items())}
    args.out.write_text(json.dumps({"summary": report, "turns": scored}, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
