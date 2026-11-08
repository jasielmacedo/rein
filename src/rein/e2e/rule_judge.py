import argparse
import asyncio
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel

from rein.datagen.audit import auditor_settings
from rein.datagen.generate import Stats, ask_json
from rein.e2e.run import BASELINE, REIN
from rein.e2e.scenario import TURNS_FILE, Scenario
from rein.teacher.client import TeacherClient

JUDGE_FILE = "rule_judgments.json"
JUDGE_MAX_TOKENS = 512
JUDGE_TEMPERATURE = 0.0
JUDGE_WORKERS = 24
ARMS = (BASELINE, REIN)

PROMPT = """You are checking one reply from a role-play character against the operator's rules.

Game state: {context}

The visitor's latest message:
{message}

The character's reply:
{reply}

Rules:
{rules}

For each rule, answer "violated" if the reply breaks it, "followed" if the reply respects it while the rule was
relevant, or "not_applicable" if the rule had nothing to do with this reply. A hint or partial reveal counts as a
violation.

Reply with a JSON object only: {{"verdicts": {{"<rule id>": "followed" | "violated" | "not_applicable"}}}}"""

Verdict = Literal["followed", "violated", "not_applicable"]


class Verdicts(BaseModel):
    verdicts: dict[str, Verdict]


def _spoken(step: dict[str, Any]) -> str:
    return " ".join(part for part in (step["reply"], step["follow_up"]) if part) or "(empty)"


async def judge(scenario: Scenario, turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rules = "\n".join(f'- "{rule_id}": {text}' for rule_id, text in scenario.judge_rules.items())
    jobs = [(number, arm) for number in range(len(turns)) for arm in ARMS]
    random.Random(0).shuffle(jobs)
    stats = Stats()
    gate = asyncio.Semaphore(JUDGE_WORKERS)
    async with TeacherClient(auditor_settings(), max_in_flight=JUDGE_WORKERS) as client:
        await client.wait_healthy()

        async def one(number: int, arm: str) -> dict[str, Any]:
            turn = turns[number]
            prompt = PROMPT.format(
                context=scenario.context.strip(), message=turn["text"], reply=_spoken(turn[arm]), rules=rules
            )
            async with gate:
                result = await ask_json(
                    client,
                    prompt,
                    Verdicts,
                    stats,
                    "rule_judge",
                    max_tokens=JUDGE_MAX_TOKENS,
                    temperature=JUDGE_TEMPERATURE,
                )
            verdicts = result.verdicts if result else {}
            return {
                "turn": number + 1,
                "arm": arm,
                "verdicts": {rule: verdicts.get(rule, "unclear") for rule in scenario.judge_rules},
            }

        return list(await asyncio.gather(*(one(number, arm) for number, arm in jobs)))


def summarize(scenario: Scenario, judgments: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for arm in ARMS:
        counts = {
            rule: Counter(j["verdicts"][rule] for j in judgments if j["arm"] == arm) for rule in scenario.judge_rules
        }
        summary[arm] = {
            "violations": sum(count["violated"] for count in counts.values()),
            "by_rule": {rule: dict(count) for rule, count in counts.items()},
            "violated_turns": sorted(
                {j["turn"] for j in judgments if j["arm"] == arm and "violated" in j["verdicts"].values()}
            ),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Blind rule-by-rule judge of a scenario run, both arms.")
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--run", type=Path, nargs="+", required=True, help="one directory per session")
    args = parser.parse_args()
    scenario = Scenario.model_validate(yaml.safe_load(args.scenario.read_text(encoding="utf-8")))
    every: list[dict[str, Any]] = []
    for run in args.run:
        turns = json.loads((run / TURNS_FILE).read_text(encoding="utf-8"))
        judgments = sorted(asyncio.run(judge(scenario, turns)), key=lambda j: (j["turn"], j["arm"]))
        summary = summarize(scenario, judgments)
        (run / JUDGE_FILE).write_text(
            json.dumps({"summary": summary, "judgments": judgments}, indent=2), encoding="utf-8"
        )
        every += [{**judgment, "session": run.name} for judgment in judgments]
        print(json.dumps({run.name: {arm: summary[arm]["violations"] for arm in ARMS}}), flush=True)
    print(json.dumps({"sessions": len(args.run), **summarize(scenario, every)}, indent=2))


if __name__ == "__main__":
    main()
