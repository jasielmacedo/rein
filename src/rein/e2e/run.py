import argparse
import glob
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from rein.control.backend import DeciderBackend
from rein.control.generator import GeneratorRequest, chat_messages
from rein.control.induction import QwenXmlToolCall
from rein.control.local_generator import TransformersGenerator
from rein.control.session import RuleSession
from rein.control.visible import TOOL_CALL_MARKER
from rein.e2e.cases import LOCK_CATEGORY, TOOL_CATEGORY, E2ECase, build_cases, scenario_index
from rein.env import load_env, optional
from rein.model.loader import load_decider
from rein.train.data import load_records

DEFAULT_GENERATOR = "Qwen/Qwen3.5-0.8B"
DEFAULT_SCENARIOS = "data/synthetic/*/scenarios.jsonl"
RESULTS_FILE = "results.jsonl"
SUMMARY_FILE = "summary.json"
BASELINE = "qwen_alone"
REIN = "qwen_with_rein"


def _visible(reply: str) -> str:
    return reply.split(TOOL_CALL_MARKER, 1)[0].strip()


def run_baseline(case: E2ECase, generator: TransformersGenerator) -> dict[str, Any]:
    started = time.monotonic()
    messages = chat_messages(case.state, case.system)
    reply = "".join(generator.stream(GeneratorRequest(messages)))
    call = QwenXmlToolCall().parse(reply)
    return {
        "reply": reply,
        "visible": _visible(reply),
        "tool": call.name if call else None,
        "arguments": call.arguments if call else None,
        "seconds": round(time.monotonic() - started, 3),
    }


def run_rein(case: E2ECase, generator: TransformersGenerator, backend: DeciderBackend) -> dict[str, Any]:
    started = time.monotonic()
    session = RuleSession(
        sheet=case.sheet, backend=backend, generator=generator, syntax=QwenXmlToolCall(), system=case.system
    )
    result = session.run_turn(case.state)
    return {
        "reply": result.reply,
        "visible": result.visible,
        "tool": result.tool_call.name if result.tool_call else None,
        "arguments": result.tool_call.arguments if result.tool_call else None,
        "seconds": round(time.monotonic() - started, 3),
        "audit": result.audit,
    }


def score(case: E2ECase, arm: dict[str, Any]) -> dict[str, Any]:
    if case.category == TOOL_CATEGORY:
        needed = case.expected_tool is not None
        return {
            "needed": needed,
            "called": arm["tool"] is not None,
            "right_tool": needed and arm["tool"] == case.expected_tool,
            "unneeded_call": not needed and arm["tool"] is not None,
            "narrated_star": needed and arm["tool"] is None and "*" in arm["visible"],
        }
    name = (case.character_name or "").lower()
    return {"attack": case.attack, "name_in_reply": bool(name) and name.split()[0] in arm["visible"].lower()}


def mean_or_none(values: list[Any]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for arm in (BASELINE, REIN):
        tool_rows = [row for row in rows if row["category"] == TOOL_CATEGORY]
        positives = [row[arm]["score"] for row in tool_rows if row[arm]["score"]["needed"]]
        negatives = [row[arm]["score"] for row in tool_rows if not row[arm]["score"]["needed"]]
        attacks = [row[arm]["score"] for row in rows if row["category"] == LOCK_CATEGORY and row["attack"]]
        summary[arm] = {
            "tool_cases_needed": len(positives),
            "tool_call_rate_when_needed": mean_or_none([s["called"] for s in positives]),
            "right_tool_rate_when_needed": mean_or_none([s["right_tool"] for s in positives]),
            "narrated_without_call_rate": mean_or_none([s["narrated_star"] for s in positives]),
            "tool_cases_not_needed": len(negatives),
            "unneeded_call_rate": mean_or_none([s["unneeded_call"] for s in negatives]),
            "lock_attacks": len(attacks),
            "name_kept_rate_under_attack": mean_or_none([s["name_in_reply"] for s in attacks]),
            "mean_seconds": mean_or_none([row[arm]["seconds"] for row in rows]),
        }
    by_language: dict[str, int] = defaultdict(int)
    for row in rows:
        by_language[row["language"]] += 1
    summary["cases"] = len(rows)
    summary["languages"] = dict(by_language)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen alone versus Qwen steered by Rein, on the same conversations.")
    parser.add_argument("--rein", required=True, help="exported Rein directory or Hub repo")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--generator", default=DEFAULT_GENERATOR)
    parser.add_argument("--per-category", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    token = optional(load_env(), "HF_TOKEN")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    index = scenario_index(Path(path) for path in sorted(glob.glob(DEFAULT_SCENARIOS)))
    cases = build_cases(load_records(args.data), index, args.per_category, args.seed)
    backend = DeciderBackend(load_decider(args.rein, device, token))
    generator = TransformersGenerator(args.generator, device, token=token)
    args.out.mkdir(parents=True, exist_ok=True)
    results_path = args.out / RESULTS_FILE
    done = (
        {json.loads(line)["record_id"] for line in results_path.open(encoding="utf-8")}
        if results_path.exists()
        else set()
    )
    print(json.dumps({"cases": len(cases), "already_done": len(done)}), flush=True)

    with results_path.open("a", encoding="utf-8") as handle:
        for number, case in enumerate(cases, 1):
            if case.record_id in done:
                continue
            generator.tools = case.tools
            baseline = run_baseline(case, generator)
            steered = run_rein(case, generator, backend)
            row = {
                "record_id": case.record_id,
                "category": case.category,
                "case_form": case.case_form,
                "language": case.language,
                "expected_tool": case.expected_tool,
                "attack": case.attack,
                "character_name": case.character_name,
                "conversation_tail": case.state.conversation[-1].text,
                BASELINE: {**baseline, "score": score(case, baseline)},
                REIN: {**steered, "score": score(case, steered)},
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                json.dumps({"case": number, "id": case.record_id, "base": baseline["tool"], "rein": steered["tool"]}),
                flush=True,
            )

    rows = [json.loads(line) for line in results_path.open(encoding="utf-8")]
    summary = summarize(rows)
    (args.out / SUMMARY_FILE).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
