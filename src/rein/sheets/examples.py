"""Examples for the per-sheet fit (DR-025): cases written for each rule of an end-to-end sheet, then judged.

Two sources, compared in the fit: the teacher writes and judges (`teacher`), or Qwen3.5-4B writes the cases in one
batch on Modal (`prompts`, then `rein.sheets.qwen_writer`) and the auditor judges them (`judge`), with no teacher at
all. Either way the records land in `<out>/<source>/train.jsonl`.
"""

import argparse
import asyncio
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rein.compiler.oracle_data import OracleRule, export_records, extend_rules
from rein.datagen.audit import AUDITOR_APP, auditor_settings
from rein.datagen.models import ScenarioDraft
from rein.datagen.plan import CaseSpec, plan_cases
from rein.datagen.prompts import case_prompt
from rein.datagen.taxonomy import Taxonomy, load_taxonomy
from rein.sheets.model import Sheet, as_scenario, load_sheet
from rein.teacher.client import Completion, TeacherError
from rein.teacher.modal_ops import TEACHER_APP, stop_app_containers

TEACHER_SOURCE, QWEN_SOURCE = "teacher", "qwen"
PROMPTS_FILE = "qwen-prompts.jsonl"
DEFAULT_PER_RULE = 60
DEFAULT_WORKERS = 64
SEED = 0
RULE_STYLE = "a short direct question"
DOMAIN = "end-to-end NPC sheet"


def sheet_rules(sheet: Sheet) -> list[OracleRule]:
    scenario = as_scenario(sheet).model_dump()
    meta = {"domain": DOMAIN, "lang_rule": sheet.language, "rule_style": RULE_STYLE}
    return [OracleRule(sheet.id, rule.id, rule.category, scenario, meta) for rule in sheet.rules]


def rule_cases(taxonomy: Taxonomy, rule: OracleRule, scenario: ScenarioDraft, per_rule: int) -> list[CaseSpec]:
    """`per_rule` cases aimed at one rule, with the forms, languages and lengths the taxonomy samples."""
    shared = sum(item.category == rule.category for item in scenario.rules)
    planned = plan_cases(
        taxonomy, f"{rule.scenario_id}-{rule.rule_id}", scenario, per_rule * shared, SEED, (rule.category,)
    )
    return [case for case in planned if case.target_rule == rule.rule_id][:per_rule]


def case_prompts(taxonomy: Taxonomy, sheet: Sheet, per_rule: int) -> list[dict[str, Any]]:
    """Each planned case with the exact prompt the case writer answers."""
    scenario = as_scenario(sheet)
    rows = []
    for rule in sheet_rules(sheet):
        for case in rule_cases(taxonomy, rule, scenario, per_rule):
            rows.append(
                {"case_id": case.case_id, "prompt": case_prompt(taxonomy, scenario, case), "spec": asdict(case)}
            )
    return rows


class ReplayModel:
    """A case writer that answers each prompt with what Qwen3.5-4B already wrote for it in the batch on Modal."""

    def __init__(self, written: Mapping[str, str]) -> None:
        self._written = dict(written)

    async def chat(self, messages: Sequence[Mapping[str, str]], **_: Any) -> Completion:
        content = self._written.get(messages[-1]["content"])
        if content is None:
            raise TeacherError("giving up: Qwen wrote nothing valid for this case")
        return Completion(content=content, prompt_tokens=0, completion_tokens=0, finish_reason="stop", latency_s=0.0)


def load_written(path: Path) -> dict[str, str]:
    rows = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return {row["prompt"]: row["content"] for row in rows if row.get("content")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Write and judge examples for the rules of an end-to-end sheet.")
    parser.add_argument("step", choices=("prompts", TEACHER_SOURCE, "judge"))
    parser.add_argument("--sheet", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-rule", type=int, default=DEFAULT_PER_RULE)
    parser.add_argument("--written", type=Path, default=None, help="the Qwen writer's output, for the judge step")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()
    sheet = load_sheet(args.sheet)
    taxonomy = load_taxonomy()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.step == "prompts":
        rows = case_prompts(taxonomy, sheet, args.per_rule)
        (args.out / PROMPTS_FILE).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )
        print(json.dumps({"prompts": len(rows)}), flush=True)
        return
    source = TEACHER_SOURCE if args.step == TEACHER_SOURCE else QWEN_SOURCE
    run_name = f"{sheet.id}-{source}"

    def planner(rule: OracleRule, scenario: ScenarioDraft, scenario_id: str) -> list[CaseSpec]:
        return rule_cases(taxonomy, rule, scenario, args.per_rule)

    judge = None if source == TEACHER_SOURCE else auditor_settings()
    writer = None if source == TEACHER_SOURCE else ReplayModel(load_written(args.written))
    app = TEACHER_APP if source == TEACHER_SOURCE else AUDITOR_APP
    try:
        stats = asyncio.run(
            extend_rules(sheet_rules(sheet), (args.out, run_name, ""), args.workers, taxonomy, planner, judge, writer)
        )
        print(json.dumps(stats), flush=True)
    finally:
        print("stopped containers:", stop_app_containers(app), flush=True)
    count = export_records(args.out, args.out / source / "train.jsonl", run_name)
    print(json.dumps({"source": source, "records": count}), flush=True)


if __name__ == "__main__":
    main()
