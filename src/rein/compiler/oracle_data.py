"""Oracle test, part 1: many fresh cases for a few training rules, generated and judged like the rest of the data."""

import argparse
import asyncio
import json
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rein.compiler.constants import ORACLE_PREFIX
from rein.datagen.generate import ChatModel
from rein.datagen.models import ScenarioDraft
from rein.datagen.pipeline import DatagenConfig, Pipeline, SliceOverrides
from rein.datagen.plan import CaseSpec, ScenarioSpec, plan_cases
from rein.datagen.taxonomy import Taxonomy, load_taxonomy
from rein.genview import constants as gc
from rein.genview.scenarios import scenario_key
from rein.schema import TrainingRecord
from rein.teacher.client import TeacherClient
from rein.teacher.modal_ops import TEACHER_APP, stop_app_containers
from rein.teacher.settings import TeacherSettings
from rein.train.data import load_records

ORACLE_RUN = "oracle-v1"
DEFAULT_CATEGORIES = ("npc_secret", "lore_consistency", "tool_trigger")
DEFAULT_RULES_PER_CATEGORY = 5
DEFAULT_CASES_PER_RULE = 60
DEFAULT_WORKERS = 32
SEED = 0
JUDGE_COLD_START_S = 900.0
RECORDS_FILE = "records.jsonl"


@dataclass(frozen=True)
class OracleRule:
    """A training rule the oracle studies, with the scenario it lives in and how that scenario was planned."""

    scenario_id: str
    rule_id: str
    category: str
    scenario: dict[str, Any]
    meta: dict[str, Any]


def _raw_scenarios(synthetic_dir: Path) -> dict[tuple[str, frozenset[str]], dict[str, Any]]:
    raw: dict[tuple[str, frozenset[str]], dict[str, Any]] = {}
    for path in sorted(synthetic_dir.glob(gc.SCENARIO_GLOB)):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                value = row.get("value")
                if row.get("ok") and isinstance(value, dict):
                    raw.setdefault(scenario_key(value["context"], (r["id"] for r in value["rules"])), value)
    return raw


def choose_rules(
    records: Sequence[TrainingRecord],
    synthetic_dir: Path,
    categories: Sequence[str],
    per_category: int,
    seed: int = SEED,
    exclude: frozenset[tuple[str, str]] = frozenset(),
) -> list[OracleRule]:
    """Training rules of the given categories, each the only rule of its category in its scenario, from distinct
    domains where possible."""
    raw = _raw_scenarios(synthetic_dir)
    seen: set[str] = set()
    candidates: dict[str, list[OracleRule]] = {category: [] for category in categories}
    for record in records:
        if record.split != "train" or record.source != "synthetic" or record.scenario_id in seen:
            continue
        seen.add(record.scenario_id)
        value = raw.get(scenario_key(record.state.context, record.questions))
        if value is None:
            continue
        for category in categories:
            rules = [rule for rule in value["rules"] if rule["category"] == category]
            if len(rules) == 1 and (record.scenario_id, rules[0]["id"]) not in exclude:
                candidates[category].append(
                    OracleRule(record.scenario_id, rules[0]["id"], category, value, record.meta)
                )
    rng = random.Random(seed)
    chosen: list[OracleRule] = []
    for options in candidates.values():
        rng.shuffle(options)
        chosen.extend(_spread_over_domains(options, per_category))
    return chosen


def _spread_over_domains(options: Sequence[OracleRule], count: int) -> list[OracleRule]:
    """Up to `count` rules, one per domain first, then the rest in order."""
    picked: list[OracleRule] = []
    domains: set[object] = set()
    for rule in options:
        if rule.meta.get("domain") not in domains:
            picked.append(rule)
            domains.add(rule.meta.get("domain"))
    picked += [rule for rule in options if rule not in picked]
    return picked[:count]


CasePlanner = Callable[[OracleRule, ScenarioDraft, str], list[CaseSpec]]


async def extend_rules(
    rules: Sequence[OracleRule],
    run: tuple[Path, str, str],
    workers: int,
    taxonomy: Taxonomy,
    planner: CasePlanner,
    judge: TeacherSettings | None = None,
    writer: ChatModel | None = None,
) -> dict[str, Any]:
    """New judged cases for existing rules, written into `run` (folder, run name, scenario prefix); `planner` decides
    the cases of each rule. The teacher writes and judges unless another `judge` endpoint or `writer` is given."""
    out_dir, run_name, prefix = run
    config = DatagenConfig(
        run_name=run_name,
        scenarios=len(rules),
        cases_per_scenario=1,
        seed=SEED,
        max_in_flight=workers,
        output_dir=out_dir,
        overrides=SliceOverrides(force_split="train"),
    )
    async with TeacherClient(judge or TeacherSettings.from_env(), max_in_flight=workers) as client:
        print(f"judge healthy after {await client.wait_healthy(JUDGE_COLD_START_S):.0f}s", flush=True)
        pipeline = Pipeline(config, client, taxonomy, writer)

        async def one(rule: OracleRule) -> None:
            scenario = ScenarioDraft.model_validate(rule.scenario)
            spec = ScenarioSpec(
                scenario_id=f"{prefix}{rule.scenario_id}",
                domain=str(rule.meta.get("domain")),
                categories=tuple(item.category for item in scenario.rules),
                rule_language=str(rule.meta.get("lang_rule")),
                rule_style=str(rule.meta.get("rule_style")),
                holdout=False,
            )
            await pipeline.extend(spec, scenario, planner(rule, scenario, spec.scenario_id))

        await asyncio.gather(*(one(rule) for rule in rules))
        return {"accepted": dict(pipeline.stats.accepted), "rejected": dict(pipeline.stats.rejected)}


async def generate(rules: Sequence[OracleRule], out_dir: Path, cases_per_rule: int, workers: int) -> dict[str, Any]:
    taxonomy = load_taxonomy()

    def planner(rule: OracleRule, scenario: ScenarioDraft, scenario_id: str) -> list[CaseSpec]:
        cases = plan_cases(taxonomy, scenario_id, scenario, cases_per_rule, SEED, (rule.category,))
        return [case for case in cases if case.target_rule == rule.rule_id]

    return await extend_rules(rules, (out_dir, ORACLE_RUN, ORACLE_PREFIX), workers, taxonomy, planner)


def export_records(out_dir: Path, target: Path, run_name: str = ORACLE_RUN) -> int:
    """The judged records of a run as a `train.jsonl` the view builder reads."""
    rows = (json.loads(line) for line in (out_dir / run_name / RECORDS_FILE).read_text(encoding="utf-8").splitlines())
    records = [TrainingRecord.model_validate({k: v for k, v in row.items() if k != "key"}) for row in rows]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(record.model_dump_json() + "\n" for record in records), encoding="utf-8")
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate many judged cases for a few training rules.")
    parser.add_argument("--data", type=Path, required=True, help="the audited mix, to pick training rules from")
    parser.add_argument("--synthetic", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="where the oracle run and its train.jsonl go")
    parser.add_argument("--categories", nargs="+", default=list(DEFAULT_CATEGORIES))
    parser.add_argument("--per-category", type=int, default=DEFAULT_RULES_PER_CATEGORY)
    parser.add_argument("--cases", type=int, default=DEFAULT_CASES_PER_RULE)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--keep-teacher", action="store_true")
    args = parser.parse_args()
    rules = choose_rules(load_records(args.data / "train.jsonl"), args.synthetic, args.categories, args.per_category)
    (args.out).mkdir(parents=True, exist_ok=True)
    (args.out / "rules.json").write_text(
        json.dumps(
            [{"scenario_id": r.scenario_id, "rule_id": r.rule_id, "category": r.category} for r in rules], indent=2
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"rules": len(rules), "by_category": {c: sum(r.category == c for r in rules) for c in args.categories}}
        )
    )
    try:
        stats = asyncio.run(generate(rules, args.out, args.cases, args.workers))
        print(json.dumps(stats), flush=True)
    finally:
        if not args.keep_teacher:
            print("stopped containers:", stop_app_containers(TEACHER_APP))
    print(json.dumps({"records": export_records(args.out, args.out / "train.jsonl")}))


if __name__ == "__main__":
    main()
