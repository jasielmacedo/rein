"""More cases for training tool rules, aimed at their weak spot: requests that sound like an action but need no tool.

For each chosen tool rule (training split, the oracle's rules left out), the teacher writes two new kinds of negative,
a request no tool of the sheet can serve and a mention of a tool's subject without a request, plus the taxonomy's own
hard negative and one case that needs a tool, for naming. The cases are judged like any other, then audited.
"""

import argparse
import asyncio
import json
import random
import time
from dataclasses import replace
from pathlib import Path

from rein.compiler.holdout import load_rule_pairs
from rein.compiler.oracle_data import TEACHER_APP, OracleRule, choose_rules, export_records, extend_rules
from rein.datagen.models import ScenarioDraft
from rein.datagen.plan import CaseSpec, plan_cases, sample_turns, target_option
from rein.datagen.taxonomy import CaseForm, Taxonomy, load_taxonomy
from rein.teacher import constants as tc
from rein.teacher.modal_ops import stop_app_containers
from rein.train.data import load_records

RUN_NAME = "toolneg-v1"
PREFIX = "toolneg-"
TOOL_CATEGORY = "tool_trigger"
LOOKALIKE, MENTION, HARD_NEGATIVE = "tool_lookalike", "tool_mention", "hard_negative"
EXTRA_FORMS = {
    LOOKALIKE: CaseForm(
        fires=False,
        guidance=(
            "The latest message asks for something that sounds like an action or a service (looking something up, "
            "sending, booking, checking, fixing, calling someone), but none of the scenario's tools can do it, so no "
            "tool may be called. It must not be a request for one of the scenario's tools in other words."
        ),
    ),
    MENTION: CaseForm(
        fires=False,
        guidance=(
            "The latest message talks about something one of the tools does (its object, its history, how it works, "
            "a past use) without asking for that action now, so no tool may be called."
        ),
    ),
}
SCHEDULE: tuple[str | None, ...] = (LOOKALIKE, LOOKALIKE, MENTION, HARD_NEGATIVE, None)
POSITIVE_FORMS = ("direct", "paraphrase", "indirect", "slang", "spread_over_turns")
DEFAULT_RULES = 400
DEFAULT_WORKERS = 32
SEED = 0
SECONDS_PER_HOUR = 3600


def with_tool_forms(taxonomy: Taxonomy) -> Taxonomy:
    return taxonomy.model_copy(update={"case_forms": {**taxonomy.case_forms, **EXTRA_FORMS}})


def tool_cases(taxonomy: Taxonomy, rule: OracleRule, scenario: ScenarioDraft, scenario_id: str) -> list[CaseSpec]:
    """One case per slot of SCHEDULE for the rule, answered at the reply start; a None slot needs a tool."""
    rng = random.Random(f"{SEED}:{scenario_id}:{RUN_NAME}")
    target = next(item for item in scenario.rules if item.id == rule.rule_id)
    planned = plan_cases(taxonomy, scenario_id, scenario, len(SCHEDULE), SEED, (TOOL_CATEGORY,))
    cases = []
    for spec, slot in zip(planned, SCHEDULE, strict=True):
        form = slot or rng.choice(POSITIVE_FORMS)
        cases.append(
            replace(
                spec,
                target_rule=target.id,
                case_form=form,
                target_option=target_option(target, taxonomy.case_forms[form].fires, rng),
                turns=sample_turns(taxonomy, form, rng),
                draft_kind=None,
                truncate_draft=False,
                second_rule=None,
                second_option=None,
            )
        )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate hard negatives and naming cases for training tool rules.")
    parser.add_argument("--data", type=Path, required=True, help="the audited mix, to pick training tool rules from")
    parser.add_argument("--synthetic", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, default=None, help="rules to leave out, e.g. the oracle's rules.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rules", type=int, default=DEFAULT_RULES)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--keep-teacher", action="store_true")
    args = parser.parse_args()
    taxonomy = with_tool_forms(load_taxonomy())
    exclude = frozenset(load_rule_pairs(args.exclude))
    rules = choose_rules(
        load_records(args.data / "train.jsonl"), args.synthetic, (TOOL_CATEGORY,), args.rules, SEED, exclude
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "rules.json").write_text(
        json.dumps(
            [{"scenario_id": r.scenario_id, "rule_id": r.rule_id, "category": r.category} for r in rules], indent=2
        ),
        encoding="utf-8",
    )
    print(json.dumps({"rules": len(rules), "cases_planned": len(rules) * len(SCHEDULE)}), flush=True)

    def planner(rule: OracleRule, scenario: ScenarioDraft, scenario_id: str) -> list[CaseSpec]:
        return tool_cases(taxonomy, rule, scenario, scenario_id)

    started = time.monotonic()
    try:
        stats = asyncio.run(extend_rules(rules, (args.out, RUN_NAME, PREFIX), args.workers, taxonomy, planner))
        print(json.dumps(stats), flush=True)
    finally:
        if not args.keep_teacher:
            print("stopped containers:", stop_app_containers(TEACHER_APP), flush=True)
        seconds = time.monotonic() - started
        usd = seconds / SECONDS_PER_HOUR * tc.ENDPOINT_GPU_USD_PER_HOUR
        print(json.dumps({"teacher_seconds": round(seconds), "estimated_usd": round(usd, 2)}), flush=True)
    print(json.dumps({"records": export_records(args.out, args.out / "train.jsonl", RUN_NAME)}), flush=True)


if __name__ == "__main__":
    main()
