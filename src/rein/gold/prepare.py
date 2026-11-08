"""Turns a saved gold project into what the evaluation reads: records, bases, tool rules and the generator's views.

Bases the author wrote are used as they are (and linted); rules left without one get a base from the teacher endpoint
when `--write-bases` is given. Every problem is printed, so an unfinished project can still be prepared.
"""

import argparse
import asyncio
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path

from transformers import AutoTokenizer

from rein.compiler.category_probe import record_categories
from rein.control.base import base_problems
from rein.datagen.generate import Stats
from rein.genview import constants as gc
from rein.genview.bases import TEACHER_APP, unique_rules, write_bases
from rein.genview.build import GOLD_FILE, NO_THINKING, build_views
from rein.genview.keys import load_bases
from rein.genview.render import GeneratorView
from rein.gold.project import GoldProject, build_records, gold_index, sheet_bases
from rein.schema import TrainingRecord
from rein.teacher.client import TeacherClient
from rein.teacher.modal_ops import stop_app_containers
from rein.teacher.settings import TeacherSettings
from rein.teacher.store import JsonlStore

TEACHER_BASES_FILE = "teacher-bases.jsonl"
BASES_FILE = "bases.jsonl"
TOOL_RULES_FILE = "tool_rules.json"
VIEWS_FILE = "views.jsonl"
CATEGORIES_FILE = "categories.json"
TOOL_CATEGORY = "tool_trigger"
TEACHER_WORKERS = 8


def load_project(path: Path) -> GoldProject:
    return GoldProject.model_validate_json(path.read_text(encoding="utf-8"))


def tool_rules(project: GoldProject, records: Iterable[TrainingRecord]) -> dict[str, str]:
    """Each record's tool rule (a list rule of category tool_trigger), so its yes/no versions can be asked."""
    tools = {(sheet.id, rule.id) for sheet in project.sheets for rule in sheet.rules if rule.category == TOOL_CATEGORY}
    mapping: dict[str, str] = {}
    for record in records:
        found = [
            qid
            for qid, question in record.questions.items()
            if (record.scenario_id, qid) in tools and question.type == "choice"
        ]
        if len(found) == 1:
            mapping[record.id] = found[0]
    return mapping


def base_lint(project: GoldProject) -> list[str]:
    return [
        f"base of {sheet.id}/{rule.id}: {problem}"
        for sheet in project.sheets
        for rule in sheet.rules
        if rule.base.strip()
        for problem in base_problems(rule.base)
    ]


async def _teacher_bases(records: list[TrainingRecord], known: Mapping[str, str], store: JsonlStore) -> Stats:
    missing = {key: rule for key, rule in unique_rules(records).items() if key not in known}
    stats = Stats()
    if not missing:
        return stats
    async with TeacherClient(TeacherSettings.from_env(), max_in_flight=TEACHER_WORKERS) as client:
        print(f"teacher healthy after {await client.wait_healthy():.0f}s", flush=True)
        await write_bases(client, missing, store, stats)
    return stats


def write_records(records: Iterable[TrainingRecord], path: Path) -> None:
    path.write_text("".join(record.model_dump_json() + "\n" for record in records), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a gold project from tools/gold for evaluation.")
    parser.add_argument("--project", type=Path, required=True, help="the .json saved by the gold editor")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--write-bases", action="store_true", help="ask the teacher for the bases the author left out")
    parser.add_argument("--keep-teacher", action="store_true")
    parser.add_argument("--modes", nargs="+", default=[NO_THINKING])
    parser.add_argument("--prompt", choices=gc.PROMPTS, default=gc.PROMPT_BASES, help="how the rules reach Qwen")
    parser.add_argument("--chat-template", type=Path, default=None, help="render with this template instead")
    args = parser.parse_args()
    project = load_project(args.project)
    built = build_records(project)
    args.out.mkdir(parents=True, exist_ok=True)
    write_records(built.records, args.out / GOLD_FILE)
    teacher_store = JsonlStore(args.out / TEACHER_BASES_FILE)
    if args.write_bases:
        try:
            stats = asyncio.run(
                _teacher_bases(
                    built.records, {**load_bases(args.out / TEACHER_BASES_FILE), **sheet_bases(project)}, teacher_store
                )
            )
            print(json.dumps({"teacher_bases": dict(stats.accepted), "rejected": dict(stats.rejected)}), flush=True)
        finally:
            if not args.keep_teacher:
                print("stopped containers:", stop_app_containers(TEACHER_APP), flush=True)
    bases = {**load_bases(args.out / TEACHER_BASES_FILE), **sheet_bases(project)}
    (args.out / BASES_FILE).write_text(
        "".join(json.dumps({"key": key, "base": base}) + "\n" for key, base in bases.items()), encoding="utf-8"
    )
    (args.out / TOOL_RULES_FILE).write_text(json.dumps(tool_rules(project, built.records)), encoding="utf-8")
    template = args.chat_template.read_text(encoding="utf-8") if args.chat_template else None
    view = GeneratorView(AutoTokenizer.from_pretrained(gc.CHAT_TOKENIZER_REPO), chat_template=template)
    index = gold_index(project)
    by_id = {record.id: record for record in built.records}
    (args.out / CATEGORIES_FILE).write_text(json.dumps(record_categories(by_id, index)), encoding="utf-8")
    counts: Counter[str] = Counter()
    with (args.out / VIEWS_FILE).open("w", encoding="utf-8") as handle:
        for item in build_views(built.records, bases, index, view, args.prompt, args.modes):
            if isinstance(item, str):
                counts["without_base"] += 1
                continue
            handle.write(item.to_json() + "\n")
            counts[item.mode] += 1
    problems = built.problems + base_lint(project)
    print(
        json.dumps(
            {"records": len(built.records), "views": dict(counts), "problems": problems}, indent=2, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
