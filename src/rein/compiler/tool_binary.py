"""Tool rules as yes/no questions: "does the request need a tool now?" and one "does it need tool X now?" per tool.

The cached states stay the same; only the questions asked about them change. Each derived question keeps the base of
the tool rule it comes from, and its id names that rule, so slices can find it.
"""

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

from rein.genview.keys import rule_key
from rein.genview.scenarios import ScenarioIndex
from rein.schema import Question, TrainingRecord
from rein.schema.constants import MAX_OPTION_TEXT_CHARS
from rein.train.data import load_records

DERIVED_MARK = "::"
ANY_SUFFIX = f"{DERIVED_MARK}any"
EACH_SEPARATOR = f"{DERIVED_MARK}tool="
TOOL_CATEGORY = "tool_trigger"
NULL_TOOLS = frozenset({"none", "general"})
SPLIT_FILES = ("train.jsonl", "development.jsonl", "calibration.jsonl", "test.jsonl")
ANY_INSTRUCTIONS = "Does the latest request need one of the available tools to be called now?"
EACH_INSTRUCTIONS = "Does the latest request need the tool `{tool}` to be called now?"
CRITERIA_SEPARATOR = "; "


def parent_question(question_id: str) -> str:
    """The tool rule a derived question comes from, or the id itself for any other question."""
    return question_id.split(DERIVED_MARK, 1)[0]


def is_derived(question_id: str) -> bool:
    return parent_question(question_id) != question_id


def _fitted(parts: list[str]) -> str:
    """The parts joined, each cut to an even share when the whole would not fit in one option text."""
    joined = CRITERIA_SEPARATOR.join(parts)
    if len(joined) <= MAX_OPTION_TEXT_CHARS:
        return joined
    share = (MAX_OPTION_TEXT_CHARS - len(CRITERIA_SEPARATOR) * (len(parts) - 1)) // len(parts)
    return CRITERIA_SEPARATOR.join(part[:share] for part in parts)


def any_tool_criteria(criteria: Mapping[str, str | None]) -> dict[str, str | None] | None:
    """The sheet's own words for when one of its tools is called and when none is."""
    called = [f"{tool}: {text}" if text else tool for tool, text in criteria.items() if tool not in NULL_TOOLS]
    idle = [text for tool, text in criteria.items() if tool in NULL_TOOLS and text]
    described: dict[str, str | None] = {
        flag: _fitted(parts) for flag, parts in (("true", called), ("false", idle)) if parts
    }
    return described or None


def derived_questions(question_id: str, question: Question) -> dict[str, Question]:
    """The yes/no questions a labelled tool choice implies, with the same answer."""
    keys = question.option_keys()
    index = question.label_index()
    if question.type != "choice" or index is None:
        return {}
    label = keys[index]
    criteria = question.criteria if isinstance(question.criteria, dict) else {}
    derived = {
        f"{question_id}{ANY_SUFFIX}": Question(
            type="noul",
            instructions=ANY_INSTRUCTIONS,
            criteria=any_tool_criteria(criteria),
            label=label not in NULL_TOOLS,
        )
    }
    for tool in keys:
        if tool in NULL_TOOLS:
            continue
        described = criteria.get(tool)
        derived[f"{question_id}{EACH_SEPARATOR}{tool}"] = Question(
            type="noul",
            instructions=EACH_INSTRUCTIONS.format(tool=tool),
            criteria={"true": described} if described else None,
            label=label == tool,
        )
    return derived


def with_tool_questions(
    records: Mapping[str, TrainingRecord], tool_rules: Mapping[str, str]
) -> dict[str, TrainingRecord]:
    """Records whose tool rule also carries its yes/no versions."""
    updated: dict[str, TrainingRecord] = {}
    for record_id, record in records.items():
        question_id = tool_rules.get(record_id)
        question = record.questions.get(question_id) if question_id else None
        if question_id is None or question is None:
            updated[record_id] = record
            continue
        questions = {**record.questions, **derived_questions(question_id, question)}
        updated[record_id] = record.model_copy(update={"questions": questions})
    return updated


def derived_bases(records: Mapping[str, TrainingRecord], bases: Mapping[str, str]) -> dict[str, str]:
    """Bases for the derived questions: each borrows the base of its tool rule."""
    extra: dict[str, str] = {}
    for record in records.values():
        for question_id, question in record.questions.items():
            parent_id = parent_question(question_id)
            if not is_derived(question_id) or parent_id not in record.questions:
                continue
            parent_base = bases.get(rule_key(record.questions[parent_id]))
            if parent_base is not None:
                extra.setdefault(rule_key(question), parent_base)
    return extra


def load_tool_rules(path: Path | None) -> dict[str, str]:
    return {} if path is None else json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Find each record's tool rule, by the category of its scenario.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--synthetic", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    index = ScenarioIndex.from_directory(args.synthetic)
    tool_rules: dict[str, str] = {}
    for name in SPLIT_FILES:
        if not (args.data / name).exists():
            continue
        for record in load_records(args.data / name):
            info = index.lookup(record)
            if info is None:
                continue
            found = [qid for qid in record.questions if info.categories.get(qid) == TOOL_CATEGORY]
            if len(found) == 1:
                tool_rules[record.id] = found[0]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(tool_rules), encoding="utf-8")
    print(json.dumps({"records_with_a_tool_rule": len(tool_rules)}))


if __name__ == "__main__":
    main()
