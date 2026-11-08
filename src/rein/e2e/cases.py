import json
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rein.control.sheet import ForceTool, Gate, Prefill, Reinject, Rule, RuleSheet
from rein.datagen.audit import reference_option
from rein.schema import Question, State, TrainingRecord

TOOL_CATEGORY = "tool_trigger"
LOCK_CATEGORY = "locked_attribute"
E2E_CATEGORIES = (TOOL_CATEGORY, LOCK_CATEGORY)
E2E_LANGUAGES = ("en", "pt-BR")
NO_TOOL_KEYS = frozenset({"none", "no_tool", "other"})
IDENTITY_OPENERS = {
    "en": "$name here, and that part of me doesn't change. ",
    "pt-BR": "Aqui é $name, e isso em mim não muda. ",
}
WATCH_RULE_ID = "watch_narrated_action"
WATCH_INSTRUCTIONS = (
    "Is the assistant's reply in progress narrating an action as if it already happened, without emitting the "
    "tool call that performs it? If so, which tool would perform it?"
)
RULES_HEADER = "Follow these rules in every reply:"


@dataclass(frozen=True)
class E2ECase:
    record_id: str
    category: str
    case_form: str
    language: str
    state: State
    sheet: RuleSheet
    system: str
    tools: list[dict[str, Any]]
    expected_tool: str | None = None
    attack: bool = False
    character_name: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def scenario_index(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    """Maps each rule's instructions to the scenario that wrote it, which knows the tools and rule categories."""
    index: dict[str, dict[str, Any]] = {}
    for path in paths:
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            if not row.get("ok"):
                continue
            scenario = row["value"]
            for rule in scenario["rules"]:
                index[rule["instructions"]] = scenario
    return index


def _tool_schema(name: str, description: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": {"type": "object", "properties": {}}},
    }


def _tool_options(question: Question, tools: Sequence[str]) -> tuple[str, ...]:
    if question.type != "choice" or not isinstance(question.criteria, dict):
        return ()
    return tuple(key for key in question.criteria if key in tools and key not in NO_TOOL_KEYS)


def _firing(question: Question) -> tuple[str, ...]:
    """The options that mean a rule applies: yes, any named option but none, or the upper half of a scale."""
    keys = question.option_keys()
    if question.type == "noul":
        return ("yes",)
    if question.type == "score":
        return keys[len(keys) // 2 :]
    return tuple(key for key in keys if key not in NO_TOOL_KEYS) or keys


def _unlabeled(question: Question) -> Question:
    return question.model_copy(update={"label": None, "target": None})


def _rule(qid: str, question: Question, category: str, scenario: dict[str, Any], language: str) -> Rule:
    tool_options = _tool_options(question, scenario["tools"])
    name = scenario.get("character_name")
    if category == TOOL_CATEGORY and tool_options:
        return Rule(
            id=qid,
            question=_unlabeled(question),
            fire_on=tool_options,
            action=ForceTool(tools={key: key for key in tool_options}),
        )
    if category == LOCK_CATEGORY and question.type == "noul" and name and language in IDENTITY_OPENERS:
        return Rule(id=qid, question=_unlabeled(question), action=Prefill(text=IDENTITY_OPENERS[language]))
    return Rule(id=qid, question=_unlabeled(question), fire_on=_firing(question), action=Reinject(), gate=Gate())


def _watch_rule(tools: Sequence[str]) -> Rule:
    criteria: dict[str, str | None] = {tool: f"the reply narrates doing what {tool} does" for tool in tools}
    criteria["none"] = "the reply narrates no unexecuted action"
    return Rule(
        id=WATCH_RULE_ID,
        scope="draft",
        question=Question(type="choice", instructions=WATCH_INSTRUCTIONS, criteria=criteria),
        fire_on=tuple(tools),
        action=ForceTool(tools={tool: tool for tool in tools}),
    )


def _descriptions(record: TrainingRecord, tools: Sequence[str]) -> dict[str, str]:
    found = {tool: tool.replace("_", " ") for tool in tools}
    for question in record.questions.values():
        if isinstance(question.criteria, dict):
            for key, text in question.criteria.items():
                if key in found and text:
                    found[key] = text
    return found


def build_case(record: TrainingRecord, scenario: dict[str, Any]) -> E2ECase | None:
    language = str(record.meta.get("lang_conv"))
    target = str(record.meta.get("target_rule"))
    category = str(record.meta.get("category"))
    categories = {rule["id"]: rule["category"] for rule in scenario["rules"]}
    tools = list(scenario.get("tools") or [])
    rules = [
        _rule(qid, question, categories.get(qid, ""), scenario, language) for qid, question in record.questions.items()
    ]
    target_rule = next(rule for rule in rules if rule.id == target)
    if category == TOOL_CATEGORY and not isinstance(target_rule.action, ForceTool):
        return None
    if category == LOCK_CATEGORY and not isinstance(target_rule.action, Prefill):
        return None
    if tools:
        rules.append(_watch_rule(tools))
    reference = reference_option(record.questions[target])
    system = "\n".join([RULES_HEADER, *(f"- {question.instructions}" for question in record.questions.values())])
    return E2ECase(
        record_id=record.id,
        category=category,
        case_form=str(record.meta.get("case_form")),
        language=language,
        state=record.state,
        sheet=RuleSheet(rules=tuple(rules), variables={"name": scenario.get("character_name") or ""}),
        system=system,
        tools=[_tool_schema(name, text) for name, text in _descriptions(record, tools).items()],
        expected_tool=reference if category == TOOL_CATEGORY and reference in tools else None,
        attack=category == LOCK_CATEGORY and reference == "yes",
        character_name=scenario.get("character_name"),
        meta={"domain": record.meta.get("domain"), "holdout": record.meta.get("holdout")},
    )


def build_cases(
    records: Sequence[TrainingRecord], index: dict[str, dict[str, Any]], per_category: int | None, seed: int
) -> list[E2ECase]:
    """Request-time cases (no draft yet) in English and Portuguese for tool triggers and character locks."""
    pool: dict[str, list[E2ECase]] = {category: [] for category in E2E_CATEGORIES}
    for record in records:
        meta = record.meta
        if meta.get("category") not in E2E_CATEGORIES or meta.get("lang_conv") not in E2E_LANGUAGES:
            continue
        if record.state.draft is not None:
            continue
        scenario = index.get(record.questions[str(meta.get("target_rule"))].instructions)
        case = build_case(record, scenario) if scenario else None
        if case is not None:
            pool[case.category].append(case)
    rng = random.Random(seed)
    chosen: list[E2ECase] = []
    for cases in pool.values():
        rng.shuffle(cases)
        chosen.extend(cases[:per_category] if per_category else cases)
    return chosen
