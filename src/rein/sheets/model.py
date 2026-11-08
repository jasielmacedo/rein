"""The sheets compiled into the weights for the end-to-end tests: the gold editor's sheet plus, per rule, the action the
compiled neurons take, the tools' full schemas, and a long scripted conversation with the expected decisions.

One file per NPC feeds every step: the cases written for the per-sheet fit (as a scenario), the compile, the gold
editor (as a sheet), and the runner (the script).
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from rein.compiler.constants import NULL_OPTIONS
from rein.datagen.models import RuleDraft, ScenarioDraft
from rein.gold.project import GoldMessage, GoldRule, GoldSheet, GoldTool

NO_HINT = "nothing in the latest turn calls for it"
QUESTION_TYPES = {"yes_no": "noul", "choice": "choice", "scale": "score"}


class SteerAction(BaseModel):
    """Add a behavior vector to the residual stream while the rule fires."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["steer"] = "steer"
    vector: str


class ToolAction(BaseModel):
    """DR-024: force `<tool_call>` when the any-tool question is sure, veto it when it is sure there is no call."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["tool"] = "tool"


class CallAction(BaseModel):
    """Open a tool call where the reply narrates an action instead of performing it (a rule on the NPC's reply)."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["call"] = "call"


RuleAction = SteerAction | ToolAction | CallAction


class SheetTool(GoldTool):
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})

    def function_spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }


class SheetRule(GoldRule):
    action: RuleAction = Field(discriminator="kind")


class Sheet(GoldSheet):
    tools: Sequence[SheetTool] = Field(default_factory=list)
    rules: Sequence[SheetRule] = Field(default_factory=list)
    script: list[GoldMessage] = Field(default_factory=list)
    tool_result: str = "done"
    keep_name: str | None = None
    forbidden_names: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _actions_fit_rules(self) -> "Sheet":
        tools = {tool.name for tool in self.tools}
        for rule in self.rules:
            if isinstance(rule.action, ToolAction | CallAction):
                firing = {key for key in rule.option_keys() if key not in NULL_OPTIONS}
                if rule.kind != "choice" or not firing <= tools:
                    raise ValueError(f"{rule.id}: a tool action needs a list rule whose options are the sheet's tools")
            if isinstance(rule.action, CallAction) and rule.looks_at != "npc_reply":
                raise ValueError(f"{rule.id}: a call action watches the NPC's reply")
        return self

    def system(self, lines: list[str]) -> str:
        """The system prompt: persona, scene, then one line per rule (bases, or whole rules for the full-prompt arm)."""
        return "\n\n".join(part for part in (self.context(), "\n".join(f"- {line}" for line in lines)) if part.strip())

    def tool_schemas(self) -> list[dict[str, Any]]:
        return [tool.function_spec() for tool in self.tools]


def load_sheet(path: Path) -> Sheet:
    return Sheet.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def tool_rule(sheet: Sheet) -> str:
    return next(rule.id for rule in sheet.rules if isinstance(rule.action, ToolAction))


def expected_tool(sheet: Sheet, turn: dict[str, Any]) -> str | None:
    answer = turn["answers"].get(tool_rule(sheet))
    return None if answer is None or answer in NULL_OPTIONS else answer


def _hints(rule: SheetRule) -> tuple[str, str]:
    if rule.kind == "yes_no":
        return rule.yes or rule.question, rule.no or NO_HINT
    if rule.kind == "choice":
        firing = [
            f"{option.key}: {option.when}" if option.when else option.key
            for option in rule.options
            if option.key not in NULL_OPTIONS
        ]
        quiet = [option.when for option in rule.options if option.key in NULL_OPTIONS and option.when]
        return "; ".join(firing), (quiet[0] if quiet else NO_HINT)
    return rule.levels[-1], rule.levels[0]


def as_scenario(sheet: Sheet) -> ScenarioDraft:
    """The sheet as a datagen scenario, so the case writers can write examples for its rules."""
    rules = []
    for rule in sheet.rules:
        fires_when, quiet_when = _hints(rule)
        question = rule.as_question()
        rules.append(
            RuleDraft(
                id=rule.id,
                category=rule.category,
                type=question.type,
                instructions=rule.question,
                criteria=question.criteria,
                fires_when=fires_when,
                does_not_fire_when=quiet_when,
            )
        )
    return ScenarioDraft(
        title=sheet.npc or sheet.id,
        context=sheet.context(),
        character_name=sheet.npc or None,
        tools=tuple(tool.name for tool in sheet.tools),
        rules=tuple(rules),
    )


def as_gold_sheet(sheet: Sheet) -> dict[str, Any]:
    """The sheet as the gold editor stores it, so gold conversations can be written against it."""
    sheet_fields = set(GoldSheet.model_fields) - {"tools", "rules"}
    gold = GoldSheet.model_validate(
        {
            **sheet.model_dump(include=sheet_fields),
            "tools": [tool.model_dump(include=set(GoldTool.model_fields)) for tool in sheet.tools],
            "rules": [rule.model_dump(include=set(GoldRule.model_fields)) for rule in sheet.rules],
        }
    )
    return gold.model_dump()
