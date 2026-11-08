import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rein.genview import constants as c
from rein.schema import TrainingRecord

ScenarioKey = tuple[str, frozenset[str]]


@dataclass(frozen=True)
class ScenarioInfo:
    """What the generator needs from a synthetic scenario: its tools and the category of each rule."""

    context: str
    tools: tuple[str, ...] = ()
    categories: Mapping[str, str] = field(default_factory=dict)
    tool_descriptions: Mapping[str, str] = field(default_factory=dict)


def _info(value: Mapping[str, Any]) -> ScenarioInfo:
    rules = value.get("rules") or []
    descriptions: dict[str, str] = {}
    for rule in rules:
        criteria = rule.get("criteria")
        if rule.get("category") == c.TOOL_CATEGORY and isinstance(criteria, dict):
            for key, text in criteria.items():
                if key not in c.NULL_OPTIONS and text:
                    descriptions.setdefault(key, text)
    return ScenarioInfo(
        context=value["context"],
        tools=tuple(value.get("tools") or ()),
        categories={rule["id"]: rule["category"] for rule in rules},
        tool_descriptions=descriptions,
    )


def scenario_key(context: str | None, rule_ids: Iterable[str]) -> ScenarioKey:
    return (context or "", frozenset(rule_ids))


class ScenarioIndex:
    """Finds a record's scenario by its context and rule ids, which identify it uniquely in the generated data."""

    def __init__(self, entries: Mapping[ScenarioKey, ScenarioInfo]) -> None:
        self._entries = dict(entries)

    @classmethod
    def load(cls, files: Iterable[Path]) -> "ScenarioIndex":
        entries: dict[ScenarioKey, ScenarioInfo] = {}
        for path in files:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    value = row.get("value")
                    if not row.get("ok") or not isinstance(value, dict):
                        continue
                    key = scenario_key(value["context"], (rule["id"] for rule in value.get("rules") or []))
                    entries.setdefault(key, _info(value))
        return cls(entries)

    @classmethod
    def from_directory(cls, synthetic_dir: Path) -> "ScenarioIndex":
        return cls.load(sorted(synthetic_dir.glob(c.SCENARIO_GLOB)))

    def lookup(self, record: TrainingRecord) -> ScenarioInfo | None:
        return self._entries.get(scenario_key(record.state.context, record.questions))

    def __len__(self) -> int:
        return len(self._entries)
