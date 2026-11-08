import asyncio
import json
from collections import defaultdict
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from rein.datagen import generate
from rein.datagen.assemble import build_record
from rein.datagen.generate import ChatModel, Rejected, Stats
from rein.datagen.labels import needs_tiebreak, resolve
from rein.datagen.models import ConversationDraft, ScenarioDraft
from rein.datagen.plan import CaseSpec, ScenarioSpec, plan_cases, plan_scenarios
from rein.datagen.prompts import ANNOTATOR, PROMPT_VERSION, REVIEWER
from rein.datagen.taxonomy import Taxonomy
from rein.schema import Split
from rein.teacher.store import JsonlStore, content_key

M = TypeVar("M", bound=BaseModel)
STAGE_FILES = ("scenarios", "cases", "judgments", "records")


class SliceOverrides(BaseModel):
    """Narrows a run to one evaluation slice: languages, lengths, categories, and a forced split."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    conversation_languages: dict[str, float] | None = None
    length_bands: tuple[str, ...] | None = None
    required_categories: tuple[str, ...] = ()
    target_categories: tuple[str, ...] = ()
    force_split: Split | None = None


def apply_overrides(taxonomy: Taxonomy, overrides: SliceOverrides | None) -> Taxonomy:
    if overrides is None:
        return taxonomy
    data = taxonomy.model_dump()
    if overrides.conversation_languages is not None:
        data["languages"]["conversation"] = overrides.conversation_languages
    if overrides.length_bands is not None:
        data["conversation_lengths"] = {
            name: band for name, band in data["conversation_lengths"].items() if name in overrides.length_bands
        }
    return Taxonomy.model_validate(data)


class DatagenConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_name: str
    scenarios: int = Field(ge=1)
    cases_per_scenario: int = Field(ge=1)
    seed: int = 0
    max_in_flight: int = Field(default=32, ge=1)
    scenario_workers: int = Field(default=16, ge=1)
    output_dir: Path = Path("data/synthetic")
    overrides: SliceOverrides | None = None


class Stage:
    """A resumable stage: every outcome (value or rejection) is stored once under a content key."""

    def __init__(self, path: Path) -> None:
        self._index: dict[str, dict[str, Any]] = {}
        self._store = JsonlStore(path)
        for row in self._store.rows():
            self._index[row["key"]] = row

    async def run(self, key: str, schema: type[M], produce: Callable[[], Awaitable[M]]) -> M | None:
        if key in self._index:
            row = self._index[key]
            return schema.model_validate(row["value"]) if row["ok"] else None
        try:
            value = await produce()
        except Rejected as exc:
            row = {"ok": False, "reason": str(exc), "value": None}
        else:
            row = {"ok": True, "reason": None, "value": value.model_dump(mode="json")}
        self._store.append(key, row)
        self._index[key] = {"key": key, **row}
        return value if row["ok"] else None


class _Judgments(BaseModel):
    answers: dict[str, str]


class Pipeline:
    def __init__(
        self, config: DatagenConfig, model: ChatModel, taxonomy: Taxonomy, writer: ChatModel | None = None
    ) -> None:
        """`model` judges every case; `writer` writes the cases, `model` too when not given."""
        self.config = config
        self.model = model
        self.writer = writer or model
        self.taxonomy = apply_overrides(taxonomy, config.overrides)
        self.overrides = config.overrides or SliceOverrides()
        self.stats = Stats()
        self.out_dir = config.output_dir / config.run_name
        self.stages = {name: Stage(self.out_dir / f"{name}.jsonl") for name in STAGE_FILES[:3]}
        self.records = JsonlStore(self.out_dir / "records.jsonl")
        self._workers = asyncio.Semaphore(config.scenario_workers)

    def _key(self, *parts: Any) -> str:
        return content_key(PROMPT_VERSION, self.config.seed, *parts)

    async def _judge(
        self,
        scenario: ScenarioDraft,
        case: CaseSpec,
        conversation: ConversationDraft,
        framing: str,
        thinking: bool = False,
    ) -> dict[str, str] | None:
        async def produce() -> _Judgments:
            answers = await generate.judge_case(self.model, scenario, conversation, framing, self.stats, thinking)
            return _Judgments(answers=answers)

        result = await self.stages["judgments"].run(
            self._key("judge", case.case_id, framing, thinking), _Judgments, produce
        )
        return None if result is None else result.answers

    async def _case(self, scenario_spec: ScenarioSpec, scenario: ScenarioDraft, case: CaseSpec) -> None:
        if case.case_id in self.records:
            return
        conversation = await self.stages["cases"].run(
            self._key("case", asdict(case)),
            ConversationDraft,
            lambda: generate.generate_case(self.writer, self.taxonomy, scenario, case, self.stats),
        )
        if conversation is None:
            return
        first, second = await asyncio.gather(
            self._judge(scenario, case, conversation, ANNOTATOR), self._judge(scenario, case, conversation, REVIEWER)
        )
        if first is None or second is None:
            return
        votes = {rule.id: [first[rule.id], second[rule.id]] for rule in scenario.rules}
        votes[case.target_rule].append(case.target_option)
        if case.second_rule is not None and case.second_option is not None:
            votes[case.second_rule].append(case.second_option)
        if any(needs_tiebreak(rule_votes) for rule_votes in votes.values()):
            third = await self._judge(scenario, case, conversation, ANNOTATOR, thinking=True)
            if third is None:
                self.stats.accepted["tiebreak_missing_soft"] += 1
            else:
                for rule_id, rule_votes in votes.items():
                    rule_votes.append(third[rule_id])
        resolved = {rule.id: resolve(rule, votes[rule.id]) for rule in scenario.rules}
        record = build_record(scenario_spec, scenario, case, conversation, resolved, self.overrides.force_split)
        self.records.append(record.id, record.model_dump(mode="json"))
        self.stats.accepted["record"] += 1

    async def extend(self, spec: ScenarioSpec, scenario: ScenarioDraft, cases: Sequence[CaseSpec]) -> Stats:
        """More cases for a scenario that already exists, generated and judged like any other case."""
        async with self._workers:
            await asyncio.gather(*(self._case(spec, scenario, case) for case in cases))
        return self.stats

    async def _scenario(self, spec: ScenarioSpec) -> None:
        async with self._workers:
            scenario = await self.stages["scenarios"].run(
                self._key("scenario", asdict(spec)),
                ScenarioDraft,
                lambda: generate.generate_scenario(self.model, self.taxonomy, spec, self.stats),
            )
            if scenario is None:
                return
            cases = plan_cases(
                self.taxonomy,
                spec.scenario_id,
                scenario,
                self.config.cases_per_scenario,
                self.config.seed,
                self.overrides.target_categories,
            )
            await asyncio.gather(*(self._case(spec, scenario, case) for case in cases))

    async def run(self) -> Stats:
        specs = plan_scenarios(
            self.taxonomy,
            self.config.scenarios,
            self.config.seed,
            self.config.run_name,
            self.overrides.required_categories,
        )
        await asyncio.gather(*(self._scenario(spec) for spec in specs))
        self.export()
        return self.stats

    def export(self) -> dict[str, int]:
        by_split: dict[str, list[str]] = defaultdict(list)
        for row in self.records.rows():
            record = {key: value for key, value in row.items() if key != "key"}
            by_split[record["split"]].append(json.dumps(record, ensure_ascii=False))
        counts = {}
        for split, lines in by_split.items():
            (self.out_dir / f"{split}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
            counts[split] = len(lines)
        report = {
            "splits": counts,
            "accepted": dict(self.stats.accepted),
            "rejected": dict(self.stats.rejected),
            "completion_tokens": self.stats.completion_tokens,
        }
        (self.out_dir / "stats.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return counts
