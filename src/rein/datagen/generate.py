import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from rein.datagen import prompts
from rein.datagen.language import matches_language
from rein.datagen.models import ConversationDraft, JudgeVerdict, ScenarioDraft
from rein.datagen.plan import CaseSpec, ScenarioSpec
from rein.datagen.prompts import SCORE_LEVELS
from rein.datagen.taxonomy import Taxonomy
from rein.teacher.client import Completion, TeacherError, parse_json_object

SCENARIO_MAX_TOKENS = 2048
CASE_MAX_TOKENS = 3072
JUDGE_MAX_TOKENS = 512
TIEBREAK_MAX_TOKENS = 12288
THINKING_TEMPERATURE = 0.6
THINKING_TOP_P = 0.95
GENERATION_TEMPERATURE = 0.8
REPAIR_TEMPERATURE = 0.4
JUDGE_TEMPERATURE = 0.0
JUDGE_TOP_P = 1.0
ATTEMPTS = 2
TURN_TOLERANCE = 0.5
TRUNCATE_MIN_FRACTION = 0.3
TRUNCATE_MAX_FRACTION = 0.85
DRAFTS_WITH_CALLS = frozenset({"tool_call", "narration_after_call"})


class ChatModel(Protocol):
    async def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float = ...,
        top_p: float = ...,
        max_tokens: int,
        json_mode: bool = ...,
        seed: int | None = ...,
        thinking: bool = ...,
        extra: Mapping[str, Any] | None = ...,
    ) -> Completion: ...


@dataclass
class Stats:
    accepted: Counter[str] = field(default_factory=Counter)
    rejected: Counter[str] = field(default_factory=Counter)
    completion_tokens: int = 0

    def reject(self, stage: str, reason: str) -> None:
        self.rejected[f"{stage}:{reason}"] += 1


class Rejected(ValueError):
    pass


async def ask_json[T: BaseModel](
    model: ChatModel,
    prompt: str,
    schema: type[T],
    stats: Stats,
    stage: str,
    *,
    max_tokens: int,
    temperature: float,
    top_p: float | None = None,
    thinking: bool = False,
) -> T | None:
    for attempt in range(ATTEMPTS):
        kwargs: dict[str, Any] = {
            "max_tokens": max_tokens,
            "temperature": temperature if attempt == 0 else min(temperature, REPAIR_TEMPERATURE),
            "thinking": thinking,
        }
        if top_p is not None:
            kwargs["top_p"] = top_p
        try:
            completion = await model.chat([{"role": "user", "content": prompt}], **kwargs)
            stats.completion_tokens += completion.completion_tokens
            return schema.model_validate(parse_json_object(completion.content))
        except TeacherError as exc:
            stats.reject(stage, "teacher" if "giving up" in str(exc) else "json")
        except ValidationError as exc:
            first = exc.errors()[0]
            location = ".".join(str(part) for part in first["loc"] if not isinstance(part, int)) or "root"
            stats.reject(stage, f"schema:{location}:{first['type']}")
    return None


async def generate_scenario(model: ChatModel, taxonomy: Taxonomy, spec: ScenarioSpec, stats: Stats) -> ScenarioDraft:
    scenario = await ask_json(
        model,
        prompts.scenario_prompt(taxonomy, spec),
        ScenarioDraft,
        stats,
        "scenario",
        max_tokens=SCENARIO_MAX_TOKENS,
        temperature=GENERATION_TEMPERATURE,
    )
    if scenario is None:
        raise Rejected("scenario output invalid")
    categories = [rule.category for rule in scenario.rules]
    if sorted(categories) != sorted(spec.categories):
        stats.reject("scenario", "categories")
        raise Rejected("scenario rules do not match the requested categories")
    for rule in scenario.rules:
        if rule.type not in taxonomy.categories[rule.category].types:
            stats.reject("scenario", "rule_type")
            raise Rejected(f"rule {rule.id} has a type its category does not allow")
        if rule.type == "score" and len(rule.question().option_keys()) != SCORE_LEVELS:
            stats.reject("scenario", "score_levels")
            raise Rejected(f"score rule {rule.id} needs {SCORE_LEVELS} anchored levels")
    stats.accepted["scenario"] += 1
    return scenario


def check_case(conversation: ConversationDraft, spec: CaseSpec) -> str | None:
    low = max(1, int(spec.turns * (1 - TURN_TOLERANCE)))
    high = int(spec.turns * (1 + TURN_TOLERANCE)) + 1
    if not low <= len(conversation.conversation) <= high:
        return "turns"
    if (spec.draft_kind is None) != (conversation.draft is None):
        return "draft_presence"
    if conversation.draft is not None:
        has_calls = bool(conversation.draft.tool_calls)
        if (spec.draft_kind in DRAFTS_WITH_CALLS) != has_calls:
            return "draft_tool_calls"
    if not matches_language(conversation.conversation, spec.conversation_language):
        return f"language_{spec.conversation_language}"
    return None


def truncate_draft(conversation: ConversationDraft, seed: str) -> ConversationDraft:
    """Cuts a call-free draft mid-sentence, the way the watcher receives streamed chunks."""
    draft = conversation.draft
    if draft is None or draft.tool_calls:
        return conversation
    rng = random.Random(seed)
    cut = max(1, int(len(draft.text) * rng.uniform(TRUNCATE_MIN_FRACTION, TRUNCATE_MAX_FRACTION)))
    return conversation.model_copy(update={"draft": draft.model_copy(update={"text": draft.text[:cut]})})


async def generate_case(
    model: ChatModel, taxonomy: Taxonomy, scenario: ScenarioDraft, spec: CaseSpec, stats: Stats
) -> ConversationDraft:
    conversation = await ask_json(
        model,
        prompts.case_prompt(taxonomy, scenario, spec),
        ConversationDraft,
        stats,
        "case",
        max_tokens=CASE_MAX_TOKENS,
        temperature=GENERATION_TEMPERATURE,
    )
    if conversation is None:
        raise Rejected("case output invalid")
    problem = check_case(conversation, spec)
    if problem:
        stats.reject("case", problem)
        raise Rejected(f"case failed check: {problem}")
    if spec.truncate_draft:
        conversation = truncate_draft(conversation, spec.case_id)
    stats.accepted["case"] += 1
    return conversation


async def judge_case(
    model: ChatModel,
    scenario: ScenarioDraft,
    conversation: ConversationDraft,
    framing: str,
    stats: Stats,
    thinking: bool = False,
) -> dict[str, str]:
    stage = f"judge_{framing}{'_thinking' if thinking else ''}"
    verdict = await ask_json(
        model,
        prompts.judge_prompt(scenario, conversation.conversation, conversation.draft, framing),
        JudgeVerdict,
        stats,
        stage,
        max_tokens=TIEBREAK_MAX_TOKENS if thinking else JUDGE_MAX_TOKENS,
        temperature=THINKING_TEMPERATURE if thinking else JUDGE_TEMPERATURE,
        top_p=THINKING_TOP_P if thinking else JUDGE_TOP_P,
        thinking=thinking,
    )
    if verdict is None:
        raise Rejected(f"{stage} output invalid")
    for rule in scenario.rules:
        answer = verdict.answers.get(rule.id)
        if answer not in rule.question().option_keys():
            stats.reject(stage, "answer_keys")
            raise Rejected(f"{stage} gave no valid answer for {rule.id}")
    stats.accepted[stage] += 1
    return {rule.id: verdict.answers[rule.id] for rule in scenario.rules}
