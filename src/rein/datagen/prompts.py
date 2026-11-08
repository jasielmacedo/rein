import json
from functools import cache
from pathlib import Path
from string import Template

from rein.datagen.models import RuleDraft, ScenarioDraft
from rein.datagen.plan import CaseSpec, ScenarioSpec
from rein.datagen.taxonomy import Taxonomy
from rein.schema import DecisionRequest, Draft, Message, Question

PROMPT_DIR = Path(__file__).with_name("prompts")
PROMPT_VERSION = "v2.2"
SCORE_LEVELS = 5

ANNOTATOR = "annotator"
REVIEWER = "reviewer"
AUDITOR = "auditor"
FRAMINGS = {
    ANNOTATOR: "You are a careful annotator. Read the whole conversation, then decide each rule independently for "
    "the latest user message.",
    REVIEWER: "You are a skeptical reviewer checking labels. For each rule, find the exact evidence in the latest "
    "user message (and the reply in progress, when there is one) and answer the option that evidence supports. "
    "The same words used with a different intent do not count, and a trigger that was already handled earlier "
    "does not count for the latest turn.",
    AUDITOR: "You are an independent annotator. For each rule, find the exact evidence in the latest user message "
    "(and the reply in progress, when there is one) and answer the option that evidence supports. The same words "
    "used with a different intent do not count, and a trigger that was already handled earlier does not count for "
    "the latest turn.",
}

DRAFT_INSTRUCTIONS = {
    None: ('Set "draft" to null.', "null"),
    "narrated_action": (
        'Also write "draft": the assistant\'s reply in progress to the latest message. It narrates performing an '
        "action that needs one of the tools ($tools) as if it already happened, but it contains no tool call: "
        '"tool_calls" must be an empty list.',
        '{"text": str, "tool_calls": []}',
    ),
    "tool_call": (
        'Also write "draft": the assistant\'s reply in progress that performs the needed action correctly through a '
        'tool call in "tool_calls" (a name from: $tools, with arguments) plus at most one short sentence of text.',
        '{"text": str, "tool_calls": [{"name": str, "arguments": object}]}',
    ),
    "narration_after_call": (
        'Also write "draft": the assistant\'s reply in progress that has ALREADY emitted the right tool call in '
        '"tool_calls" (a name from: $tools, with arguments) and whose text confirms or narrates the finished action.',
        '{"text": str, "tool_calls": [{"name": str, "arguments": object}]}',
    ),
    "refusal": (
        'Also write "draft": the assistant\'s reply in progress that declines or postpones the requested action in '
        "character, without performing it, narrating it or calling any tool.",
        '{"text": str, "tool_calls": []}',
    ),
    "dialogue": (
        'Also write "draft": the assistant\'s reply in progress, ordinary dialogue with no action and no tool calls.',
        '{"text": str, "tool_calls": []}',
    ),
}


@cache
def _template(name: str) -> Template:
    return Template((PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8"))


def options_text(rule: RuleDraft) -> str:
    return question_options_text(rule.question())


def question_options_text(question: Question) -> str:
    return "; ".join(
        f'"{key}" ({text})' for key, text in zip(question.option_keys(), question.option_texts(), strict=True)
    )


def scenario_prompt(taxonomy: Taxonomy, spec: ScenarioSpec) -> str:
    lines = []
    for name in spec.categories:
        category = taxonomy.categories[name]
        lines.append(f"- {name} (type: {' or '.join(category.types)}): {category.description}")
    return _template("scenario").substitute(
        domain=spec.domain,
        rule_language=taxonomy.languages.names[spec.rule_language],
        rule_style=spec.rule_style,
        categories="\n".join(lines),
    )


def case_prompt(taxonomy: Taxonomy, scenario: ScenarioDraft, spec: CaseSpec) -> str:
    rule = next(rule for rule in scenario.rules if rule.id == spec.target_rule)
    form = taxonomy.case_forms[spec.case_form]
    tools = ", ".join(scenario.tools) or "the tools the context implies"
    instruction, schema = DRAFT_INSTRUCTIONS[spec.draft_kind]
    return _template("case").substitute(
        context=scenario.context,
        rule_id=rule.id,
        rule_type=rule.type,
        rule_instructions=rule.instructions,
        options=options_text(rule),
        fires_when=rule.fires_when,
        does_not_fire_when=rule.does_not_fire_when,
        turns=spec.turns,
        language=taxonomy.languages.names[spec.conversation_language],
        target_option=spec.target_option,
        case_form=spec.case_form.replace("_", " "),
        case_guidance=form.guidance,
        character_name=scenario.character_name or "the assistant",
        draft_instruction=Template(instruction).safe_substitute(tools=tools),
        second_rule_block=_second_rule_block(scenario, spec),
        draft_schema=schema,
    )


def _second_rule_block(scenario: ScenarioDraft, spec: CaseSpec) -> str:
    if spec.second_rule is None:
        return ""
    rule = next(rule for rule in scenario.rules if rule.id == spec.second_rule)
    return (
        f'A second rule must ALSO apply to the latest message (id "{rule.id}"): {rule.instructions} '
        f'Options: {options_text(rule)}. Its correct answer must be "{spec.second_option}".\n'
    )


def render_conversation(messages: tuple[Message, ...]) -> str:
    lines = []
    for message in messages:
        speaker = message.name or message.role
        lines.append(f"[{message.role}] {speaker}: {message.text}")
    return "\n".join(lines)


def render_draft(draft: Draft | None) -> str:
    if draft is None:
        return ""
    calls = json.dumps([call.model_dump() for call in draft.tool_calls], ensure_ascii=False)
    return f"\nReply in progress (not sent yet):\n{draft.text}\nTool calls emitted so far: {calls}\n"


def judge_prompt(scenario: ScenarioDraft, messages: tuple[Message, ...], draft: Draft | None, framing: str) -> str:
    rules = "\n".join(f'- "{rule.id}": {rule.instructions} Options: {options_text(rule)}' for rule in scenario.rules)
    return _template("judge").substitute(
        framing=FRAMINGS[framing],
        context=scenario.context,
        conversation=render_conversation(messages),
        draft_block=render_draft(draft),
        draft_scope=" (and the reply in progress, for rules about the reply)" if draft else "",
        rules=rules,
    )


def audit_prompt(request: DecisionRequest, framing: str = AUDITOR) -> str:
    """The judge prompt rebuilt from a finished record, so an independent model can re-answer its questions."""
    rules = "\n".join(
        f'- "{qid}": {question.instructions} Options: {question_options_text(question)}'
        for qid, question in request.questions.items()
    )
    draft = request.state.draft
    return _template("judge").substitute(
        framing=FRAMINGS[framing],
        context=request.state.context or "(none)",
        conversation=render_conversation(request.state.conversation),
        draft_block=render_draft(draft),
        draft_scope=" (and the reply in progress, for rules about the reply)" if draft else "",
        rules=rules,
    )
