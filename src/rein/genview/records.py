from collections.abc import Mapping

from rein.control.prompt import compose_system
from rein.genview import constants as c
from rein.genview.keys import rule_key
from rein.genview.render import PromptSpec
from rein.genview.scenarios import ScenarioInfo
from rein.genview.tools import tool_specs
from rein.schema import Question, State, TrainingRecord


class MissingBase(KeyError):
    pass


def rule_line(question: Question) -> str:
    """A whole rule as text: the question and its options. The rules ablation writes it into the system prompt; the
    block door (DR-026) passes it through the chat template's `rules`."""
    return f"{question.instructions} Options: {'; '.join(question.option_texts())}."


def record_spec(
    record: TrainingRecord, bases: Mapping[str, str], scenario: ScenarioInfo | None, prompt: str = c.PROMPT_BASES
) -> PromptSpec:
    """The generator's prompt for a record: its context and the bases of its rules, never the rules themselves.

    The ablations drop the bases (`PROMPT_NONE`) or write each whole rule instead (`PROMPT_RULES`); the block door
    (`PROMPT_BLOCK`) passes each whole rule, typed by its category, to the template's `<rules>` block.
    """
    if prompt not in c.PROMPTS:
        raise ValueError(f"unknown prompt variant {prompt!r}")
    missing = [qid for qid, question in record.questions.items() if rule_key(question) not in bases]
    if prompt == c.PROMPT_BASES and missing:
        raise MissingBase(f"record {record.id} has rules without a base: {missing}")
    lines = {
        c.PROMPT_BASES: lambda: [bases[rule_key(question)] for question in record.questions.values()],
        c.PROMPT_NONE: lambda: [],
        c.PROMPT_RULES: lambda: [rule_line(question) for question in record.questions.values()],
        c.PROMPT_BLOCK: lambda: [],
    }[prompt]()
    system = compose_system(record.state.context or c.GENERIC_SYSTEM, lines)
    tools = tool_specs(scenario.tools, scenario.tool_descriptions) if scenario else ()
    rules = block_rules(record, scenario) if prompt == c.PROMPT_BLOCK else ()
    return PromptSpec(system=system, state=State(conversation=record.state.conversation), tools=tools, rules=rules)


def block_rules(record: TrainingRecord, scenario: ScenarioInfo | None) -> tuple[dict[str, str], ...]:
    """Each rule of the record for the `<rules>` block: its category as the type, when the scenario declared one."""
    categories = scenario.categories if scenario else {}
    return tuple(
        {"type": categories.get(question_id, ""), "text": rule_line(question)}
        for question_id, question in record.questions.items()
    )
