"""The hand-written gold set from `tools/gold`: NPC sheets and chat-like conversations, turned into decision records.

Each player or narration message becomes one record (the conversation up to it) over the sheet's rules that look at
the player. Each NPC message becomes one record over the rules that look at the NPC's reply, with that reply as the
draft in progress. Answers left unmarked take the rule's null option, so a turn where nothing fires needs no clicks.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from rein.compiler.constants import NULL_OPTIONS
from rein.genview.keys import rule_key
from rein.genview.scenarios import ScenarioIndex, ScenarioInfo, scenario_key
from rein.schema import Message, Question, State, TrainingRecord
from rein.schema import constants as sc
from rein.schema.conversation import Draft, Role, ToolCall

FORMAT = "rein-gold-project"
VERSION = 1
SOURCE = "gold"
SPLIT: Final = "gold"
ROLES: dict[str, Role] = {"player": "user", "npc": "assistant", "narration": "narrator"}
PLAYER_SCOPE: Final = "player"
NPC_SCOPE: Final = "npc_reply"
AUTO_FORM, FIRING_FORM, QUIET_FORM = "auto", "direct", "neutral"
RECORD_ID = "gold-{conversation}-{turn:03d}"

Identifier = Annotated[str, StringConstraints(pattern=sc.ID_PATTERN)]
OptionText = Annotated[str, StringConstraints(max_length=sc.MAX_OPTION_TEXT_CHARS)]


class GoldTool(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Identifier
    description: OptionText = ""


class GoldOption(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: Annotated[str, StringConstraints(min_length=1, max_length=sc.MAX_OPTION_KEY_CHARS)]
    when: OptionText = ""


class GoldRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: Identifier
    question: Annotated[str, StringConstraints(min_length=1, max_length=sc.MAX_INSTRUCTION_CHARS)]
    kind: Literal["yes_no", "choice", "scale"]
    yes: OptionText = ""
    no: OptionText = ""
    options: list[GoldOption] = Field(default_factory=list)
    levels: list[OptionText] = Field(default_factory=list)
    looks_at: Literal["player", "npc_reply"] = PLAYER_SCOPE
    category: str = "other"
    base: str = ""

    @model_validator(mode="after")
    def _valid_question(self) -> "GoldRule":
        self.as_question()
        return self

    def option_keys(self) -> tuple[str, ...]:
        return self.as_question().option_keys()

    def as_question(self, answer: str | None = None) -> Question:
        """The rule as a typed question, labelled with `answer` (an option key) when given."""
        if self.kind == "yes_no":
            criteria: dict[str, str | None] = {
                flag: text for flag, text in (("true", self.yes), ("false", self.no)) if text.strip()
            }
            label = None if answer is None else answer == "yes"
            return Question(type="noul", instructions=self.question, criteria=criteria or None, label=label)
        if self.kind == "choice":
            options = {option.key: option.when or None for option in self.options}
            return Question(type="choice", instructions=self.question, criteria=options, label=answer)
        level = None if answer is None else int(answer)
        return Question(type="score", instructions=self.question, criteria=list(self.levels), label=level)

    def default_answer(self) -> str | None:
        """What an unmarked message means: the option where the rule stays off, when the rule has one."""
        return next((key for key in self.option_keys() if key in NULL_OPTIONS), None)

    def fires(self, answer: str) -> bool:
        return self.kind != "scale" and answer != self.default_answer()


class GoldSheet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: Annotated[str, StringConstraints(min_length=1, max_length=sc.MAX_SCENARIO_ID_CHARS)]
    npc: str = ""
    persona: str = ""
    scene: str = ""
    language: str = "en"
    tools: Sequence[GoldTool] = Field(default_factory=list)
    rules: Sequence[GoldRule] = Field(default_factory=list)

    def context(self) -> str:
        return "\n\n".join(part.strip() for part in (self.persona, self.scene) if part.strip())

    def rules_for(self, speaker: str) -> list[GoldRule]:
        scope = NPC_SCOPE if speaker == "npc" else PLAYER_SCOPE
        return [rule for rule in self.rules if rule.looks_at == scope]


class GoldMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    speaker: Literal["player", "npc", "narration"] = Field(alias="from")
    name: str = ""
    text: Annotated[str, Field(max_length=sc.MAX_MESSAGE_CHARS)]
    tool_calls: list[Identifier] = Field(default_factory=list)
    answers: dict[str, str] = Field(default_factory=dict)
    tests: str = ""
    form: str = AUTO_FORM
    note: str = ""
    skip: bool = False


class GoldConversation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: Identifier
    sheet: str
    title: str = ""
    language: str = "en"
    messages: list[GoldMessage] = Field(default_factory=list)


class GoldProject(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format: Literal["rein-gold-project"]
    version: Literal[1]
    sheets: list[GoldSheet] = Field(default_factory=list)
    conversations: list[GoldConversation] = Field(default_factory=list)


@dataclass
class GoldBuild:
    records: list[TrainingRecord] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def _message(sheet: GoldSheet, message: GoldMessage) -> Message:
    name = (sheet.npc.strip() if message.speaker == "npc" else message.name.strip()) or None
    calls = tuple(ToolCall(name=call) for call in message.tool_calls)
    return Message(role=ROLES[message.speaker], text=message.text, name=name, tool_calls=calls)


def _answer(rule: GoldRule, message: GoldMessage) -> str | None:
    marked = message.answers.get(rule.id)
    return marked if marked in rule.option_keys() else rule.default_answer()


def decision_record(
    sheet: GoldSheet, conversation: GoldConversation, turn: int, problems: list[str]
) -> TrainingRecord | None:
    """The record for message `turn` of a conversation, or None when it is not a decision or misses an answer."""
    message = conversation.messages[turn]
    rules = sheet.rules_for(message.speaker)
    if message.skip or not rules or not message.text.strip():
        return None
    answers = {rule.id: _answer(rule, message) for rule in rules}
    missing = [rule_id for rule_id, answer in answers.items() if answer is None]
    if missing:
        problems.append(f"{conversation.id}, message {turn + 1}: no answer for {', '.join(missing)}")
        return None
    firing = [rule.id for rule in rules if rule.fires(str(answers[rule.id]))]
    target = message.tests if message.tests in answers else (firing[0] if firing else None)
    by_id = {rule.id: rule for rule in rules}
    is_reply = message.speaker == "npc"
    history = [_message(sheet, earlier) for earlier in conversation.messages[:turn]]
    if not is_reply:
        history.append(_message(sheet, message))
    draft = Draft(text=message.text, tool_calls=_message(sheet, message).tool_calls) if is_reply else None
    form = message.form if message.form != AUTO_FORM else (FIRING_FORM if firing else QUIET_FORM)
    return TrainingRecord(
        id=RECORD_ID.format(conversation=conversation.id, turn=turn + 1),
        source=SOURCE,
        scenario_id=sheet.id,
        split=SPLIT,
        state=State(context=sheet.context() or None, conversation=tuple(history), draft=draft),
        questions={rule.id: rule.as_question(answers[rule.id]) for rule in rules},
        meta={
            "category": by_id[target].category if target else None,
            "case_form": form,
            "target_rule": target,
            "firing_rules": firing,
            "lang_rule": sheet.language,
            "lang_conv": conversation.language,
            "decision_turn": turn + 1,
            "unseen_rule": True,
            "author": "human",
            "conversation": conversation.id,
            "notes": message.note.strip() or None,
        },
    )


def build_records(project: GoldProject) -> GoldBuild:
    built = GoldBuild()
    sheets = {sheet.id: sheet for sheet in project.sheets}
    for conversation in project.conversations:
        sheet = sheets.get(conversation.sheet)
        if sheet is None:
            built.problems.append(f"{conversation.id}: sheet {conversation.sheet!r} not found")
            continue
        for turn in range(len(conversation.messages)):
            record = decision_record(sheet, conversation, turn, built.problems)
            if record is not None:
                built.records.append(record)
    return built


def gold_index(project: GoldProject) -> ScenarioIndex:
    """Each sheet's tools and rule categories, found by context and rule ids like a synthetic scenario."""
    entries = {}
    for sheet in project.sheets:
        info = ScenarioInfo(
            context=sheet.context(),
            tools=tuple(tool.name for tool in sheet.tools),
            categories={rule.id: rule.category for rule in sheet.rules},
            tool_descriptions={tool.name: tool.description for tool in sheet.tools if tool.description},
        )
        scopes = [[rule.id for rule in sheet.rules if rule.looks_at == scope] for scope in (PLAYER_SCOPE, NPC_SCOPE)]
        for rule_ids in [*scopes, [rule.id for rule in sheet.rules]]:
            if rule_ids:
                entries[scenario_key(sheet.context() or None, rule_ids)] = info
    return ScenarioIndex(entries)


def sheet_bases(project: GoldProject) -> dict[str, str]:
    """The bases the author wrote, by rule key; rules left without one get a teacher-written base."""
    return {
        rule_key(rule.as_question()): rule.base.strip()
        for sheet in project.sheets
        for rule in sheet.rules
        if rule.base.strip()
    }
