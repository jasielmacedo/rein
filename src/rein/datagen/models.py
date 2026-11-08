import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator

from rein.schema import Draft, Message, Question, QuestionType
from rein.schema import constants as sc

RuleId = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{1,47}$")]
MIN_RULES = 2
MIN_INSTRUCTION_CHARS = 8
MIN_HINT_CHARS = 4
MIN_TITLE_CHARS = 3
MAX_TITLE_CHARS = 120
MIN_CONTEXT_CHARS = 40
MAX_RULES = 8
MIN_TURNS = 1
MAX_TURNS = 120
DraftKind = Literal["narrated_action", "tool_call", "dialogue"]
MAX_TOOL_NAME_CHARS = 48
_NON_IDENTIFIER = re.compile(r"[^a-z0-9]+")
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _snake_case(name: str) -> str:
    snake = _NON_IDENTIFIER.sub("_", _CAMEL_BOUNDARY.sub("_", name).lower()).strip("_")
    if not snake or not snake[0].isalpha():
        snake = f"tool_{snake}"
    return snake[:MAX_TOOL_NAME_CHARS]


class RuleDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: RuleId
    category: str
    type: QuestionType
    instructions: Annotated[
        str, StringConstraints(min_length=MIN_INSTRUCTION_CHARS, max_length=sc.MAX_INSTRUCTION_CHARS)
    ]
    criteria: dict[str, str | None] | list[str] | None = None
    fires_when: Annotated[str, StringConstraints(min_length=MIN_HINT_CHARS, max_length=sc.MAX_OPTION_TEXT_CHARS)]
    does_not_fire_when: Annotated[
        str, StringConstraints(min_length=MIN_HINT_CHARS, max_length=sc.MAX_OPTION_TEXT_CHARS)
    ]

    def question(self) -> Question:
        return Question(type=self.type, instructions=self.instructions, criteria=self.criteria)

    @model_validator(mode="after")
    def _valid_question(self) -> "RuleDraft":
        self.question()
        return self


class ScenarioDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    title: Annotated[str, StringConstraints(min_length=MIN_TITLE_CHARS, max_length=MAX_TITLE_CHARS)]
    context: Annotated[str, StringConstraints(min_length=MIN_CONTEXT_CHARS, max_length=sc.MAX_CONTEXT_CHARS)]
    character_name: Annotated[str | None, Field(max_length=sc.MAX_NAME_CHARS)] = None
    tools: tuple[Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{1,47}$")], ...] = ()
    rules: Annotated[tuple[RuleDraft, ...], Field(min_length=MIN_RULES, max_length=MAX_RULES)]

    @field_validator("tools", mode="before")
    @classmethod
    def _snake_case_tools(cls, tools: Any) -> Any:
        """Teachers sometimes write `sendPhoto` or `send-photo`; the tool is the same, so keep the scenario."""
        if not isinstance(tools, list | tuple):
            return tools
        return tuple(_snake_case(str(tool)) for tool in tools)

    @field_validator("rules")
    @classmethod
    def _unique_ids(cls, rules: tuple[RuleDraft, ...]) -> tuple[RuleDraft, ...]:
        ids = [rule.id for rule in rules]
        if len(ids) != len(set(ids)):
            raise ValueError("rule ids must be unique")
        return rules


class ConversationDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    conversation: Annotated[tuple[Message, ...], Field(min_length=MIN_TURNS, max_length=MAX_TURNS)]
    draft: Draft | None = None

    @model_validator(mode="before")
    @classmethod
    def _cut_after_latest_user(cls, data: Any) -> Any:
        """Drops turns the teacher wrote after the decisive user message; judges label what remains."""
        if not isinstance(data, dict) or not isinstance(data.get("conversation"), list):
            return data
        messages = data["conversation"]
        last_user = max(
            (i for i, message in enumerate(messages) if isinstance(message, dict) and message.get("role") == "user"),
            default=None,
        )
        if last_user is None:
            return data
        return {**data, "conversation": messages[: last_user + 1]}

    @model_validator(mode="after")
    def _ends_on_user(self) -> "ConversationDraft":
        if self.conversation[-1].role != "user":
            raise ValueError("the decisive (latest) message must come from the user")
        return self


class JudgeVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    answers: dict[str, str]
