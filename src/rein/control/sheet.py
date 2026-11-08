from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from rein.control import constants as c
from rein.control.base import base_problems
from rein.schema import Question
from rein.schema.constants import ID_PATTERN

Identifier = Annotated[str, StringConstraints(pattern=ID_PATTERN)]
Template = Annotated[str, StringConstraints(min_length=1, max_length=c.MAX_TEMPLATE_CHARS)]
Probability = Annotated[float, Field(gt=0.0, le=1.0)]
Term = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=c.MAX_FORBIDDEN_TERM_CHARS)]
BaseText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=c.MAX_BASE_CHARS)]


class ForceTool(BaseModel):
    """Opens the reply with a call to `tool`, or to the tool mapped from the winning option of a choice question."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["force_tool"] = "force_tool"
    tool: Identifier | None = None
    tools: dict[str, Identifier] | None = None

    @model_validator(mode="after")
    def _one_source(self) -> "ForceTool":
        if (self.tool is None) == (self.tools is None):
            raise ValueError("force_tool needs exactly one of tool or tools")
        return self

    def tool_for(self, option: str) -> str:
        if self.tool is not None:
            return self.tool
        assert self.tools is not None
        return self.tools[option]


class Prefill(BaseModel):
    """Opens the reply with operator text, such as a mandatory phrase, an identity statement or a format opener."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["prefill"] = "prefill"
    text: Template
    end_reply: bool = False


class Escalate(BaseModel):
    """Opens the reply with a handoff message and tells the application to route the session to a person."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["escalate"] = "escalate"
    text: Template


class Reinject(BaseModel):
    """Only reminds the generator of the rule for this reply; the generator still decides."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["reinject"] = "reinject"


Action = Annotated[ForceTool | Prefill | Escalate | Reinject, Field(discriminator="kind")]


class Gate(BaseModel):
    """At or above `fire` the action runs (tight rein); at or above `reinject` the rule is only reinjected (loose)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    fire: Probability = c.DEFAULT_FIRE_THRESHOLD
    reinject: Probability | None = None

    @model_validator(mode="after")
    def _ordered(self) -> "Gate":
        if self.reinject is not None and self.reinject >= self.fire:
            raise ValueError("the reinject threshold must sit below the fire threshold")
        return self


class Rule(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: Identifier
    scope: Literal["turn", "draft"] = "turn"
    question: Question
    fire_on: tuple[str, ...] = ("yes",)
    action: Action
    gate: Gate = Gate()
    reinject_text: Template | None = None
    priority: int = 0
    forbidden_terms: Annotated[tuple[Term, ...], Field(max_length=c.MAX_FORBIDDEN_TERMS)] = ()
    base: BaseText | None = None

    @model_validator(mode="after")
    def _consistent(self) -> "Rule":
        if self.question.label is not None or self.question.target is not None:
            raise ValueError("rule questions carry no labels")
        keys = set(self.question.option_keys())
        if not self.fire_on or not set(self.fire_on) <= keys:
            raise ValueError(f"fire_on must name options of the question: {sorted(keys)}")
        if self.forbidden_terms and self.scope != "turn":
            raise ValueError("forbidden_terms apply to turn rules, where the decision comes before the reply")
        tools = self.action.tools if isinstance(self.action, ForceTool) else None
        if tools is not None and set(tools) != set(self.fire_on):
            raise ValueError("force_tool.tools must map every fire_on option to a tool")
        if self.base is not None:
            action_text = self.action.text if isinstance(self.action, Prefill | Escalate) else None
            problems = base_problems(self.base, self.forbidden_terms, [action_text] if action_text else [])
            if problems:
                raise ValueError("; ".join(problems))
        return self

    @property
    def reminder(self) -> str:
        return self.reinject_text or self.question.instructions


class RuleSheet(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    context: Annotated[str | None, StringConstraints(max_length=c.MAX_TEMPLATE_CHARS)] = None
    rules: Annotated[tuple[Rule, ...], Field(min_length=1, max_length=c.MAX_RULES)]
    variables: Annotated[
        dict[Identifier, Annotated[str, StringConstraints(max_length=c.MAX_VARIABLE_CHARS)]],
        Field(max_length=c.MAX_VARIABLES),
    ] = Field(default_factory=dict)
    reinject_placement: Literal["system_tail", "user_suffix"] = "user_suffix"
    tool_suppress_below: Probability | None = c.DEFAULT_TOOL_SUPPRESS_BELOW
    watch_request_floor: Probability | None = c.DEFAULT_WATCH_REQUEST_FLOOR
    unrequested_action_reminder: Template = c.UNREQUESTED_ACTION_REMINDER

    @model_validator(mode="after")
    def _unique_ids(self) -> "RuleSheet":
        ids = [rule.id for rule in self.rules]
        if len(ids) != len(set(ids)):
            raise ValueError("rule ids must be unique")
        return self

    def scoped(self, scope: str) -> tuple[Rule, ...]:
        return tuple(rule for rule in self.rules if rule.scope == scope)
