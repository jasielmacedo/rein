from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from rein.schema import constants as c

Role = Literal["user", "assistant", "narrator", "tool", "system"]


class ToolCall(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: Annotated[str, StringConstraints(min_length=1, max_length=c.MAX_NAME_CHARS)]
    arguments: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Role
    text: Annotated[str, Field(max_length=c.MAX_MESSAGE_CHARS)]
    name: Annotated[str | None, Field(max_length=c.MAX_NAME_CHARS)] = None
    tool_calls: Annotated[tuple[ToolCall, ...], Field(max_length=c.MAX_TOOL_CALLS)] = ()


class Draft(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    text: Annotated[str, Field(max_length=c.MAX_DRAFT_CHARS)]
    tool_calls: Annotated[tuple[ToolCall, ...], Field(max_length=c.MAX_TOOL_CALLS)] = ()


class State(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    context: Annotated[str | None, Field(max_length=c.MAX_CONTEXT_CHARS)] = None
    conversation: Annotated[tuple[Message, ...], Field(max_length=c.MAX_MESSAGES)] = ()
    draft: Draft | None = None
