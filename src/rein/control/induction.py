import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from string import Template
from typing import Protocol

from rein.control import constants as c
from rein.control.plan import Directive
from rein.control.sheet import Escalate, ForceTool, Prefill
from rein.schema import ToolCall
from rein.schema.constants import ID_PATTERN


class ToolCallSyntax(Protocol):
    """How one generator's chat template writes a tool call. Rein decides the same thing for every generator."""

    def opening(self, tool: str) -> str: ...

    def parse(self, text: str) -> ToolCall | None: ...


_QWEN_CALL = re.compile(r"<tool_call>\s*<function=([A-Za-z0-9_.:\-]{1,64})>(.*?)</function>\s*</tool_call>", re.DOTALL)
_QWEN_PARAMETER = re.compile(r"<parameter=([A-Za-z0-9_.:\-]{1,64})>\n?(.*?)\n?</parameter>", re.DOTALL)
_HERMES_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _bounded(text: str) -> str | None:
    return text if len(text) <= c.MAX_TOOL_ARGUMENT_CHARS else None


class QwenXmlToolCall:
    """Qwen3.5 template: <tool_call> <function=name> <parameter=key>value</parameter> </function> </tool_call>."""

    def opening(self, tool: str) -> str:
        return f"<tool_call>\n<function={tool}>\n"

    def parse(self, text: str) -> ToolCall | None:
        match = _QWEN_CALL.search(text)
        if match is None or _bounded(match.group(2)) is None:
            return None
        parameters = _QWEN_PARAMETER.findall(match.group(2))[: c.MAX_TOOL_PARAMETERS]
        return ToolCall(name=match.group(1), arguments={key: value for key, value in parameters})


class HermesJsonToolCall:
    """JSON inside <tool_call> markers, as in Hermes-style templates."""

    def opening(self, tool: str) -> str:
        return f'<tool_call>\n{{"name": "{tool}", "arguments": '

    def parse(self, text: str) -> ToolCall | None:
        match = _HERMES_CALL.search(text)
        if match is None or _bounded(match.group(1)) is None:
            return None
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict) or not isinstance(payload.get("arguments", {}), dict):
            return None
        name = payload.get("name")
        if not isinstance(name, str) or re.fullmatch(ID_PATTERN, name) is None:
            return None
        return ToolCall(name=name, arguments=payload.get("arguments", {}))


def fill(template: str, variables: Mapping[str, str]) -> str:
    """Only $name placeholders are replaced; str.format is avoided because it can reach object attributes."""
    return Template(template).safe_substitute(variables)


@dataclass(frozen=True)
class Opening:
    text: str
    tool: str | None = None


def render_opening(chosen: Directive, variables: Mapping[str, str], syntax: ToolCallSyntax) -> Opening:
    """The first tokens of the reply that put the generator where only the decided behavior can follow."""
    action = chosen.rule.action
    if isinstance(action, ForceTool):
        tool = action.tool_for(chosen.option)
        return Opening(syntax.opening(tool), tool)
    if isinstance(action, Prefill | Escalate):
        return Opening(fill(action.text, variables))
    raise TypeError(f"{type(action).__name__} does not open a reply")
