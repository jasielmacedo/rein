import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from transformers import PreTrainedTokenizerBase

from rein.control.generator import ChatMessage, chat_messages, template_message
from rein.genview import constants as c
from rein.model.sanitize import TokenSanitizer
from rein.schema import Draft, State, ToolCall


@dataclass(frozen=True)
class PromptSpec:
    """What the generator sees before it replies: its system prompt, the conversation and the tools on offer."""

    system: str
    state: State
    tools: tuple[dict[str, Any], ...] = ()
    rules: tuple[dict[str, str], ...] = ()
    reminders: tuple[str, ...] = ()


@dataclass(frozen=True)
class View:
    ids: tuple[int, ...]
    prompt_length: int
    positions: Mapping[str, tuple[int, ...]] = field(default_factory=dict)


def _sanitized(value: Any, sanitize: TokenSanitizer) -> Any:
    if isinstance(value, str):
        return sanitize(value)
    if isinstance(value, Mapping):
        return {key: _sanitized(item, sanitize) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_sanitized(item, sanitize) for item in value]
    return value


def tool_call_markup(call: ToolCall, sanitize: TokenSanitizer) -> str:
    """A finished call as the Qwen3.5 template writes it in an assistant turn."""
    lines = [f"<tool_call>\n<function={call.name}>\n"]
    for key, value in call.arguments.items():
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        lines.append(f"<parameter={sanitize(key)}>\n{sanitize(text)}\n</parameter>\n")
    lines.append("</function>\n</tool_call>")
    return "".join(lines)


def draft_continuation(draft: Draft, sanitize: TokenSanitizer) -> str:
    """The text of a reply in progress, with its finished tool calls written the way the template writes them."""
    text = sanitize(draft.text)
    calls = [tool_call_markup(call, sanitize) for call in draft.tool_calls]
    joined = text + ("\n\n" if text and calls else "") + "\n".join(calls)
    return joined


class GeneratorView:
    """Token ids of a conversation exactly as Qwen3.5 renders it for generation, with the positions to read."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        silence_positions: int = c.DEFAULT_SILENCE_POSITIONS,
        chat_template: str | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.sanitize = TokenSanitizer(tokenizer)
        self.silence_positions = silence_positions

    def messages(self, spec: PromptSpec) -> list[ChatMessage]:
        messages = chat_messages(spec.state, spec.system, spec.reminders)
        return [_sanitized(template_message(message), self.sanitize) for message in messages]

    def _ids(self, spec: PromptSpec, thinking: bool, generation_prompt: bool) -> list[int]:
        text = self.tokenizer.apply_chat_template(
            self.messages(spec),
            tools=[_sanitized(tool, self.sanitize) for tool in spec.tools] or None,
            add_generation_prompt=generation_prompt,
            tokenize=False,
            enable_thinking=thinking,
            **self._template_options(spec),
        )
        return list(self.tokenizer(str(text), add_special_tokens=False)["input_ids"])

    def _template_options(self, spec: PromptSpec) -> dict[str, Any]:
        """The rules template and its `rules`, when the view uses them; Qwen's own template otherwise."""
        options: dict[str, Any] = {}
        if self.chat_template is not None:
            options["chat_template"] = self.chat_template
        if spec.rules:
            if self.chat_template is None:
                raise ValueError("rules need the rules chat template")
            options["rules"] = _sanitized(list(spec.rules), self.sanitize)
        return options

    def text_ids(self, text: str) -> list[int]:
        return list(self.tokenizer(text, add_special_tokens=False)["input_ids"]) if text else []

    def render(self, spec: PromptSpec, thinking: bool, key: str, continuation: Sequence[int] = ()) -> View:
        """`continuation` holds token ids already written into the reply: a draft, or reasoning the model produced."""
        prompt = self._ids(spec, thinking, generation_prompt=True)
        history = self._ids(spec, thinking, generation_prompt=False)
        if prompt[: len(history)] != history:
            raise ValueError("the generation prompt does not extend the conversation tokens")
        ids = prompt + list(continuation)
        sampled = random.Random(key).sample(range(len(history)), min(self.silence_positions, len(history)))
        positions: dict[str, tuple[int, ...]] = {
            c.REPLY_START: (len(prompt) - 1,),
            c.SILENCE: tuple(sorted(sampled)),
        }
        if continuation:
            positions[c.CONTINUATION_END] = (len(ids) - 1,)
        return View(tuple(ids), len(prompt), positions)

    def render_draft(self, spec: PromptSpec, draft: Draft, key: str) -> View:
        """A reply in progress follows the closed, empty reasoning block, as in non-thinking generation."""
        text = draft_continuation(draft, self.sanitize)
        return self.render(spec, thinking=False, key=key, continuation=self.text_ids(text))
