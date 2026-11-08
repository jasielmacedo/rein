import json
from collections.abc import Generator as PyGenerator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import httpx

from rein.control import constants as c
from rein.schema import Message, State

ChatMessage = dict[str, Any]
SSE_DATA = "data: "
SSE_DONE = "[DONE]"
Chunks = PyGenerator[str, None, None]


@dataclass(frozen=True)
class GeneratorRequest:
    messages: tuple[ChatMessage, ...]
    assistant_prefix: str = ""
    stop: tuple[str, ...] = field(default_factory=tuple)
    tools_enabled: bool = True
    banned_terms: tuple[str, ...] = field(default_factory=tuple)


def term_variants(terms: Sequence[str]) -> tuple[str, ...]:
    """Each term as the generator may spell it, in lower, sentence, title and upper case, with or without a space."""
    forms = {form for term in terms for form in (term, term.lower(), term.capitalize(), term.title(), term.upper())}
    return tuple(sorted({prefix + form for form in forms for prefix in ("", " ")}))


class Generator(Protocol):
    def stream(self, request: GeneratorRequest) -> Chunks: ...


def _chat_message(message: Message) -> ChatMessage:
    if message.role == "narrator":
        return {"role": "user", "content": c.NARRATOR_PREFIX + message.text}
    payload: ChatMessage = {"role": message.role, "content": message.text}
    if message.name and message.role == "tool":
        payload["name"] = message.name
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "type": "function",
                "id": f"call_{index}",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments, ensure_ascii=False)},
            }
            for index, call in enumerate(message.tool_calls)
        ]
    return payload


def template_message(message: ChatMessage) -> ChatMessage:
    """Chat templates want tool-call arguments as a mapping, while the OpenAI wire format sends a JSON string."""
    calls = message.get("tool_calls")
    if not calls:
        return message
    converted = []
    for call in calls:
        function = dict(call["function"])
        if isinstance(function.get("arguments"), str):
            function["arguments"] = json.loads(function["arguments"] or "{}")
        converted.append({**call, "function": function})
    return {**message, "tool_calls": converted}


def reminder_text(reminders: Sequence[str]) -> str:
    return "\n".join([c.REINJECT_HEADER, *(f"- {reminder}" for reminder in reminders)])


def chat_messages(
    state: State,
    system: str | None,
    reminders: Sequence[str] = (),
    placement: Literal["system_tail", "user_suffix"] = "user_suffix",
) -> tuple[ChatMessage, ...]:
    """The generator's context, with this turn's reminders near the end, where they weigh the most."""
    messages: list[ChatMessage] = []
    if system or state.context:
        messages.append({"role": "system", "content": "\n\n".join(part for part in (system, state.context) if part)})
    messages.extend(_chat_message(message) for message in state.conversation)
    if not reminders:
        return tuple(messages)
    note = reminder_text(reminders)
    if placement == "system_tail":
        messages.append({"role": "system", "content": note})
    elif messages and messages[-1]["role"] == "user":
        messages[-1] = {**messages[-1], "content": f"{messages[-1]['content']}\n\n{note}"}
    else:
        messages.append({"role": "user", "content": note})
    return tuple(messages)


@dataclass(frozen=True)
class OpenAICompatibleGenerator:
    """Streams from /v1/chat/completions. A prefilled reply uses vLLM's continue_final_message."""

    base_url: str
    model: str
    headers: Mapping[str, str] = field(default_factory=dict)
    max_tokens: int = 512
    temperature: float = 0.7
    timeout_s: float = 120.0
    extra_body: Mapping[str, Any] = field(default_factory=dict)

    def _body(self, request: GeneratorRequest) -> dict[str, Any]:
        messages = list(request.messages)
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": True,
            **self.extra_body,
        }
        if request.assistant_prefix:
            messages.append({"role": "assistant", "content": request.assistant_prefix})
            body.update({"continue_final_message": True, "add_generation_prompt": False})
        if request.stop:
            body["stop"] = list(request.stop)
        if not request.tools_enabled:
            body["tool_choice"] = "none"
        if request.banned_terms:
            body["bad_words"] = list(term_variants(request.banned_terms))
        body["messages"] = messages
        return body

    def stream(self, request: GeneratorRequest) -> Chunks:
        url = f"{self.base_url.rstrip('/')}/v1/chat/completions"
        with (
            httpx.Client(timeout=self.timeout_s, headers=dict(self.headers)) as client,
            client.stream("POST", url, json=self._body(request)) as response,
        ):
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith(SSE_DATA):
                    continue
                data = line[len(SSE_DATA) :].strip()
                if data == SSE_DONE:
                    return
                choices = json.loads(data).get("choices") or [{}]
                delta = choices[0].get("delta", {}).get("content")
                if delta:
                    yield delta
