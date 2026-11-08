"""Rein control layer: steers a chat generator, such as Qwen3.5 instruct, with Rein's decisions on a rule sheet.

Rein steers; the generator writes the reply. Each turn, Rein scores the sheet's rules on the conversation. A rule
that fires opens the reply (a tool call, a fixed phrase, a handoff), is reminded to the generator, or bans words for
that turn.
Rules with `scope: draft` watch the reply while it is written and cut it when it breaks a rule.
Needs torch and transformers; `OpenAIChatGenerator` also needs httpx.
"""

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from string import Template
from typing import Any, Protocol, cast

import torch

DEFAULT_GENERATOR = "Qwen/Qwen3.5-0.8B"
DEFAULT_FIRE = 0.85
TOOL_SUPPRESS_BELOW = 0.05
WATCH_REQUEST_FLOOR = 0.2
WATCH_WINDOW_TOKENS = 32
WATCH_HOLD_WINDOWS = 1
WATCH_MAX_INTERVENTIONS = 3
MAX_REPLY_TOKENS = 256
NARRATOR_PREFIX = "[narrator] "
REMINDER_HEADER = "Reminder for this reply:"
UNREQUESTED_ACTION_REMINDER = (
    "Only describe actions you actually perform with a tool. Do not narrate sending, giving or doing anything here."
)
TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
OPENING_ACTIONS = ("force_tool", "prefill", "escalate")
WATCH_ACTIONS = ("force_tool", "prefill")
END_MARKERS = ("<|im_end|>", "<|endoftext|>")
STOP_TAIL_TOKENS = 12
QWEN_CALL = re.compile(r"<tool_call>\s*<function=([A-Za-z0-9_.:\-]{1,64})>(.*?)</function>\s*</tool_call>", re.DOTALL)
QWEN_PARAMETER = re.compile(r"<parameter=([A-Za-z0-9_.:\-]{1,64})>\n?(.*?)\n?</parameter>", re.DOTALL)
QWEN_SAMPLING = {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.15}


class Generator(Protocol):
    def continue_reply(
        self,
        messages: list[dict[str, Any]],
        prefix: str,
        max_tokens: int,
        tools: list[dict[str, Any]] | None,
        banned: Sequence[str],
        stop: Sequence[str],
    ) -> tuple[str, bool]:
        """Continues the assistant reply that starts with `prefix`; returns the new text and whether it ended."""
        ...


def term_variants(terms: Sequence[str]) -> list[str]:
    forms = {form for term in terms for form in (term, term.lower(), term.capitalize(), term.title(), term.upper())}
    return sorted({prefix + form for form in forms for prefix in ("", " ")})


def _strip_end(text: str) -> str:
    for marker in END_MARKERS:
        text = text.replace(marker, "")
    return text


class QwenGenerator:
    """A local chat model through transformers, for example Qwen/Qwen3.5-0.8B or Qwen/Qwen3.5-4B."""

    def __init__(self, model_id: str, device: str = "cuda", sampling: dict[str, Any] | None = None) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map=device).eval()
        self.sampling = dict(QWEN_SAMPLING if sampling is None else sampling)
        tokenizer = self.tokenizer

        class StopOnText(StoppingCriteria):
            def __init__(self, start: int, stops: Sequence[str]) -> None:
                self.start, self.stops = start, tuple(stops)

            def __call__(
                self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any
            ) -> torch.BoolTensor:
                tail = tokenizer.decode(input_ids[0, max(self.start, input_ids.shape[1] - STOP_TAIL_TOKENS) :])
                hit = any(stop in tail for stop in self.stops)
                return torch.full((input_ids.shape[0],), hit, dtype=torch.bool, device=input_ids.device)  # type: ignore[return-value]

        self._stop_on_text = StopOnText
        self._criteria = StoppingCriteriaList

    @torch.no_grad()
    def continue_reply(
        self,
        messages: list[dict[str, Any]],
        prefix: str,
        max_tokens: int,
        tools: list[dict[str, Any]] | None,
        banned: Sequence[str],
        stop: Sequence[str],
    ) -> tuple[str, bool]:
        prompt = self.tokenizer.apply_chat_template(
            messages, tools=cast(Any, tools or None), add_generation_prompt=True, tokenize=False, enable_thinking=False
        )
        inputs = self.tokenizer(str(prompt) + prefix, return_tensors="pt", add_special_tokens=False).to(
            self.model.device
        )
        start = inputs["input_ids"].shape[1]
        bad = {tuple(self.tokenizer.encode(form, add_special_tokens=False)) for form in term_variants(banned)}
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            **self.sampling,
            **({"bad_words_ids": [list(ids) for ids in bad if ids]} if bad else {}),
            stopping_criteria=self._criteria([self._stop_on_text(start, stop)]) if stop else None,
        )
        new = output[0, start:]
        ended = len(new) < max_tokens or int(new[-1]) == self.tokenizer.eos_token_id
        text = str(self.tokenizer.decode(new, skip_special_tokens=False))
        return _strip_end(text), ended or any(marker in text for marker in END_MARKERS)


@dataclass
class OpenAIChatGenerator:
    """An OpenAI-compatible server that can continue a prefilled reply, such as vLLM (`continue_final_message`)."""

    base_url: str
    model: str
    headers: dict[str, str] = field(default_factory=dict)
    sampling: dict[str, Any] = field(default_factory=lambda: {"temperature": 0.7, "top_p": 0.8})
    timeout_s: float = 120.0

    def continue_reply(
        self,
        messages: list[dict[str, Any]],
        prefix: str,
        max_tokens: int,
        tools: list[dict[str, Any]] | None,
        banned: Sequence[str],
        stop: Sequence[str],
    ) -> tuple[str, bool]:
        import httpx

        body: dict[str, Any] = {"model": self.model, "max_tokens": max_tokens, **self.sampling}
        messages = [_wire_message(message, index) for index, message in enumerate(messages)]
        if prefix:
            messages = [*messages, {"role": "assistant", "content": prefix}]
            body.update({"continue_final_message": True, "add_generation_prompt": False})
        body["messages"] = messages
        if tools:
            body["tools"] = tools
        if banned:
            body["bad_words"] = term_variants(banned)
        if stop:
            body.update({"stop": list(stop), "include_stop_str_in_output": True})
        url = f"{self.base_url.rstrip('/')}/v1/chat/completions"
        response = httpx.post(url, json=body, headers=self.headers, timeout=self.timeout_s)
        response.raise_for_status()
        choice = response.json()["choices"][0]
        return choice["message"].get("content") or "", choice.get("finish_reason") != "length"


def chat_message(message: dict[str, Any]) -> dict[str, Any]:
    """A conversation message in chat-template form; narration goes to the chat model as a user turn."""
    role, text = message["role"], message.get("text", "")
    if role == "narrator":
        return {"role": "user", "content": NARRATOR_PREFIX + text}
    chat: dict[str, Any] = {"role": role, "content": text}
    if message.get("tool_calls"):
        chat["tool_calls"] = [
            {"type": "function", "function": {"name": call["name"], "arguments": call.get("arguments", {})}}
            for call in message["tool_calls"]
        ]
    if role == "tool" and message.get("name"):
        chat["name"] = message["name"]
    return chat


def _wire_message(message: dict[str, Any], index: int) -> dict[str, Any]:
    """OpenAI wire format: tool-call arguments as a JSON string, with ids that tool results point back to."""
    if message.get("tool_calls"):
        calls = [
            {
                "id": f"call_{index}_{n}",
                "type": "function",
                "function": {"name": c["function"]["name"], "arguments": json.dumps(c["function"]["arguments"])},
            }
            for n, c in enumerate(message["tool_calls"])
        ]
        return {**message, "tool_calls": calls}
    if message["role"] == "tool":
        return {**message, "tool_call_id": f"call_{index - 1}_0"}
    return message


@dataclass(frozen=True)
class Directive:
    rule: dict[str, Any]
    band: str
    option: str
    probability: float

    @property
    def kind(self) -> str:
        return str(self.rule["action"]["kind"])


def _tool_for(rule: dict[str, Any], option: str) -> str:
    action = rule["action"]
    return str(action["tool"] if action.get("tool") else action["tools"][option])


def _fire_on(rule: dict[str, Any]) -> list[str]:
    return list(rule.get("fire_on", ["yes"]))


def directive(rule: dict[str, Any], probabilities: dict[str, float]) -> Directive:
    option = max(_fire_on(rule), key=lambda key: probabilities.get(key, 0.0))
    probability = probabilities.get(option, 0.0)
    gate = rule.get("gate", {})
    if probability >= gate.get("fire", DEFAULT_FIRE):
        band = "tight"
    elif gate.get("reinject") is not None and probability >= gate["reinject"]:
        band = "loose"
    else:
        band = "off"
    return Directive(rule, band, option, probability)


def parse_tool_call(text: str) -> dict[str, Any] | None:
    match = QWEN_CALL.search(text)
    if match is None:
        return None
    return {"name": match.group(1), "arguments": dict(QWEN_PARAMETER.findall(match.group(2)))}


@dataclass
class TurnResult:
    reply: str
    tool_call: dict[str, Any] | None
    escalate: bool
    audit: dict[str, Any]

    @property
    def visible(self) -> str:
        """What the user sees: the reply up to the tool call, which belongs to the application."""
        return visible_text(self.reply)

    def as_messages(self, tool_result: str | None = None) -> list[dict[str, Any]]:
        """This turn as conversation history: the reply with its tool call, then the tool's result if given.

        Keep tool calls in the history, so Rein and the chat model both see that an action already happened.
        """
        message: dict[str, Any] = {"role": "assistant", "text": self.visible}
        if self.tool_call is None:
            return [message]
        message["tool_calls"] = [self.tool_call]
        if tool_result is None:
            return [message]
        return [message, {"role": "tool", "name": self.tool_call["name"], "text": tool_result}]


class RuleSession:
    """One generator under one rule sheet. Call `run_turn(conversation)` with the history ending at the user turn."""

    def __init__(
        self,
        rein: Any,
        generator: Generator,
        sheet: dict[str, Any],
        system: str | None = None,
        context: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_reply_tokens: int = MAX_REPLY_TOKENS,
    ) -> None:
        self.rein, self.generator, self.sheet = rein, generator, sheet
        self.system, self.context, self.tools = system, context, tools
        self.max_reply_tokens = max_reply_tokens
        self.variables = {str(k): str(v) for k, v in sheet.get("variables", {}).items()}

    def _rules(self, scope: str) -> list[dict[str, Any]]:
        return [rule for rule in self.sheet["rules"] if rule.get("scope", "turn") == scope]

    def _decide(self, conversation: list[dict[str, Any]], scope: str, draft: str | None = None) -> list[Directive]:
        rules = self._rules(scope)
        if not rules:
            return []
        state: dict[str, Any] = {"context": self.context, "conversation": conversation}
        if draft is not None:
            state["draft"] = {"text": draft}
        probabilities = self.rein.decide(state, {rule["id"]: rule["question"] for rule in rules})
        return [directive(rule, probabilities[rule["id"]]) for rule in rules]

    def _opening(self, directives: list[Directive], kinds: Sequence[str]) -> Directive | None:
        tight = [d for d in directives if d.band == "tight" and d.kind in kinds]
        return max(tight, key=lambda d: (d.rule.get("priority", 0), d.probability), default=None)

    def _render(self, chosen: Directive) -> str:
        if chosen.kind == "force_tool":
            return f"{TOOL_CALL_OPEN}\n<function={_tool_for(chosen.rule, chosen.option)}>\n"
        return Template(chosen.rule["action"]["text"]).safe_substitute(self.variables)

    def _messages(self, conversation: list[dict[str, Any]], reminders: list[str]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        system = "\n\n".join(part for part in (self.system, self.context) if part)
        if system:
            messages.append({"role": "system", "content": system})
        for message in conversation:
            messages.append(chat_message(message))
        if reminders:
            note = "\n".join([REMINDER_HEADER, *(f"- {reminder}" for reminder in reminders)])
            if messages and messages[-1]["role"] == "user":
                messages[-1] = {**messages[-1], "content": f"{messages[-1]['content']}\n\n{note}"}
            else:
                messages.append({"role": "user", "content": note})
        return messages

    def _generate(self, messages: list[dict[str, Any]], prefix: str, tools: Any, banned: Sequence[str]) -> str:
        text, _ = self.generator.continue_reply(
            messages, prefix, self.max_reply_tokens, tools, banned, (TOOL_CALL_CLOSE,)
        )
        return text

    def _watch(
        self,
        conversation: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        reply: str,
        tools: Any,
        banned: Sequence[str],
        audit: dict[str, Any],
    ) -> tuple[str, Directive | None]:
        """Writes the reply window by window and checks each one. A cut drops the held windows too."""
        written, passed, used = "", [0], 0
        while used < self.max_reply_tokens:
            chunk, ended = self.generator.continue_reply(
                messages, reply + written, WATCH_WINDOW_TOKENS, tools, banned, (TOOL_CALL_CLOSE,)
            )
            written += chunk
            used += WATCH_WINDOW_TOKENS
            if TOOL_CALL_OPEN not in written and chunk.strip():
                fired = self._opening(self._decide(conversation, "draft", reply + written), WATCH_ACTIONS)
                audit["watch_checks"] += 1
                if fired is not None:
                    return written[: passed[max(0, len(passed) - 1 - WATCH_HOLD_WINDOWS)]], fired
                passed.append(len(written))
            if ended or not chunk:
                break
        return written, None

    def _requested(self, directives: list[Directive], fired: Directive) -> bool:
        if fired.kind != "force_tool":
            return True
        tools = self._tool_probabilities(directives)
        return not tools or tools.get(_tool_for(fired.rule, fired.option), 0.0) >= WATCH_REQUEST_FLOOR

    def _tool_probabilities(self, directives: list[Directive]) -> dict[str, float]:
        tools: dict[str, float] = {}
        for d in directives:
            if d.kind == "force_tool":
                tool = _tool_for(d.rule, d.option)
                tools[tool] = max(tools.get(tool, 0.0), d.probability)
        return tools

    def run_turn(self, conversation: list[dict[str, Any]]) -> TurnResult:
        directives = self._decide(conversation, "turn")
        opening = self._opening(directives, OPENING_ACTIONS)
        reminders = [
            d.rule.get("reinject_text") or d.rule["question"]["instructions"]
            for d in directives
            if d is not opening and d.band != "off"
        ]
        banned = list(
            dict.fromkeys(t for d in directives if d.band != "off" for t in d.rule.get("forbidden_terms", []))
        )
        tool_odds = self._tool_probabilities(directives)
        forced = opening is not None and opening.kind == "force_tool"
        suppressed = bool(tool_odds) and not forced and max(tool_odds.values()) < TOOL_SUPPRESS_BELOW
        tools = None if suppressed else self.tools
        messages = self._messages(conversation, reminders)
        audit: dict[str, Any] = {
            "decisions": {d.rule["id"]: round(d.probability, 4) for d in directives},
            "opening": opening.rule["id"] if opening else None,
            "reinjected": [d.rule["id"] for d in directives if d is not opening and d.band != "off"],
            "banned_terms": banned,
            "tools_suppressed": suppressed,
            "watch_checks": 0,
            "interventions": [],
        }
        reply = self._render(opening) if opening else ""
        ended = opening is not None and opening.rule["action"].get("end_reply", False)
        watching = bool(self._rules("draft")) and not forced and not (opening and opening.kind == "escalate")
        while not ended:
            if not watching or len(audit["interventions"]) >= WATCH_MAX_INTERVENTIONS:
                reply += self._generate(messages, reply, tools, banned)
                break
            kept, fired = self._watch(conversation, messages, reply, tools, banned, audit)
            reply += kept
            if fired is None:
                break
            if not self._requested(directives, fired):
                audit["unrequested_action_dropped"] = fired.rule["id"]
                if not reply.strip():
                    retry = self._messages(conversation, [*reminders, UNREQUESTED_ACTION_REMINDER])
                    reply += self._generate(retry, "", tools, banned)
                break
            audit["interventions"].append(fired.rule["id"])
            reply += self._render(fired)
            if fired.kind == "force_tool":
                reply += self._generate(messages, reply, self.tools, banned)
                break
            ended = fired.rule["action"].get("end_reply", False)
        if suppressed and TOOL_CALL_OPEN in reply:
            reply = reply[: reply.index(TOOL_CALL_OPEN)]
        escalate = opening is not None and opening.kind == "escalate"
        return TurnResult(reply=reply, tool_call=parse_tool_call(reply), escalate=escalate, audit=audit)


def visible_text(reply: str) -> str:
    """What the user should see: the reply up to the tool call, which belongs to the application."""
    return reply.split(TOOL_CALL_OPEN, 1)[0].strip()
