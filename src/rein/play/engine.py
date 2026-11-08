"""A play character's reply to the conversation so far, as the end-to-end runner plays the checkpoint: the prompt
rendered exactly as the training views are, the model served by a local llama-server, and each tool call answered with
the sheet's tool result, then followed by the reply after it. With a decider run, the engine plays the Rein arm: the
persona alone in the prompt, every rule asked to the same server in Rein's layout at every turn, and the answers acted
on (the tool call opened, tool calls vetoed, the firing rule's base as a reminder); the decisions travel with the
reply."""

import secrets
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from rein.control.induction import QwenXmlToolCall
from rein.control.plan import DecisionBackend
from rein.genview import constants as gc
from rein.genview.render import GeneratorView
from rein.model.constants import TOOL_CALL_OPEN
from rein.play.store import ASSISTANT, TOOL, USER
from rein.schema import Message, ToolCall
from rein.sheets.e2e import GENERIC_ARM, REIN_ARM, Backend, LlamaServerBackend, render_prompt, visible_text
from rein.sheets.model import Sheet
from rein.sheets.rein_arm import AFTER_TOOL_RESULT, TurnActions, decision_request, plan_actions

MAX_CALLS_PER_REPLY = 4
MAX_REPLY_TOKENS = 400
SEED_RANGE = 2**31


def history(messages: Sequence[Mapping[str, Any]]) -> list[Message]:
    """The stored messages as the conversation the prompt is rendered from."""
    rebuilt = []
    for message in messages:
        if message["role"] == TOOL:
            rebuilt.append(Message(role=TOOL, name=message["tool_name"], text=message["text"]))
            continue
        call = message.get("tool_call")
        calls = (ToolCall.model_validate(call),) if call else ()
        rebuilt.append(Message(role=message["role"], text=message["text"], tool_calls=calls))
    return rebuilt


class Engine:
    def __init__(
        self,
        view: GeneratorView,
        backend: Backend,
        decide: DecisionBackend | None = None,
        tool_call_token: int | None = None,
    ) -> None:
        self.view = view
        self.backend = backend
        self.decide = decide
        self.tool_call_token = tool_call_token
        self.arm = REIN_ARM if decide is not None else GENERIC_ARM
        self.parser = QwenXmlToolCall()

    @classmethod
    def served(cls, llama_url: str, decider: Path | None = None) -> "Engine":
        """The generic checkpoint through its rules template, or, with a decider run, the Rein arm on the soft-token
        GGUF the same server serves."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(gc.CHAT_TOKENIZER_REPO)
        backend = LlamaServerBackend(llama_url, MAX_REPLY_TOKENS)
        if decider is None:
            template = Path(gc.RULES_TEMPLATE).read_text(encoding="utf-8")
            return cls(GeneratorView(tokenizer, chat_template=template), backend)
        from rein.control.server_decider import ServerDecisionBackend

        token = cast(list[int], tokenizer.convert_tokens_to_ids([TOOL_CALL_OPEN]))[0]
        return cls(GeneratorView(tokenizer), backend, ServerDecisionBackend(llama_url, decider), token)

    def actions(self, sheet: Sheet, conversation: Sequence[Message]) -> TurnActions:
        """What the decider asks of this reply: nothing without a decider, only the veto right after a tool result."""
        if self.decide is None:
            return TurnActions()
        if conversation and conversation[-1].role == TOOL:
            return AFTER_TOOL_RESULT
        return plan_actions(sheet, self.decide.decide(decision_request(sheet, conversation)))

    def reply(self, sheet: Sheet, past: Sequence[Mapping[str, Any]], text: str) -> list[dict[str, Any]]:
        """The user's message and every turn of the model's answer to it: its reply, and for each tool it calls (at
        most MAX_CALLS_PER_REPLY in a row, every one answered) the tool's result and the reply after it."""
        conversation = [*history(past), Message(role=USER, text=text)]
        turns: list[dict[str, Any]] = [{"role": USER, "text": text}]
        for _ in range(MAX_CALLS_PER_REPLY):
            actions = self.actions(sheet, conversation)
            ids = render_prompt(self.view, sheet, self.arm, conversation, actions.reminders, actions.opening)
            seed = secrets.randbelow(SEED_RANGE)
            bans = [(self.tool_call_token,)] if actions.veto_tools and self.tool_call_token is not None else [()]
            raw = actions.opening + self.backend.generate([ids], [seed], bans)[0]
            call = self.parser.parse(raw)
            reply = visible_text(raw)
            turns.append(
                {
                    "role": ASSISTANT,
                    "text": reply,
                    "tool_call": call.model_dump() if call else None,
                    "prompt_ids": ids,
                    "raw": raw,
                    "seed": seed,
                    "decisions": actions.as_record() if self.decide is not None else None,
                }
            )
            conversation.append(Message(role=ASSISTANT, text=reply, tool_calls=(call,) if call else ()))
            if call is None:
                break
            turns.append({"role": TOOL, "text": sheet.tool_result, "tool_name": call.name})
            conversation.append(Message(role=TOOL, name=call.name, text=sheet.tool_result))
        return turns
