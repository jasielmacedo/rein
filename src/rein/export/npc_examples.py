"""The two requests of one Rein-NPC turn, written as files a game can send as they are: the decision, a completion of
one token with its top log-probabilities in llama.cpp's `/completion` format, and the reply, an OpenAI-compatible chat
request with the acting rule applied. The Rein-NPC cards show them and the Hub repos ship them beside the sheet. The
decision prompts are the encoder's own ids written out as text, and the answers come from a llama-server that serves
the model and reads those texts:

    python -m rein.export.npc_examples --sheet configs/sheets/examples/shop-keeper.yaml \\
        --say "Afternoon! Rough road out there. What have you got for sale?" \\
        --decider runs/checkpoints/<run> --server http://127.0.0.1:8080 --out configs/sheets/examples
"""

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from rein.control.server_decider import ServerDecisionBackend, decision_body, label_probabilities
from rein.genview.render import GeneratorView, PromptSpec
from rein.model.constants import TOOL_CALL_OPEN
from rein.model.encode import RequestEncoder
from rein.model.run_encoder import build_encoder
from rein.model.run_meta import read_meta
from rein.play.engine import MAX_REPLY_TOKENS
from rein.play.store import ASSISTANT, USER
from rein.schema import DecisionRequest, Message, State
from rein.sheets.e2e import REIN_ARM, TEMPERATURE, TOP_K, TOP_P, system_prompt
from rein.sheets.model import Sheet, load_sheet, tool_rule
from rein.sheets.rein_arm import TurnActions, decision_request, plan_actions

DECIDE_FILE = "decide.json"
REPLY_FILE = "reply.json"
BAN_BIAS = -100
NO_THINKING = {"enable_thinking": False}
JSON_INDENT = 2


@dataclass(frozen=True)
class DecisionPrompt:
    rule_id: str
    text: str
    letters: tuple[int, ...]
    options: tuple[str, ...]


def decision_prompts(
    encoder: RequestEncoder, tokenizer: PreTrainedTokenizerBase, request: DecisionRequest
) -> list[DecisionPrompt]:
    """Each question's state and row as the text a runtime sends, checked to tokenize back to the encoder's ids."""
    encoded = encoder.encode(request)
    prompts = []
    for question in encoded.questions:
        if not question.label_token_ids:
            raise ValueError("the examples need the lettered layout (readout labels)")
        ids = [*encoded.state_ids, *question.rows[0]]
        text = str(tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False))
        if list(tokenizer(text, add_special_tokens=False)["input_ids"]) != ids:
            raise ValueError(f"the {question.question_id} prompt does not tokenize back to its ids")
        prompts.append(
            DecisionPrompt(question.question_id, text, tuple(question.label_token_ids), tuple(question.option_keys))
        )
    return prompts


def ask(backend: ServerDecisionBackend, prompt: DecisionPrompt) -> dict[str, float]:
    logprobs = backend.next_token_logprobs(prompt.text)
    return dict(zip(prompt.options, label_probabilities(logprobs, prompt.letters, backend.temperature), strict=True))


def reply_body(
    view: GeneratorView, sheet: Sheet, history: Sequence[Message], actions: TurnActions, tool_call_id: int
) -> dict[str, Any]:
    """The reply as a chat request: the persona and scene as the system message, the sheet's tools, the conversation
    with this turn's reminders, the opened tool call as the reply's start, and the tool-call token banned on a veto."""
    spec = PromptSpec(
        system=system_prompt(sheet, REIN_ARM),
        state=State(conversation=tuple(history)),
        tools=tuple(sheet.tool_schemas()),
        reminders=actions.reminders,
    )
    messages: list[dict[str, Any]] = [dict(message) for message in view.messages(spec)]
    if actions.opening:
        messages.append({"role": ASSISTANT, "content": actions.opening})
    body: dict[str, Any] = {
        "messages": messages,
        "tools": list(spec.tools),
        "chat_template_kwargs": NO_THINKING,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "max_tokens": MAX_REPLY_TOKENS,
    }
    if actions.veto_tools:
        body["logit_bias"] = {str(tool_call_id): BAN_BIAS}
    return body


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=JSON_INDENT, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the decision and reply requests of one Rein-NPC turn.")
    parser.add_argument("--sheet", type=Path, required=True)
    parser.add_argument("--say", required=True, help="what the player says on the turn")
    parser.add_argument("--decider", type=Path, required=True, help="the training run, for its rein_config.json")
    parser.add_argument("--server", required=True, help="a llama-server serving the model")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sheet = load_sheet(args.sheet)
    history = [Message(role=USER, text=args.say)]
    meta = read_meta(args.decider)
    tokenizer = AutoTokenizer.from_pretrained(meta["base_repo"])
    encoder = build_encoder(meta, tokenizer)
    backend = ServerDecisionBackend(args.server, args.decider, encoder)
    prompts = decision_prompts(encoder, tokenizer, decision_request(sheet, history))
    actions = plan_actions(sheet, {prompt.rule_id: ask(backend, prompt) for prompt in prompts})
    tool_call_id = cast(int, tokenizer.convert_tokens_to_ids(TOOL_CALL_OPEN))
    tool_prompt = next(prompt for prompt in prompts if prompt.rule_id == tool_rule(sheet))
    write_json(args.out / DECIDE_FILE, decision_body(tool_prompt.text))
    write_json(args.out / REPLY_FILE, reply_body(GeneratorView(tokenizer), sheet, history, actions, tool_call_id))
    print(json.dumps(actions.as_record(), indent=JSON_INDENT))


if __name__ == "__main__":
    main()
