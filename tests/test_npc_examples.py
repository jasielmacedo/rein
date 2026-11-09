import json
from pathlib import Path
from typing import Any, cast

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from rein.control.induction import QwenXmlToolCall
from rein.control.server_decider import decision_body
from rein.export.npc_examples import DECIDE_FILE, REPLY_FILE, decision_prompts, reply_body
from rein.genview.constants import CHAT_TOKENIZER_REPO
from rein.genview.render import GeneratorView
from rein.model.constants import TOOL_CALL_OPEN
from rein.model.encode import RequestEncoder
from rein.model.run_encoder import build_encoder
from rein.schema import Message
from rein.sheets.model import Sheet, load_sheet, tool_rule
from rein.sheets.rein_arm import TurnActions, decision_request

EXAMPLES = Path("configs/sheets/examples")
SHEET = EXAMPLES / "shop-keeper.yaml"
META = {"soft_tokens": {"front": 48, "tail": 16}, "max_state_tokens": 8192, "readout": "labels"}
FIRST_SOFT_ID = 248077
SOFT_TOKENS = 64
TOOL_CALL_ID = 248058


def shipped(name: str) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads((EXAMPLES / name).read_text(encoding="utf-8")))


@pytest.fixture(scope="module")
def chat() -> tuple[PreTrainedTokenizerBase, RequestEncoder]:
    tokenizer = AutoTokenizer.from_pretrained(CHAT_TOKENIZER_REPO)
    return tokenizer, build_encoder(META, tokenizer)


@pytest.fixture(scope="module")
def turn() -> tuple[Sheet, list[Message]]:
    first_line = shipped(REPLY_FILE)["messages"][1]["content"]
    return load_sheet(SHEET), [Message(role="user", text=first_line)]


def test_the_shipped_decision_is_the_encoders_prompt_as_text(
    chat: tuple[PreTrainedTokenizerBase, RequestEncoder], turn: tuple[Sheet, list[Message]]
) -> None:
    tokenizer, encoder = chat
    sheet, history = turn
    prompts = decision_prompts(encoder, tokenizer, decision_request(sheet, history))
    tool_prompt = next(prompt for prompt in prompts if prompt.rule_id == tool_rule(sheet))
    assert shipped(DECIDE_FILE) == decision_body(tool_prompt.text)
    assert tool_prompt.options == ("show_inventory", "sell_item", "buy_item", "repair_gear", "call_guards", "none")
    assert tool_prompt.letters == tuple(cast(list[int], tokenizer.convert_tokens_to_ids(list("ABCDEF"))))


def test_every_soft_token_of_the_shipped_decision_is_one_token(
    chat: tuple[PreTrainedTokenizerBase, RequestEncoder],
) -> None:
    tokenizer, _ = chat
    ids = tokenizer(shipped(DECIDE_FILE)["prompt"], add_special_tokens=False)["input_ids"]
    soft = [token for token in ids if token >= FIRST_SOFT_ID]
    assert soft == list(range(FIRST_SOFT_ID, FIRST_SOFT_ID + SOFT_TOKENS))


def test_the_shipped_reply_is_the_chat_request_with_the_call_opened(
    chat: tuple[PreTrainedTokenizerBase, RequestEncoder], turn: tuple[Sheet, list[Message]]
) -> None:
    tokenizer, _ = chat
    sheet, history = turn
    opened = TurnActions(opening=QwenXmlToolCall().opening("show_inventory"))
    tool_call_id = cast(int, tokenizer.convert_tokens_to_ids(TOOL_CALL_OPEN))
    assert tool_call_id == TOOL_CALL_ID
    assert shipped(REPLY_FILE) == reply_body(GeneratorView(tokenizer), sheet, history, opened, tool_call_id)


def test_a_vetoed_reply_bans_the_tool_call_token_and_opens_nothing(
    chat: tuple[PreTrainedTokenizerBase, RequestEncoder], turn: tuple[Sheet, list[Message]]
) -> None:
    tokenizer, _ = chat
    sheet, history = turn
    body = reply_body(GeneratorView(tokenizer), sheet, history, TurnActions(veto_tools=True), TOOL_CALL_ID)
    assert body["logit_bias"] == {str(TOOL_CALL_ID): -100}
    assert body["messages"][-1]["role"] == "user"
