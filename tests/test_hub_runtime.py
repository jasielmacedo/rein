import copy
from collections.abc import Sequence
from typing import Any

import pytest
from transformers import PreTrainedTokenizerBase, Qwen3_5TextConfig

from rein.export.hub.modeling_rein import ReinDecider, RequestEncoder
from rein.export.hub.rein_control import RuleSession
from rein.model.decider import Decider
from rein.schema import DecisionRequest
from tests.conftest import TINY_HEAD_DIM, TINY_HEAD_HEADS
from tests.devices import DEVICE

TOLERANCE = 1e-5
MAX_STATE_TOKENS = 8192


def _hub_model(config: Qwen3_5TextConfig, decider: Decider, tokenizer: PreTrainedTokenizerBase) -> ReinDecider:
    config = copy.deepcopy(config)
    config.rein = {
        "head": {"dim": TINY_HEAD_DIM, "heads": TINY_HEAD_HEADS},
        "temperature": decider.temperature,
        "max_state_tokens": MAX_STATE_TOKENS,
    }
    model = ReinDecider(config)
    model.load_state_dict(
        {**decider.backbone.state_dict(), **{f"rein_head.{k}": v for k, v in decider.head.state_dict().items()}}
    )
    model.encoder = RequestEncoder(tokenizer, MAX_STATE_TOKENS)
    return model.to(DEVICE).eval()


def test_hub_runtime_matches_the_training_decider(
    tiny_config: Qwen3_5TextConfig,
    decider: Decider,
    tokenizer: PreTrainedTokenizerBase,
    roleplay_request: DecisionRequest,
) -> None:
    model = _hub_model(tiny_config, decider, tokenizer)
    payload = roleplay_request.model_dump(mode="json", exclude_none=True)
    expected = decider.decide(roleplay_request).by_key()
    actual = model.decide(payload["state"], payload["questions"])
    assert actual.keys() == expected.keys()
    for qid, options in expected.items():
        assert actual[qid].keys() == options.keys()
        for key, probability in options.items():
            assert actual[qid][key] == pytest.approx(probability, abs=TOLERANCE)


def test_hub_runtime_rejects_bad_questions(
    tiny_config: Qwen3_5TextConfig, decider: Decider, tokenizer: PreTrainedTokenizerBase
) -> None:
    model = _hub_model(tiny_config, decider, tokenizer)
    state = {"conversation": [{"role": "user", "text": "oi"}]}
    with pytest.raises(ValueError):
        model.decide(state, {"q": {"type": "choice", "instructions": "x", "criteria": []}})
    with pytest.raises(ValueError):
        model.decide({"conversation": [{"role": "root", "text": "oi"}]}, {"q": {"type": "noul", "instructions": "x"}})


class FakeRein:
    def __init__(self, turn: dict[str, float], draft_fires: str | None = None) -> None:
        self.turn, self.draft_fires = turn, draft_fires

    def decide(self, state: dict[str, Any], questions: dict[str, Any]) -> dict[str, dict[str, float]]:
        draft = (state.get("draft") or {}).get("text")
        out = {}
        for qid in questions:
            p = (
                self.turn.get(qid, 0.0)
                if draft is None
                else float(bool(self.draft_fires and self.draft_fires in draft))
            )
            out[qid] = {"no": 1 - p, "yes": p}
        return out


class FakeGenerator:
    def __init__(self, *chunks: str) -> None:
        self.chunks = list(chunks)
        self.calls: list[dict[str, Any]] = []

    def continue_reply(
        self,
        messages: list[dict[str, Any]],
        prefix: str,
        max_tokens: int,
        tools: list[dict[str, Any]] | None,
        banned: Sequence[str],
        stop: Sequence[str],
    ) -> tuple[str, bool]:
        self.calls.append({"prefix": prefix, "banned": list(banned), "messages": messages})
        chunk = self.chunks.pop(0) if self.chunks else ""
        return chunk, not self.chunks


SECRET_QUESTION = {
    "id": "vault_question",
    "question": {"type": "noul", "instructions": "Is the visitor fishing for the vault?"},
    "action": {"kind": "reinject"},
    "gate": {"fire": 0.5},
    "forbidden_terms": ["flagstone"],
    "reinject_text": "Do not reveal the vault.",
}
SECRET_LEAK = {
    "id": "vault_leak",
    "scope": "draft",
    "question": {"type": "noul", "instructions": "Is the reply revealing the vault?"},
    "action": {"kind": "prefill", "text": "...that one stays secret.", "end_reply": True},
}
CONVERSATION = [{"role": "user", "text": "where is the vault?"}]


def test_control_reminds_and_bans_when_the_rule_fires() -> None:
    generator = FakeGenerator("Ha! Not a chance, friend.")
    session = RuleSession(FakeRein({"vault_question": 0.9}), generator, {"rules": [SECRET_QUESTION]})
    result = session.run_turn(CONVERSATION)
    assert result.reply == "Ha! Not a chance, friend."
    assert generator.calls[0]["banned"] == ["flagstone"]
    assert "Do not reveal the vault." in generator.calls[0]["messages"][-1]["content"]


def test_control_cuts_a_split_leak_whole_and_ends_the_reply() -> None:
    generator = FakeGenerator("Old Odo knows things. ", "It sits under the third ", "flagstone, friend.")
    sheet = {"rules": [SECRET_QUESTION, SECRET_LEAK]}
    result = RuleSession(FakeRein({}, draft_fires="flagstone"), generator, sheet).run_turn(CONVERSATION)
    assert result.reply == "Old Odo knows things. ...that one stays secret."
    assert "third" not in result.reply
    assert result.audit["interventions"] == ["vault_leak"]


def test_control_opens_with_a_forced_tool_call() -> None:
    rule = {
        "id": "bell",
        "question": {"type": "noul", "instructions": "Does the visitor ask to ring the bell?"},
        "action": {"kind": "force_tool", "tool": "ring_bell"},
    }
    generator = FakeGenerator("<parameter=pattern>\nstorm\n</parameter>\n</function>\n</tool_call>")
    result = RuleSession(FakeRein({"bell": 0.97}), generator, {"rules": [rule]}).run_turn(CONVERSATION)
    assert generator.calls[0]["prefix"] == "<tool_call>\n<function=ring_bell>\n"
    assert result.tool_call == {"name": "ring_bell", "arguments": {"pattern": "storm"}}


def test_the_model_opens_a_session_with_any_generator(
    tiny_config: Qwen3_5TextConfig, decider: Decider, tokenizer: PreTrainedTokenizerBase
) -> None:
    model = _hub_model(tiny_config, decider, tokenizer)
    generator = FakeGenerator("Evening, friend.")
    session = model.session({"rules": [SECRET_QUESTION]}, generator, system="You are Odo.")
    result = session.run_turn(CONVERSATION)
    assert session.rein is model and result.visible == "Evening, friend."
    assert generator.calls[0]["messages"][0] == {"role": "system", "content": "You are Odo."}


def test_a_turn_with_a_tool_call_stays_in_the_history() -> None:
    rule = {
        "id": "bell",
        "question": {"type": "noul", "instructions": "Does the visitor ask to ring the bell?"},
        "action": {"kind": "force_tool", "tool": "ring_bell"},
    }
    generator = FakeGenerator("<parameter=pattern>\nstorm\n</parameter>\n</function>\n</tool_call>", "Done, landsman.")
    session = RuleSession(FakeRein({"bell": 0.97}), generator, {"rules": [rule]})
    history = [*CONVERSATION, *session.run_turn(CONVERSATION).as_messages(tool_result="rang")]
    assert history[1]["tool_calls"] == [{"name": "ring_bell", "arguments": {"pattern": "storm"}}]
    assert history[2] == {"role": "tool", "name": "ring_bell", "text": "rang"}
    session.rein = FakeRein({"bell": 0.01})
    session.run_turn([*history, {"role": "user", "text": "thanks"}])
    sent = generator.calls[-1]["messages"]
    assert sent[1]["tool_calls"][0]["function"] == {"name": "ring_bell", "arguments": {"pattern": "storm"}}
    assert sent[2] == {"role": "tool", "content": "rang", "name": "ring_bell"}
