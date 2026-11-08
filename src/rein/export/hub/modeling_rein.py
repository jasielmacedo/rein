"""Rein decision model: Qwen3.5 text decoder plus a set-aware pointer head that scores the options of typed rules.

Load it with `AutoModel.from_pretrained(repo, trust_remote_code=True)`. `session(sheet, generator)` puts a chat
generator under a rule sheet, steered by Rein. `decide(state, questions)` returns the raw
decisions. Needs only torch and transformers. Rein never generates text and has no lm_head.
"""

import json
import re
from dataclasses import dataclass
from functools import partial
from typing import Any, cast

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
from transformers.utils import cached_file

from .rein_control import DEFAULT_GENERATOR, Generator, OpenAIChatGenerator, QwenGenerator, RuleSession

HEAD_FILE = "head.safetensors"

STATE_OPEN = "<|fim_prefix|>"
QUESTION_OPEN = "<|fim_middle|>"
QUERY_MARK = "<|fim_suffix|>"
OPTION_OPEN = "<|box_start|>"
OPTION_CLOSE = "<|box_end|>"
TURN_OPEN = "<|im_start|>"
TURN_CLOSE = "<|im_end|>"
TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
CONTEXT_HEADER = "Context:\n"
CONVERSATION_HEADER = "Conversation:\n"
DRAFT_HEADER = "Reply in progress:\n"
DRAFT_ROLE = "assistant"
NEWLINE = "\n"
SANITIZE_OPEN = chr(0x2039)
SANITIZE_CLOSE = chr(0x203A)

ROLES = frozenset({"user", "assistant", "narrator", "tool", "system"})
NOUL_OPTION_KEYS = ("no", "yes")
NOUL_CRITERIA_KEYS = frozenset({"true", "false"})
MAX_QUESTIONS = 64
MAX_CHOICE_OPTIONS = 64
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 11
MAX_INSTRUCTION_CHARS = 2_000
MAX_OPTION_TEXT_CHARS = 600
MAX_MESSAGE_CHARS = 20_000
MAX_MESSAGES = 400
MAX_CONTEXT_CHARS = 40_000
MAX_ROW_TOKENS = 1024
FULL_ATTENTION = "full_attention"
SDPA = "sdpa"
PAD_DECAY_LOGIT = -1.0e4


@dataclass(frozen=True)
class _Question:
    question_id: str
    option_keys: tuple[str, ...]
    rows: tuple[tuple[int, ...], ...]
    query_index: int
    option_ends: tuple[int, ...]


def _text(value: Any, name: str, limit: int, required: bool = True) -> str:
    if value is None and not required:
        return ""
    if not isinstance(value, str) or (required and not value) or len(value) > limit:
        raise ValueError(f"{name} must be a non-empty string of at most {limit} characters")
    return value


def _join(key: str, description: Any) -> str:
    return f"{key}: {description}" if description else key


def question_options(question: dict[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Option keys and the text each option row reads, for `noul`, `choice` and `score` questions."""
    kind = question.get("type")
    criteria = question.get("criteria")
    if kind == "noul":
        if criteria is not None and (not isinstance(criteria, dict) or not set(criteria) <= NOUL_CRITERIA_KEYS):
            raise ValueError("noul criteria may only describe 'true' and 'false'")
        described = criteria or {}
        texts = tuple(_join(k, described.get(f)) for k, f in zip(NOUL_OPTION_KEYS, ("false", "true"), strict=True))
        return NOUL_OPTION_KEYS, texts
    if kind == "choice":
        if not isinstance(criteria, dict) or not 1 <= len(criteria) <= MAX_CHOICE_OPTIONS:
            raise ValueError(f"choice needs 1 to {MAX_CHOICE_OPTIONS} named options")
        for key, text in criteria.items():
            _text(key, "option key", MAX_OPTION_TEXT_CHARS)
            _text(text, "option text", MAX_OPTION_TEXT_CHARS, required=False)
        return tuple(criteria), tuple(_join(key, text) for key, text in criteria.items())
    if kind == "score":
        if not isinstance(criteria, list) or not MIN_SCORE_LEVELS <= len(criteria) <= MAX_SCORE_LEVELS:
            raise ValueError(f"score needs {MIN_SCORE_LEVELS} to {MAX_SCORE_LEVELS} ordered levels")
        levels = [_text(text, "level text", MAX_OPTION_TEXT_CHARS) for text in criteria]
        return tuple(str(i) for i in range(len(levels))), tuple(f"{i}: {text}" for i, text in enumerate(levels))
    raise ValueError("question type must be noul, choice or score")


class RequestEncoder:
    """Turns a state and its questions into token ids: one shared state, one short row per option."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_state_tokens: int) -> None:
        added = tokenizer.get_added_vocab()
        names = (STATE_OPEN, QUESTION_OPEN, QUERY_MARK, OPTION_OPEN, OPTION_CLOSE, TURN_OPEN, TURN_CLOSE)
        missing = [token for token in (*names, TOOL_CALL_OPEN, TOOL_CALL_CLOSE) if token not in added]
        if missing or tokenizer.pad_token_id is None:
            raise ValueError(f"tokenizer lacks the Rein delimiters {missing} or a pad token")
        self.tokenizer = tokenizer
        self.max_state_tokens = max_state_tokens
        self.special = {token: added[token] for token in (*names, TOOL_CALL_OPEN, TOOL_CALL_CLOSE)}
        self.pad_id = int(tokenizer.pad_token_id)
        self._forgeable = re.compile("|".join(re.escape(t) for t in sorted(added, key=len, reverse=True)))
        self._newline = self.text_ids(NEWLINE)

    def sanitize(self, text: str) -> str:
        """User text cannot forge a delimiter: special tokens it spells are rewritten with look-alike brackets."""
        return self._forgeable.sub(lambda m: SANITIZE_OPEN + m.group(0)[1:-1] + SANITIZE_CLOSE, text)

    def text_ids(self, text: str) -> list[int]:
        return list(self.tokenizer(self.sanitize(text), add_special_tokens=False)["input_ids"])

    def _calls(self, calls: Any) -> list[int]:
        ids: list[int] = []
        for call in calls or ():
            payload = json.dumps(
                {"name": call["name"], "arguments": call.get("arguments", {})}, ensure_ascii=False, sort_keys=True
            )
            ids += [self.special[TOOL_CALL_OPEN], *self.text_ids(NEWLINE + payload + NEWLINE)]
            ids.append(self.special[TOOL_CALL_CLOSE])
        return ids

    def _message(self, message: dict[str, Any]) -> list[int]:
        role = message.get("role")
        if role not in ROLES:
            raise ValueError(f"message role must be one of {sorted(ROLES)}")
        text = _text(message.get("text", ""), "message text", MAX_MESSAGE_CHARS, required=False)
        header = role if not message.get("name") else f"{role} {message['name']}"
        ids = [
            self.special[TURN_OPEN],
            *self.text_ids(header + NEWLINE + text),
            *self._calls(message.get("tool_calls")),
        ]
        return [*ids, self.special[TURN_CLOSE], *self._newline]

    def state(self, state: dict[str, Any]) -> list[int]:
        context = _text(state.get("context"), "context", MAX_CONTEXT_CHARS, required=False)
        conversation = state.get("conversation") or []
        if not isinstance(conversation, list) or len(conversation) > MAX_MESSAGES:
            raise ValueError(f"conversation must be a list of at most {MAX_MESSAGES} messages")
        head = [self.special[STATE_OPEN]]
        if context:
            head += self.text_ids(CONTEXT_HEADER + context + NEWLINE)
        head += self.text_ids(CONVERSATION_HEADER)
        tail: list[int] = []
        draft = state.get("draft")
        if draft:
            text = _text(draft.get("text", ""), "draft text", MAX_MESSAGE_CHARS, required=False)
            tail = [*self.text_ids(DRAFT_HEADER), self.special[TURN_OPEN], *self.text_ids(DRAFT_ROLE + NEWLINE + text)]
            tail += self._calls(draft.get("tool_calls"))
        budget = self.max_state_tokens - len(head) - len(tail)
        blocks = [self._message(message) for message in conversation]
        kept: list[list[int]] = []
        used = 0
        for block in reversed(blocks):
            if used + len(block) > budget:
                break
            kept.append(block)
            used += len(block)
        if budget < 0 or (blocks and not kept):
            raise ValueError("the state does not fit in the token budget; shorten the context or the draft")
        return head + [token for block in reversed(kept) for token in block] + tail

    def question(self, question_id: str, question: dict[str, Any]) -> _Question:
        instructions = _text(question.get("instructions"), "instructions", MAX_INSTRUCTION_CHARS)
        keys, texts = question_options(question)
        prefix = [self.special[QUESTION_OPEN], *self.text_ids(instructions), self.special[QUERY_MARK]]
        rows = []
        for text in texts:
            row = (*prefix, self.special[OPTION_OPEN], *self.text_ids(text), self.special[OPTION_CLOSE])
            if len(row) > MAX_ROW_TOKENS:
                raise ValueError(f"question {question_id!r} has an option row of {len(row)} tokens")
            rows.append(row)
        return _Question(question_id, keys, tuple(rows), len(prefix) - 1, tuple(len(row) - 1 for row in rows))


class _LayerPrefix:
    """Cache for one decoder layer: records the state pass, then serves it to every option row."""

    def __init__(self, owner: torch.Tensor) -> None:
        self._owner = owner
        self._serving = False
        self._conv_tail: torch.Tensor | None = None
        self._recurrent: torch.Tensor | None = None
        self._keys: torch.Tensor | None = None
        self._values: torch.Tensor | None = None

    def serve_rows(self) -> None:
        self._serving = True

    def get_conv_state(self, layer_idx: int, state_idx: int = 0) -> None:
        return None

    def get_recurrent_state(self, layer_idx: int, state_idx: int = 0) -> torch.Tensor | None:
        if not self._serving or self._recurrent is None:
            return None
        return self._recurrent.index_select(0, self._owner)

    def is_recording_past(self, layer_idx: int) -> bool:
        return False

    def update_conv_state(
        self, conv_states: torch.Tensor, layer_idx: int, conv_kernel_size: int | None = None, **kwargs: Any
    ) -> torch.Tensor:
        if not self._serving:
            kernel = conv_kernel_size or conv_states.shape[-1]
            tail = conv_states[..., -kernel:]
            if tail.shape[-1] < kernel:
                tail = F.pad(tail, (kernel - tail.shape[-1], 0))
            self._conv_tail = tail
            return conv_states
        assert self._conv_tail is not None
        return torch.cat([self._conv_tail.index_select(0, self._owner).to(conv_states.dtype), conv_states], dim=-1)

    def update_recurrent_state(self, recurrent_states: torch.Tensor, layer_idx: int, **kwargs: Any) -> torch.Tensor:
        if not self._serving:
            self._recurrent = recurrent_states
        return recurrent_states

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._serving:
            self._keys, self._values = key_states, value_states
            return key_states, value_states
        assert self._keys is not None and self._values is not None
        keys = torch.cat([self._keys.index_select(0, self._owner), key_states], dim=-2)
        values = torch.cat([self._values.index_select(0, self._owner), value_states], dim=-2)
        return keys, values


def _attention_mask(allow: torch.Tensor, implementation: str, dtype: torch.dtype) -> torch.Tensor:
    if implementation == SDPA:
        return allow
    return torch.zeros(allow.shape, dtype=dtype, device=allow.device).masked_fill(~allow, torch.finfo(dtype).min)


def _freeze_padding_decay(valid: torch.Tensor, module: nn.Module, inputs: object, output: torch.Tensor) -> torch.Tensor:
    return torch.where(valid[..., None], output, torch.full_like(output, PAD_DECAY_LOGIT))


class _SetAttention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, tokens: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        batch, length, dim = tokens.shape
        qkv = self.qkv(tokens).view(batch, length, 3, self.heads, dim // self.heads).permute(2, 0, 3, 1, 4)
        attended = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], attn_mask=valid[:, None, None, :])
        return self.out(attended.transpose(1, 2).reshape(batch, length, dim))


class _SetBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ffn_mult: int) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = _SetAttention(dim, heads)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_mult * dim), nn.GELU(), nn.Linear(ffn_mult * dim, dim))

    def forward(self, tokens: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        tokens = tokens + self.attn(self.attn_norm(tokens), valid)
        return tokens + self.ffn(self.ffn_norm(tokens))


class SetPointerHead(nn.Module):
    """Scores each option from the query and option hidden states; option order does not matter."""

    FFN_MULT = 4

    def __init__(self, hidden_size: int, dim: int, heads: int, set_block: bool = True) -> None:
        super().__init__()
        self.query_in = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, dim))
        self.option_in = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, dim))
        self.block = _SetBlock(dim, heads, self.FFN_MULT) if set_block else None
        self.query_out = nn.Linear(dim, dim)
        self.option_out = nn.Linear(dim, dim)
        self.scale = dim**-0.5

    def forward(self, h_query: torch.Tensor, h_options: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        query = self.query_in(h_query.float())
        options = self.option_in(h_options.float())
        if self.block is not None:
            tokens = self.block(torch.cat([query[:, None], options], 1), torch.cat([mask[:, :1] | True, mask], 1))
            query, options = tokens[:, 0], tokens[:, 1:]
        logits = (self.option_out(options) * self.query_out(query)[:, None]).sum(-1) * self.scale
        return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)


class ReinDecider(Qwen3_5TextModel):
    """The merged Rein backbone (a plain Qwen3.5 text decoder) with the decision head on top."""

    _keys_to_ignore_on_load_missing = [r"^rein_head\."]  # noqa: RUF012

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        rein = config.rein
        self.rein_head = SetPointerHead(config.hidden_size, rein["head"]["dim"], rein["head"]["heads"])
        self.temperature = float(rein["temperature"])
        self.encoder: RequestEncoder | None = None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Any, *args: Any, **kwargs: Any) -> "ReinDecider":
        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        hub = {key: kwargs[key] for key in ("revision", "token", "cache_dir") if key in kwargs}
        head = cached_file(pretrained_model_name_or_path, HEAD_FILE, **hub)
        model.rein_head.load_state_dict(load_file(str(head)))
        model.rein_head.float()
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **hub)
        model.encoder = RequestEncoder(tokenizer, int(model.config.rein["max_state_tokens"]))
        return model.eval()

    def _rows(self, state_ids: list[int], questions: list[_Question]) -> torch.Tensor:
        assert self.encoder is not None
        device = self.embed_tokens.weight.device
        rows = [row for question in questions for row in question.rows]
        row_len = max(len(row) for row in rows)
        state = torch.tensor([state_ids], device=device)
        row_ids = torch.full((len(rows), row_len), self.encoder.pad_id, device=device)
        row_mask = torch.zeros((len(rows), row_len), dtype=torch.bool, device=device)
        for r, row in enumerate(rows):
            row_ids[r, : len(row)] = torch.tensor(row, device=device)
            row_mask[r, : len(row)] = True
        owner = torch.zeros(len(rows), dtype=torch.long, device=device)
        state_valid = torch.ones_like(state, dtype=torch.bool)

        state_hidden = self.embed_tokens(state)
        row_hidden = self.embed_tokens(row_ids)
        state_rope = self.rotary_emb(state_hidden, torch.arange(len(state_ids), device=device)[None])
        row_positions = len(state_ids) + torch.arange(row_len, device=device)[None].expand(len(rows), -1)
        row_rope = self.rotary_emb(row_hidden, row_positions)
        implementation = self.config._attn_implementation
        causal = torch.tril(torch.ones(len(state_ids), len(state_ids), dtype=torch.bool, device=device))[None, None]
        row_causal = torch.tril(torch.ones(row_len, row_len, dtype=torch.bool, device=device))
        row_part = (row_causal[None] & row_mask[:, None, :]) | torch.eye(row_len, dtype=torch.bool, device=device)
        full_rows = torch.cat([state_valid.expand(len(rows), -1)[:, None, :].expand(-1, row_len, -1), row_part], -1)
        full_state = _attention_mask(causal, implementation, state_hidden.dtype)
        full_rows = _attention_mask(full_rows[:, None], implementation, state_hidden.dtype)
        linear_state = state_valid.to(state_hidden.dtype)

        for layer in cast(list[Any], self.layers):
            cache = _LayerPrefix(owner)
            full = layer.block_type == FULL_ATTENTION
            handle = None
            if not full:
                handle = layer.linear_attn.in_proj_a.register_forward_hook(partial(_freeze_padding_decay, state_valid))
            try:
                state_out = layer(
                    state_hidden,
                    position_embeddings=state_rope,
                    attention_mask=full_state if full else linear_state,
                    past_key_values=cache,
                )
            finally:
                if handle is not None:
                    handle.remove()
            cache.serve_rows()
            row_hidden = layer(
                row_hidden,
                position_embeddings=row_rope,
                attention_mask=full_rows if full else None,
                past_key_values=cache,
            )
            state_hidden = state_out
        return self.norm(row_hidden)

    @torch.no_grad()
    def decide(
        self, state: dict[str, Any], questions: dict[str, dict[str, Any]], temperature: float | None = None
    ) -> dict[str, dict[str, float]]:
        """Calibrated probabilities for every option of every question: {question_id: {option: probability}}."""
        if self.encoder is None:
            raise RuntimeError("load the model with from_pretrained so it has its tokenizer")
        if not isinstance(questions, dict) or not 1 <= len(questions) <= MAX_QUESTIONS:
            raise ValueError(f"ask between 1 and {MAX_QUESTIONS} questions")
        encoded = [self.encoder.question(str(qid), question) for qid, question in questions.items()]
        hidden = self._rows(self.encoder.state(state), encoded)
        width = max(len(question.option_keys) for question in encoded)
        query_rows, option_rows, option_pos, mask = [], [], [], []
        first = 0
        for question in encoded:
            count = len(question.rows)
            query_rows.append(hidden[first, question.query_index])
            option_rows.append([first + i for i in range(count)] + [first] * (width - count))
            option_pos.append(list(question.option_ends) + [0] * (width - count))
            mask.append([True] * count + [False] * (width - count))
            first += count
        device = hidden.device
        h_options = hidden[torch.tensor(option_rows, device=device), torch.tensor(option_pos, device=device)]
        valid = torch.tensor(mask, device=device)
        with torch.autocast(device_type=device.type, enabled=False):
            logits = self.rein_head(torch.stack(query_rows).float(), h_options.float(), valid)
        scale = self.temperature if temperature is None else temperature
        probs = torch.softmax((logits / scale).masked_fill(~valid, torch.finfo(logits.dtype).min), -1)
        return {
            question.question_id: dict(
                zip(question.option_keys, row[: len(question.option_keys)].tolist(), strict=True)
            )
            for question, row in zip(encoded, probs.cpu(), strict=True)
        }

    def session(
        self,
        sheet: dict[str, Any],
        generator: str | Generator = DEFAULT_GENERATOR,
        *,
        base_url: str | None = None,
        system: str | None = None,
        context: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        sampling: dict[str, Any] | None = None,
    ) -> RuleSession:
        """A generator under `sheet`, steered by this model. `generator` is a chat model id, loaded locally with
        transformers, or served at `base_url` by an OpenAI-compatible server such as vLLM; or any `Generator`."""
        if isinstance(generator, str):
            if base_url:
                generator = OpenAIChatGenerator(base_url, generator)
            else:
                generator = QwenGenerator(generator, str(self.device), sampling)
        return RuleSession(self, generator, sheet, system=system, context=context, tools=tools)
