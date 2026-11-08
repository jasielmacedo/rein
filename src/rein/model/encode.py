import json
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from transformers import PreTrainedTokenizerBase

from rein.model import constants as c
from rein.model.sanitize import TokenSanitizer
from rein.schema import DecisionRequest, Draft, Message, Question, State, ToolCall

LAYOUTS = (c.LAYOUT_ROWS, c.LAYOUT_LABELS)


class StateTooLongError(ValueError):
    pass


class RowTooLongError(ValueError):
    pass


@dataclass(frozen=True)
class SpecialIds:
    state_open: int
    question_open: int
    query_mark: int
    option_open: int
    option_close: int
    turn_open: int
    turn_close: int
    tool_call_open: int
    tool_call_close: int

    @classmethod
    def resolve(cls, tokenizer: PreTrainedTokenizerBase) -> "SpecialIds":
        added = tokenizer.get_added_vocab()
        tokens = {
            "state_open": c.STATE_OPEN,
            "question_open": c.QUESTION_OPEN,
            "query_mark": c.QUERY_MARK,
            "option_open": c.OPTION_OPEN,
            "option_close": c.OPTION_CLOSE,
            "turn_open": c.TURN_OPEN,
            "turn_close": c.TURN_CLOSE,
            "tool_call_open": c.TOOL_CALL_OPEN,
            "tool_call_close": c.TOOL_CALL_CLOSE,
        }
        missing = [token for token in tokens.values() if token not in added]
        if missing:
            raise ValueError(f"tokenizer lacks the delimiter tokens {missing}")
        return cls(**{name: added[token] for name, token in tokens.items()})


@dataclass(frozen=True)
class LabelFrame:
    """The pieces of a lettered row: the question, then every option after the fixed label of its slot, then the
    answer mark. Each piece starts at a special token, so the row tokenizes the same whole or piece by piece."""

    opener: tuple[int, ...]
    closer: tuple[int, ...]
    option_texts: tuple[str, ...]
    label_texts: tuple[str, ...]
    line_ids: Callable[[str, str], list[int]]

    def row(self, order: Sequence[int]) -> tuple[int, ...]:
        lines = [
            token
            for slot, index in enumerate(order)
            for token in self.line_ids(self.label_texts[slot], self.option_texts[index])
        ]
        return (*self.opener, *lines, *self.closer)


@dataclass(frozen=True)
class QuestionRows:
    question_id: str
    question_type: str
    option_keys: tuple[str, ...]
    rows: tuple[tuple[int, ...], ...]
    query_index: int
    option_end_indices: tuple[int, ...]
    label_index: int | None
    target: tuple[float, ...] | None
    label_token_ids: tuple[int, ...] | None = None
    frame: LabelFrame | None = None

    def shuffled(self, rng: random.Random) -> "QuestionRows":
        order = list(range(len(self.option_keys)))
        rng.shuffle(order)
        label = None if self.label_index is None else order.index(self.label_index)
        target = None if self.target is None else tuple(self.target[i] for i in order)
        keys = tuple(self.option_keys[i] for i in order)
        if self.frame is None:
            return replace(
                self,
                option_keys=keys,
                rows=tuple(self.rows[i] for i in order),
                option_end_indices=tuple(self.option_end_indices[i] for i in order),
                label_index=label,
                target=target,
            )
        row = self.frame.row(order)
        return replace(
            self,
            option_keys=keys,
            rows=(row,),
            query_index=len(row) - 1,
            option_end_indices=(len(row) - 1,) * len(order),
            label_index=label,
            target=target,
        )


@dataclass(frozen=True)
class EncodedRequest:
    state_ids: tuple[int, ...]
    questions: tuple[QuestionRows, ...]
    dropped_messages: int = 0

    def with_shuffled_options(self, rng: random.Random) -> "EncodedRequest":
        return replace(self, questions=tuple(question.shuffled(rng) for question in self.questions))


class RequestEncoder:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_state_tokens: int = c.DEFAULT_MAX_STATE_TOKENS,
        max_row_tokens: int = c.MAX_ROW_TOKENS,
        layout: str = c.LAYOUT_ROWS,
        soft_front: Sequence[int] = (),
        soft_tail: Sequence[int] = (),
    ) -> None:
        if max_state_tokens < c.MIN_STATE_TOKENS:
            raise ValueError(f"max_state_tokens must be at least {c.MIN_STATE_TOKENS}")
        if tokenizer.pad_token_id is None:
            raise ValueError("tokenizer needs a pad token")
        if layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")
        if (soft_front or soft_tail) and layout != c.LAYOUT_LABELS:
            raise ValueError("soft tokens belong to the lettered layout")
        self.tokenizer = tokenizer
        self.max_state_tokens = max_state_tokens
        self.max_row_tokens = max_row_tokens
        self.layout = layout
        self.soft_front = tuple(soft_front)
        self.soft_tail = tuple(soft_tail)
        self.ids = SpecialIds.resolve(tokenizer)
        self.pad_id = int(tokenizer.pad_token_id)
        self.sanitize = TokenSanitizer(tokenizer)
        self._newline = self.text_ids(c.NEWLINE)
        self.label_ids, self.label_texts = self._label_codes() if layout == c.LAYOUT_LABELS else ((), ())

    def text_ids(self, text: str) -> list[int]:
        return list(self.tokenizer(self.sanitize(text), add_special_tokens=False)["input_ids"])

    def encode(self, request: DecisionRequest) -> EncodedRequest:
        state_ids, dropped = self.encode_state(request.state)
        questions = tuple(self.encode_question(qid, question) for qid, question in request.questions.items())
        return EncodedRequest(state_ids=tuple(state_ids), questions=questions, dropped_messages=dropped)

    def encode_state(self, state: State) -> tuple[list[int], int]:
        head = [self.ids.state_open]
        if state.context:
            head += self.text_ids(c.CONTEXT_HEADER + state.context + c.NEWLINE)
        head += self.text_ids(c.CONVERSATION_HEADER)
        tail = self._draft_ids(state.draft) if state.draft is not None else []
        budget = self.max_state_tokens - len(head) - len(tail)
        if budget < 0:
            raise StateTooLongError("context and draft alone exceed the state budget")
        blocks = [self._message_ids(message) for message in state.conversation]
        kept: list[list[int]] = []
        used = 0
        for block in reversed(blocks):
            if used + len(block) > budget:
                break
            kept.append(block)
            used += len(block)
        if blocks and not kept:
            raise StateTooLongError("the latest message does not fit in the state budget")
        body = [token for block in reversed(kept) for token in block]
        return head + body + tail, len(blocks) - len(kept)

    def encode_question(self, question_id: str, question: Question) -> QuestionRows:
        if self.layout == c.LAYOUT_LABELS:
            return self._encode_lettered(question_id, question)
        prefix = [self.ids.question_open, *self.text_ids(question.instructions), self.ids.query_mark]
        rows: list[tuple[int, ...]] = []
        ends: list[int] = []
        for text in question.option_texts():
            row = (*prefix, self.ids.option_open, *self.text_ids(text), self.ids.option_close)
            if len(row) > self.max_row_tokens:
                raise RowTooLongError(f"question {question_id!r} has a row of {len(row)} tokens")
            rows.append(row)
            ends.append(len(row) - 1)
        label_index, target = self._supervision(question, len(rows))
        return QuestionRows(
            question_id=question_id,
            question_type=question.type,
            option_keys=question.option_keys(),
            rows=tuple(rows),
            query_index=len(prefix) - 1,
            option_end_indices=tuple(ends),
            label_index=label_index,
            target=target,
        )

    def _encode_lettered(self, question_id: str, question: Question) -> QuestionRows:
        """One row per question: the options in order, each after the letter of its slot, read at the answer mark."""
        texts = question.option_texts()
        if len(texts) > len(self.label_ids):
            raise RowTooLongError(f"question {question_id!r} has {len(texts)} options, more than the labels")
        frame = LabelFrame(
            opener=(self.ids.question_open, *self.soft_front, *self.text_ids(question.instructions)),
            closer=(*self.soft_tail, self.ids.query_mark),
            option_texts=tuple(texts),
            label_texts=self.label_texts[: len(texts)],
            line_ids=self._line_ids,
        )
        row = frame.row(range(len(texts)))
        if len(row) > self.max_row_tokens:
            raise RowTooLongError(f"question {question_id!r} has a row of {len(row)} tokens")
        label_index, target = self._supervision(question, len(texts))
        return QuestionRows(
            question_id=question_id,
            question_type=question.type,
            option_keys=question.option_keys(),
            rows=(row,),
            query_index=len(row) - 1,
            option_end_indices=(len(row) - 1,) * len(texts),
            label_index=label_index,
            target=target,
            label_token_ids=self.label_ids[: len(texts)],
            frame=frame,
        )

    def _line_ids(self, label: str, text: str) -> list[int]:
        return [self.ids.option_open, *self.text_ids(label + c.LABEL_SEPARATOR + text)]

    def _label_codes(self) -> tuple[tuple[int, ...], tuple[str, ...]]:
        """The option labels a runtime reads as one token each: A to Z, then AA to ZZ, keeping the single tokens."""
        codes = [*c.LABEL_LETTERS, *(first + second for first in c.LABEL_LETTERS for second in c.LABEL_LETTERS)]
        ids: list[int] = []
        texts: list[str] = []
        for code in codes:
            encoded = self.tokenizer(code, add_special_tokens=False)["input_ids"]
            if len(encoded) == 1:
                ids.append(int(encoded[0]))
                texts.append(code)
        return tuple(ids), tuple(texts)

    @staticmethod
    def _supervision(question: Question, options: int) -> tuple[int | None, tuple[float, ...] | None]:
        label_index = question.label_index()
        target = question.target_vector()
        if target is None and label_index is not None:
            target = tuple(1.0 if i == label_index else 0.0 for i in range(options))
        return label_index, target

    def _message_ids(self, message: Message) -> list[int]:
        header = message.role if message.name is None else f"{message.role} {message.name}"
        ids = [self.ids.turn_open, *self.text_ids(header + c.NEWLINE + message.text)]
        ids += self._tool_call_ids(message.tool_calls)
        return [*ids, self.ids.turn_close, *self._newline]

    def _draft_ids(self, draft: Draft) -> list[int]:
        ids = [*self.text_ids(c.DRAFT_HEADER), self.ids.turn_open]
        ids += self.text_ids(c.DRAFT_ROLE + c.NEWLINE + draft.text)
        return ids + self._tool_call_ids(draft.tool_calls)

    def _tool_call_ids(self, calls: Sequence[ToolCall]) -> list[int]:
        ids: list[int] = []
        for call in calls:
            payload = json.dumps({"name": call.name, "arguments": call.arguments}, ensure_ascii=False, sort_keys=True)
            ids += [self.ids.tool_call_open, *self.text_ids(c.NEWLINE + payload + c.NEWLINE), self.ids.tool_call_close]
        return ids
