from collections.abc import Sequence
from dataclasses import dataclass, fields, replace
from typing import Any

import torch

from rein.model.encode import EncodedRequest


@dataclass(frozen=True)
class BatchPadding:
    """Rounds batch dimensions up to multiples, so kernels see fewer distinct shapes. Padding never changes outputs."""

    state_multiple: int = 1
    row_multiple: int = 1
    row_count_multiple: int = 1

    def __post_init__(self) -> None:
        if min(self.state_multiple, self.row_multiple, self.row_count_multiple) < 1:
            raise ValueError("padding multiples must be at least 1")


NO_PADDING = BatchPadding()


def _round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


@dataclass
class DecisionBatch:
    state_ids: torch.Tensor
    state_mask: torch.Tensor
    row_ids: torch.Tensor
    row_mask: torch.Tensor
    row_owner: torch.Tensor
    query_row: torch.Tensor
    query_pos: torch.Tensor
    option_row: torch.Tensor
    option_pos: torch.Tensor
    option_mask: torch.Tensor
    option_token_ids: torch.Tensor
    targets: torch.Tensor
    has_target: torch.Tensor
    label_index: torch.Tensor
    question_owner: torch.Tensor
    question_ids: list[str]
    question_types: list[str]
    option_keys: list[tuple[str, ...]]

    def to(self, device: torch.device | str) -> "DecisionBatch":
        moved: dict[str, Any] = {
            field.name: value.to(device)
            for field in fields(self)
            if isinstance(value := getattr(self, field.name), torch.Tensor)
        }
        return replace(self, **moved)

    @property
    def num_records(self) -> int:
        return int(self.state_ids.shape[0])


def collate(encoded: Sequence[EncodedRequest], pad_id: int, padding: BatchPadding = NO_PADDING) -> DecisionBatch:
    if not encoded:
        raise ValueError("cannot collate an empty batch")
    state_len = _round_up(max(len(request.state_ids) for request in encoded), padding.state_multiple)
    state_ids = torch.full((len(encoded), state_len), pad_id, dtype=torch.long)
    state_mask = torch.zeros((len(encoded), state_len), dtype=torch.bool)
    for b, request in enumerate(encoded):
        start = state_len - len(request.state_ids)
        state_ids[b, start:] = torch.tensor(request.state_ids, dtype=torch.long)
        state_mask[b, start:] = True

    rows: list[tuple[int, ...]] = []
    row_owner: list[int] = []
    query_row: list[int] = []
    query_pos: list[int] = []
    option_rows: list[list[int]] = []
    option_pos: list[list[int]] = []
    option_tokens: list[tuple[int, ...]] = []
    targets: list[tuple[float, ...] | None] = []
    labels: list[int] = []
    question_owner: list[int] = []
    question_ids: list[str] = []
    question_types: list[str] = []
    option_keys: list[tuple[str, ...]] = []
    for b, request in enumerate(encoded):
        for question in request.questions:
            first = len(rows)
            rows.extend(question.rows)
            row_owner.extend([b] * len(question.rows))
            query_row.append(first)
            query_pos.append(question.query_index)
            option_rows.append([first + min(i, len(question.rows) - 1) for i in range(len(question.option_keys))])
            option_pos.append(list(question.option_end_indices))
            option_tokens.append(question.label_token_ids or ())
            targets.append(question.target)
            labels.append(-1 if question.label_index is None else question.label_index)
            question_owner.append(b)
            question_ids.append(question.question_id)
            question_types.append(question.question_type)
            option_keys.append(question.option_keys)

    filler = _round_up(len(rows), padding.row_count_multiple) - len(rows)
    rows.extend([(pad_id,)] * filler)
    row_owner.extend([0] * filler)
    row_len = _round_up(max(len(row) for row in rows), padding.row_multiple)
    row_ids = torch.full((len(rows), row_len), pad_id, dtype=torch.long)
    row_mask = torch.zeros((len(rows), row_len), dtype=torch.bool)
    for r, row in enumerate(rows):
        row_ids[r, : len(row)] = torch.tensor(row, dtype=torch.long)
        row_mask[r, : len(row)] = True

    max_options = max(len(keys) for keys in option_keys)
    questions = len(option_keys)
    option_row = torch.zeros((questions, max_options), dtype=torch.long)
    option_position = torch.zeros((questions, max_options), dtype=torch.long)
    option_mask = torch.zeros((questions, max_options), dtype=torch.bool)
    option_token_ids = torch.zeros((questions, max_options), dtype=torch.long)
    target_tensor = torch.zeros((questions, max_options), dtype=torch.float32)
    has_target = torch.zeros(questions, dtype=torch.bool)
    for q in range(questions):
        count = len(option_rows[q])
        option_row[q, :count] = torch.tensor(option_rows[q])
        option_position[q, :count] = torch.tensor(option_pos[q])
        option_mask[q, :count] = True
        if option_tokens[q]:
            option_token_ids[q, :count] = torch.tensor(option_tokens[q])
        target = targets[q]
        if target is not None:
            target_tensor[q, :count] = torch.tensor(target, dtype=torch.float32)
            has_target[q] = True

    return DecisionBatch(
        state_ids=state_ids,
        state_mask=state_mask,
        row_ids=row_ids,
        row_mask=row_mask,
        row_owner=torch.tensor(row_owner, dtype=torch.long),
        query_row=torch.tensor(query_row, dtype=torch.long),
        query_pos=torch.tensor(query_pos, dtype=torch.long),
        option_row=option_row,
        option_pos=option_position,
        option_mask=option_mask,
        option_token_ids=option_token_ids,
        targets=target_tensor,
        has_target=has_target,
        label_index=torch.tensor(labels, dtype=torch.long),
        question_owner=torch.tensor(question_owner, dtype=torch.long),
        question_ids=question_ids,
        question_types=question_types,
        option_keys=option_keys,
    )
