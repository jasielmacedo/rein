import json
import random
from collections.abc import Iterator, Sequence
from pathlib import Path

from rein.model.encode import EncodedRequest, RequestEncoder
from rein.schema import Question, TrainingRecord

MIN_OPTIONS_TO_SUBSAMPLE = 3


def load_records(path: Path) -> list[TrainingRecord]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(TrainingRecord.model_validate(json.loads(line)))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: invalid record: {exc}") from exc
    return records


def drop_criteria(question: Question, rng: random.Random, rate: float) -> Question:
    """Hides option descriptions so the model also learns from bare option names."""
    if question.type == "score" or not isinstance(question.criteria, dict) or rng.random() >= rate:
        return question
    return question.model_copy(update={"criteria": dict.fromkeys(question.criteria)})


def subsample_options(question: Question, rng: random.Random, rate: float) -> Question:
    """Shows a random subset of a choice question's options in a random order, always keeping the label."""
    if question.type != "choice" or not isinstance(question.criteria, dict) or not isinstance(question.label, str):
        return question
    keys = list(question.criteria)
    if len(keys) < MIN_OPTIONS_TO_SUBSAMPLE or rng.random() >= rate:
        return question
    others = [key for key in keys if key != question.label]
    kept = [question.label, *rng.sample(others, rng.randint(1, len(others)))]
    rng.shuffle(kept)
    return question.model_copy(update={"criteria": {key: question.criteria[key] for key in kept}})


def augment(
    record: TrainingRecord, rng: random.Random, criteria_dropout: float, option_subsample: float = 0.0
) -> TrainingRecord:
    questions = {
        qid: subsample_options(drop_criteria(q, rng, criteria_dropout), rng, option_subsample)
        for qid, q in record.questions.items()
    }
    return record.model_copy(update={"questions": questions})


def record_cost(encoded: EncodedRequest) -> int:
    return len(encoded.state_ids) + sum(len(row) for question in encoded.questions for row in question.rows)


def token_batches(
    encoded: Sequence[EncodedRequest], batch_tokens: int, max_records: int, rng: random.Random
) -> list[list[int]]:
    """Groups records of similar length so padding stays small, then shuffles the batch order."""
    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i].state_ids))
    batches: list[list[int]] = []
    current: list[int] = []
    longest_state = 0
    rows_cost = 0
    for index in order:
        state_len = len(encoded[index].state_ids)
        row_cost = record_cost(encoded[index]) - state_len
        projected = max(longest_state, state_len) * (len(current) + 1) + rows_cost + row_cost
        if current and (projected > batch_tokens or len(current) >= max_records):
            batches.append(current)
            current, longest_state, rows_cost = [], 0, 0
        current.append(index)
        longest_state = max(longest_state, state_len)
        rows_cost += row_cost
    if current:
        batches.append(current)
    rng.shuffle(batches)
    return batches


def epoch_batches(
    records: Sequence[TrainingRecord],
    encoder: RequestEncoder,
    batch_tokens: int,
    max_records: int,
    criteria_dropout: float,
    rng: random.Random,
    option_subsample: float = 0.0,
) -> Iterator[list[EncodedRequest]]:
    encoded = [encoder.encode(augment(record, rng, criteria_dropout, option_subsample)) for record in records]
    for batch in token_batches(encoded, batch_tokens, max_records, rng):
        yield [encoded[i] for i in batch]
