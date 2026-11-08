import random
import statistics
import time
from collections.abc import Sequence

import torch

from rein.eval.metrics import PredictionRow
from rein.model.decider import Decider
from rein.schema import TrainingRecord
from rein.train.data import token_batches

PREDICT_SEED = 0


def predict_rows(
    decider: Decider, records: Sequence[TrainingRecord], batch_tokens: int, max_records: int
) -> list[PredictionRow]:
    encoded = [decider.encoder.encode(record) for record in records]
    rows: list[PredictionRow] = []
    for batch in token_batches(encoded, batch_tokens, max_records, random.Random(PREDICT_SEED)):
        output = decider.decide_encoded([encoded[i] for i in batch])
        questions = [(records[i], question) for i in batch for question in encoded[i].questions]
        for (record, question), probs in zip(questions, output.probs, strict=True):
            if question.label_index is None:
                continue
            rows.append(
                PredictionRow(
                    record_id=record.id,
                    question_id=question.question_id,
                    question_type=question.question_type,
                    option_keys=question.option_keys,
                    probs=tuple(probs[: len(question.option_keys)].tolist()),
                    label_index=question.label_index,
                    tags={"split": record.split, "source": record.source, **record.meta},
                )
            )
    return rows


PERCENTILE_95 = 0.95
MS_PER_S = 1000.0


def measure_latency(decider: Decider, records: list[TrainingRecord], count: int) -> dict[str, float]:
    """Single-request decision latency, encode included, as an application would call it."""
    sample = records[:count]
    durations: list[float] = []
    questions = 0
    for record in sample:
        if decider.device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        decider.decide(record)
        if decider.device.type == "cuda":
            torch.cuda.synchronize()
        durations.append((time.perf_counter() - started) * MS_PER_S)
        questions += len(record.questions)
    durations.sort()
    return {
        "requests": float(len(sample)),
        "ms_per_request_p50": statistics.median(durations),
        "ms_per_request_p95": durations[int(PERCENTILE_95 * (len(durations) - 1))],
        "ms_per_question_mean": sum(durations) / max(1, questions),
    }
