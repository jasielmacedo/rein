import math
import random
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from rein.eval import constants as c


@dataclass(frozen=True)
class PredictionRow:
    record_id: str
    question_id: str
    question_type: str
    option_keys: tuple[str, ...]
    probs: tuple[float, ...]
    label_index: int
    tags: Mapping[str, Any] = field(default_factory=dict)

    @property
    def item(self) -> tuple[str, str]:
        return (self.record_id, self.question_id)

    @property
    def predicted_index(self) -> int:
        return max(range(len(self.probs)), key=self.probs.__getitem__)

    @property
    def confidence(self) -> float:
        return self.probs[self.predicted_index]

    @property
    def correct(self) -> bool:
        return self.predicted_index == self.label_index


Metric = Callable[[Sequence[PredictionRow]], float]


def accuracy(rows: Sequence[PredictionRow]) -> float:
    return sum(row.correct for row in rows) / len(rows) if rows else math.nan


def expected_calibration_error(rows: Sequence[PredictionRow], bins: int = c.ECE_BINS) -> float:
    if not rows:
        return math.nan
    confidence_sum = [0.0] * bins
    correct_sum = [0.0] * bins
    for row in rows:
        index = min(int(row.confidence * bins), bins - 1)
        confidence_sum[index] += row.confidence
        correct_sum[index] += float(row.correct)
    return sum(abs(correct_sum[b] - confidence_sum[b]) for b in range(bins)) / len(rows)


def brier_score(rows: Sequence[PredictionRow]) -> float:
    if not rows:
        return math.nan
    total = 0.0
    for row in rows:
        total += sum((p - (1.0 if k == row.label_index else 0.0)) ** 2 for k, p in enumerate(row.probs))
    return total / len(rows)


def negative_log_likelihood(rows: Sequence[PredictionRow]) -> float:
    if not rows:
        return math.nan
    return -sum(math.log(max(row.probs[row.label_index], c.NLL_FLOOR)) for row in rows) / len(rows)


def coverage_at_gate(rows: Sequence[PredictionRow], gate: float = c.DEFAULT_GATE) -> tuple[float, float]:
    if not rows:
        return math.nan, math.nan
    above = [row for row in rows if row.confidence >= gate]
    return len(above) / len(rows), accuracy(above)


METRICS: dict[str, Metric] = {
    "accuracy": accuracy,
    "ece": expected_calibration_error,
    "brier": brier_score,
    "nll": negative_log_likelihood,
}


def summarize(rows: Sequence[PredictionRow], bins: int = c.ECE_BINS, gate: float = c.DEFAULT_GATE) -> dict[str, float]:
    coverage, gated_accuracy = coverage_at_gate(rows, gate)
    return {
        "n": float(len(rows)),
        "accuracy": accuracy(rows),
        "ece": expected_calibration_error(rows, bins),
        "brier": brier_score(rows),
        "nll": negative_log_likelihood(rows),
        "coverage_at_gate": coverage,
        "accuracy_at_gate": gated_accuracy,
    }


@dataclass(frozen=True)
class Delta:
    estimate: float
    low: float
    high: float


def paired_bootstrap(
    rows_a: Sequence[PredictionRow],
    rows_b: Sequence[PredictionRow],
    metric: Metric,
    samples: int = c.BOOTSTRAP_SAMPLES,
    seed: int = c.BOOTSTRAP_SEED,
    confidence: float = c.CONFIDENCE_LEVEL,
) -> Delta:
    """metric(a) - metric(b) with a percentile interval that resamples whole records."""
    by_item_b = {row.item: row for row in rows_b}
    if set(by_item_b) != {row.item for row in rows_a}:
        raise ValueError("paired comparison needs the same items")
    groups: dict[str, list[tuple[PredictionRow, PredictionRow]]] = defaultdict(list)
    for row in rows_a:
        groups[row.record_id].append((row, by_item_b[row.item]))
    records = list(groups)
    estimate = metric(rows_a) - metric(rows_b)
    rng = random.Random(seed)
    deltas = []
    for _ in range(samples):
        picked = [pair for record in rng.choices(records, k=len(records)) for pair in groups[record]]
        deltas.append(metric([a for a, _ in picked]) - metric([b for _, b in picked]))
    deltas.sort()
    tail = (1.0 - confidence) / 2
    low = deltas[int(tail * (samples - 1))]
    high = deltas[int((1.0 - tail) * (samples - 1))]
    return Delta(estimate=estimate, low=min(low, estimate), high=max(high, estimate))
