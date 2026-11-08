from collections.abc import Sequence
from dataclasses import dataclass

from rein.compiler.model import null_option_index
from rein.eval.metrics import PredictionRow

CURVE_THRESHOLDS = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)
PRECISION_FLOOR = 0.9


@dataclass(frozen=True)
class Fire:
    fired: bool
    correct: bool
    positive: bool


def fire(row: PredictionRow, threshold: float) -> Fire | None:
    """Whether the rule fires on this row: its best option other than the null one passes the threshold.

    Rules without a null option (score rules) never fire on their own and are left out.
    """
    null = null_option_index(row.option_keys)
    if null < 0:
        return None
    best = max((index for index in range(len(row.probs)) if index != null), key=row.probs.__getitem__)
    fired = row.probs[best] >= threshold
    return Fire(fired=fired, correct=fired and best == row.label_index, positive=row.label_index != null)


def firing(rows: Sequence[PredictionRow], threshold: float) -> dict[str, float]:
    """How often the rule fires at `threshold`, and how often that is right (precision) or found (recall)."""
    fires = [outcome for outcome in (fire(row, threshold) for row in rows) if outcome is not None]
    fired = sum(outcome.fired for outcome in fires)
    positives = sum(outcome.positive for outcome in fires)
    hits = sum(outcome.correct for outcome in fires)
    return {
        "threshold": threshold,
        "n": float(len(fires)),
        "positives": float(positives),
        "fired": float(fired),
        "precision": hits / max(1, fired),
        "recall": hits / max(1, positives),
    }


def firing_curve(
    rows: Sequence[PredictionRow], thresholds: Sequence[float] = CURVE_THRESHOLDS
) -> list[dict[str, float]]:
    return [firing(rows, threshold) for threshold in thresholds]


def recall_at_precision(curve: Sequence[dict[str, float]], floor: float = PRECISION_FLOOR) -> dict[str, float] | None:
    """The point of the curve with the most recall whose precision stays at or above `floor`."""
    eligible = [point for point in curve if point["fired"] > 0 and point["precision"] >= floor]
    return max(eligible, key=lambda point: (point["recall"], -point["threshold"])) if eligible else None
