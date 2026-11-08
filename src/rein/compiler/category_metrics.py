"""How a per-category detector is scored: its firing curve, the gate chosen on calibration, and its calibration."""

from typing import Any

import torch

from rein.compiler.firing import CURVE_THRESHOLDS, PRECISION_FLOOR
from rein.eval.metrics import PredictionRow, expected_calibration_error
from rein.train.calibrate import fit_temperature

DEFAULT_THRESHOLD = 0.5
GATE_LOWEST = 0.5
GATE_STEP = 0.01
GATE_STEPS = 50
GATE_GRID = tuple(round(GATE_LOWEST + step * GATE_STEP, 2) for step in range(GATE_STEPS))
YES_NO = ("no", "yes")
TIE_WEIGHT = 0.5


def _point(probs: torch.Tensor, labels: torch.Tensor, threshold: float) -> dict[str, Any]:
    fired = probs >= threshold
    hits = int((fired & (labels == 1)).sum())
    return {
        "threshold": threshold,
        "fired": int(fired.sum()),
        "precision": hits / max(1, int(fired.sum())),
        "recall": hits / max(1, int((labels == 1).sum())),
        "accuracy": float((fired.long() == labels).float().mean()) if len(labels) else 0.0,
    }


def best_point(points: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The point with the most recall among those that fire with precision of at least PRECISION_FLOOR."""
    eligible = [point for point in points if point["fired"] and point["precision"] >= PRECISION_FLOOR]
    return max(eligible, key=lambda point: (point["recall"], point["threshold"])) if eligible else None


def rates(probs: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    """Accuracy at 0.5, the firing curve, and the recall at precision 0.9 when the threshold is picked on these rows."""
    curve = [_point(probs, labels, threshold) for threshold in CURVE_THRESHOLDS]
    return {
        "n": len(labels),
        "positives": int(labels.sum()),
        "accuracy": _point(probs, labels, DEFAULT_THRESHOLD)["accuracy"],
        "curve": curve,
        "recall_at_precision_90": best_point(curve),
    }


def gate(probs: torch.Tensor, labels: torch.Tensor) -> float | None:
    """The threshold the checkpoint embeds: chosen on calibration rows, the most recall at precision 0.9 or more."""
    best = best_point([_point(probs, labels, threshold) for threshold in GATE_GRID])
    return None if best is None else float(best["threshold"])


def at_gate(probs: torch.Tensor, labels: torch.Tensor, threshold: float | None) -> dict[str, Any] | None:
    return None if threshold is None else _point(probs, labels, threshold)


def false_fires(probs: torch.Tensor, threshold: float | None) -> dict[str, float]:
    thresholds = [*CURVE_THRESHOLDS, *([threshold] if threshold is not None else [])]
    return {f"{value:.2f}": float((probs >= value).float().mean()) if len(probs) else 0.0 for value in thresholds}


def calibration_error(probs: torch.Tensor, labels: torch.Tensor) -> float | None:
    """Expected calibration error of the yes/no decision, as the evaluation computes it for every rule."""
    rows = [
        PredictionRow("", "", "noul", YES_NO, (1.0 - float(prob), float(prob)), int(label))
        for prob, label in zip(probs, labels, strict=True)
    ]
    return expected_calibration_error(rows) if rows else None


def temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return fit_temperature(logits, torch.ones_like(logits, dtype=torch.bool), labels)


def ranking_auc(probs: torch.Tensor, labels: torch.Tensor) -> float | None:
    """The chance that a random positive scores above a random negative (ties count half): threshold-free."""
    positives, negatives = probs[labels == 1], probs[labels == 0]
    if not len(positives) or not len(negatives):
        return None
    above = (positives[:, None] > negatives[None, :]).float()
    ties = (positives[:, None] == negatives[None, :]).float()
    return float((above + TIE_WEIGHT * ties).mean())
