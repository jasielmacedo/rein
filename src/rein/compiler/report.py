from collections.abc import Sequence
from typing import Any

import torch

from rein.compiler.firing import CURVE_THRESHOLDS
from rein.compiler.table import Item
from rein.eval import constants as ec
from rein.eval.metrics import PredictionRow, accuracy, summarize
from rein.eval.slices import suite_slices
from rein.neurons.bias import channel_stats
from rein.train.calibrate import fit_temperature

KEY_SLICES = (
    ec.SLICE_ALL,
    "unseen_target_rule",
    "words_same_intent_different",
    "watcher",
    "adversarial",
    "long_conversation",
    "holdout_domain_or_category",
    "category:npc_secret",
    "category:tool_trigger",
    "form:narration_attack",
)
SLICE_LANGUAGES = ("en", "pt-BR")


def layer_temperatures(logits: torch.Tensor, masks: torch.Tensor, items: Sequence[Item]) -> list[float]:
    labels = torch.tensor([item.label for item in items])
    return [fit_temperature(logits[:, layer], masks, labels) for layer in range(logits.shape[1])]


def prediction_rows(
    logits: torch.Tensor, masks: torch.Tensor, items: Sequence[Item], temperature: float
) -> list[PredictionRow]:
    probs = torch.softmax((logits / temperature).masked_fill(~masks, torch.finfo(logits.dtype).min), dim=-1)
    return [
        PredictionRow(
            record_id=item.record_id,
            question_id=item.question_id,
            question_type=item.question_type,
            option_keys=item.option_keys,
            probs=tuple(probs[position, : len(item.option_keys)].tolist()),
            label_index=item.label,
            tags=item.tags,
        )
        for position, item in enumerate(items)
        if item.label >= 0
    ]


def _position_groups(items: Sequence[Item]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for position, item in enumerate(items):
        groups.setdefault(f"{item.mode}:{item.kind}", []).append(position)
    return groups


def _slices(rows: Sequence[PredictionRow]) -> dict[str, dict[str, float]]:
    categories = {str(row.tags.get("category")) for row in rows}
    slices = suite_slices(SLICE_LANGUAGES, categories=categories, forms=("narration_attack",))
    report: dict[str, dict[str, float]] = {}
    for name in KEY_SLICES:
        selected = [row for row in rows if name in slices and slices[name](row)]
        if selected:
            report[name] = {"n": float(len(selected)), "accuracy": accuracy(selected)}
    return report


def false_fire_rate(
    logits: torch.Tensor, masks: torch.Tensor, items: Sequence[Item], temperature: float, gate: float
) -> float:
    """Share of silence items where some option other than the null one would pass the gate."""
    if not items:
        return 0.0
    probs = torch.softmax((logits / temperature).masked_fill(~masks, torch.finfo(logits.dtype).min), dim=-1)
    null = torch.tensor([item.null for item in items])
    probs = probs.scatter(1, null[:, None], 0.0)
    return float((probs.max(dim=-1).values >= gate).float().mean())


def layer_report(
    layers: Sequence[int],
    logits: torch.Tensor,
    masks: torch.Tensor,
    items: Sequence[Item],
    temperatures: Sequence[float],
    silence: tuple[torch.Tensor, torch.Tensor, Sequence[Item]] | None = None,
    states: torch.Tensor | None = None,
    channels: torch.Tensor | None = None,
    gate: float = ec.DEFAULT_GATE,
) -> dict[str, Any]:
    """Per layer and per position type (mode:kind): the Rein summary, its key slices, the false fires at silence
    positions, and how constant the bias channel stays on these states."""
    report: dict[str, Any] = {}
    groups = _position_groups(items)
    for position, layer in enumerate(layers):
        entry: dict[str, Any] = {"temperature": temperatures[position], "positions": {}}
        for name, indices in sorted(groups.items()):
            picked = [items[index] for index in indices]
            rows = prediction_rows(logits[indices, position], masks[indices], picked, temperatures[position])
            entry["positions"][name] = {"summary": summarize(rows), "slices": _slices(rows)}
        if silence is not None:
            silence_logits, silence_masks, silence_items = silence
            entry["false_fire_at_gate"] = false_fire_rate(
                silence_logits[:, position], silence_masks, silence_items, temperatures[position], gate
            )
            entry["false_fire_by_threshold"] = {
                str(threshold): false_fire_rate(
                    silence_logits[:, position], silence_masks, silence_items, temperatures[position], threshold
                )
                for threshold in CURVE_THRESHOLDS
            }
        if states is not None and channels is not None:
            stats = channel_stats(channels[position], states[:, position].float())
            entry["bias_channel"] = {"mean": stats.mean, "std": stats.std, "variation": stats.variation}
        report[str(layer)] = entry
    return report
