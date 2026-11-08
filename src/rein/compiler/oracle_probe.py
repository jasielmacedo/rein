"""Oracle test, part 2: one detector per rule, trained on that rule's own examples at the compiler's reading point.

If these detectors reach about 0.95 on held-out cases of their rule, the reading point carries the decision and the
loss is in the compiler; if they stay near the compiler, the reading point is the limit.
"""

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from rein.compiler.firing import firing_curve, recall_at_precision
from rein.compiler.readings import load_readings
from rein.compiler.report import prediction_rows
from rein.compiler.run import CHANNELS_FILE, CONFIG_FILE, build_compiler, load_compiler_weights
from rein.compiler.table import Item, StateTable, table_items
from rein.compiler.train import predict_logits
from rein.eval.metrics import PredictionRow, accuracy
from rein.genview.build import load_all_records

TEST_SHARE = 1 / 3
L2_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
FOLDS = 4
PROBABILITY_FLOOR = 1e-9
LBFGS_STEPS = 200
REPORT_FILE = "oracle.json"
MODE, KIND = "nothink", "reply_start"


def is_held_out(record_id: str) -> bool:
    digest = int(hashlib.sha256(record_id.encode("utf-8")).hexdigest(), 16)
    return digest % 1000 < TEST_SHARE * 1000


def _with_bias(features: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    column = torch.ones(len(features), 1, device=features.device) if bias is None else bias.reshape(-1, 1)
    return torch.cat([features, column.to(features)], dim=1)


def fit_probe(
    features: torch.Tensor, labels: torch.Tensor, classes: int, l2: float, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """Multinomial logistic regression with an intercept, as (classes, hidden + 1). The intercept multiplies `bias`
    when given (a bias channel, about 1 on every row, so the read stays a neuron's), else a column of ones."""
    inputs = _with_bias(features, bias)
    weights = torch.zeros(classes, inputs.shape[1], device=inputs.device, requires_grad=True)
    optimizer = torch.optim.LBFGS([weights], max_iter=LBFGS_STEPS, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(inputs @ weights.T, labels) + l2 * weights[:, :-1].pow(2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return weights.detach()


def probe_probabilities(
    weights: torch.Tensor, features: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    return torch.softmax(_with_bias(features, bias) @ weights.T, dim=-1)


def choose_l2(features: torch.Tensor, labels: torch.Tensor, classes: int, bias: torch.Tensor | None = None) -> float:
    """The L2 strength with the best cross-validated log-likelihood on the training examples."""
    folds = torch.arange(len(features), device=features.device) % FOLDS
    scores = []
    for l2 in L2_GRID:
        total = 0.0
        for fold in range(FOLDS):
            train, test = folds != fold, folds == fold
            if train.sum() == 0 or test.sum() == 0:
                continue
            weights = fit_probe(features[train], labels[train], classes, l2, None if bias is None else bias[train])
            probs = probe_probabilities(weights, features[test], None if bias is None else bias[test])
            picked = probs.gather(1, labels[test][:, None]).clamp(min=PROBABILITY_FLOOR)
            total += float(torch.log(picked).sum())
        scores.append(total)
    return L2_GRID[max(range(len(L2_GRID)), key=scores.__getitem__)]


def _summary(rows: Sequence[PredictionRow]) -> dict[str, Any]:
    curve = firing_curve(rows)
    return {"n": len(rows), "accuracy": accuracy(rows), "recall_at_precision_90": recall_at_precision(curve)}


def oracle_rows(items: Sequence[Item], states: torch.Tensor, column: int) -> list[PredictionRow]:
    """Held-out rows of every rule, each scored by a detector trained on that rule's other examples only."""
    rows: list[PredictionRow] = []
    by_rule: dict[int, list[Item]] = {}
    for item in items:
        by_rule.setdefault(item.reading, []).append(item)
    for rule_items in by_rule.values():
        train = [item for item in rule_items if not is_held_out(item.record_id)]
        test = [item for item in rule_items if is_held_out(item.record_id)]
        classes = len(rule_items[0].option_keys)
        if not train or not test or len({item.label for item in train}) < 2:
            continue
        features = states[[item.row for item in train], column].float()
        mean, scale = features.mean(0), features.std(0).clamp(min=1e-6)
        labels = torch.tensor([item.label for item in train])
        normalized = (features - mean) / scale
        weights = fit_probe(normalized, labels, classes, choose_l2(normalized, labels, classes))
        probs = probe_probabilities(weights, (states[[item.row for item in test], column].float() - mean) / scale)
        rows += [
            PredictionRow(
                item.record_id,
                item.question_id,
                item.question_type,
                item.option_keys,
                tuple(probs[index].tolist()),
                item.label,
                item.tags,
            )
            for index, item in enumerate(test)
        ]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-rule oracle detectors against the compiler on the same cases.")
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--readings", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    table = StateTable.load(args.states, args.layers)
    reading_index, readings = load_readings(args.readings)
    decisions, _ = table_items(table, load_all_records(args.data), reading_index)
    items = [
        item
        for item in decisions
        if item.mode == MODE
        and item.kind == KIND
        and item.tags.get("target_rule") == item.question_id
        and item.label >= 0
    ]
    report: dict[str, Any] = {"rules": len({item.reading for item in items}), "items": len(items), "layers": {}}
    for column, layer in enumerate(table.layers):
        rows = oracle_rows(items, table.states, column)
        by_category: dict[str, list[PredictionRow]] = {}
        for row in rows:
            by_category.setdefault(str(row.tags.get("category")), []).append(row)
        report["layers"][str(layer)] = {
            "oracle": _summary(rows),
            "oracle_by_category": {name: _summary(picked) for name, picked in sorted(by_category.items())},
        }
    config = json.loads((args.compiler / CONFIG_FILE).read_text(encoding="utf-8"))
    compiler = build_compiler(config)
    load_compiler_weights(compiler, args.compiler)
    compiler.eval()
    channels = load_file(str(args.compiler / CHANNELS_FILE))["channels"]
    compiled_table = StateTable.load(args.states, config["layers"])
    held_out = [item for item in items if is_held_out(item.record_id)]
    logits, masks = predict_logits(compiler, readings, compiled_table.states, channels, held_out)
    output_layers = config.get("output_layers", config["layers"])
    for column, layer in enumerate(output_layers):
        rows = prediction_rows(logits[:, column], masks, held_out, config["temperatures"][column])
        by_category = {}
        for row in rows:
            by_category.setdefault(str(row.tags.get("category")), []).append(row)
        report["layers"].setdefault(str(layer), {})["compiler"] = _summary(rows)
        report["layers"][str(layer)]["compiler_by_category"] = {
            name: _summary(picked) for name, picked in sorted(by_category.items())
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"written": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
