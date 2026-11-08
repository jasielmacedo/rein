import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rein.datagen.taxonomy import load_taxonomy
from rein.env import load_env, optional
from rein.eval.metrics import PredictionRow
from rein.eval.report import Report, build_report
from rein.eval.slices import suite_slices
from rein.eval.systemone import DEFAULT_CONCURRENCY, score_remote
from rein.schema import TrainingRecord
from rein.train.data import load_records

REIN_MODEL = "rein"
DELTA_METRICS = ("accuracy", "ece", "brier")
EVAL_BATCH_TOKENS = 32768
EVAL_MAX_RECORDS = 32


def save_rows(path: Path, rows: list[PredictionRow]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def load_rows(path: Path) -> list[PredictionRow]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            data = json.loads(line)
            data["option_keys"] = tuple(data["option_keys"])
            data["probs"] = tuple(data["probs"])
            rows.append(PredictionRow(**data))
    return rows


def _pairs(values: list[str]) -> dict[str, str]:
    pairs = {}
    for value in values:
        name, _, target = value.partition("=")
        if not target:
            raise SystemExit(f"expected name=value, got {value!r}")
        pairs[name] = target
    return pairs


def deltas(report: Report, baselines: list[str]) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for baseline in baselines:
        for slice_name in report.slices:
            if not report.slice_rows(slice_name, REIN_MODEL):
                continue
            for metric in DELTA_METRICS:
                delta = report.paired_delta(slice_name, REIN_MODEL, baseline, metric=metric)
                out.setdefault(baseline, {}).setdefault(slice_name, {})[metric] = round(delta.estimate, 4)
                out[baseline][slice_name][f"{metric}_ci"] = [round(delta.low, 4), round(delta.high, 4)]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Score Rein and baselines on the same items and report the suite.")
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run", type=Path, help="Rein run directory to score")
    parser.add_argument("--baseline-url", nargs="*", default=[], help="name=url of a System One server")
    parser.add_argument(
        "--baseline-concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help="requests in flight per baseline server (1 for single-threaded servers such as TinyJev)",
    )
    parser.add_argument("--rows", nargs="*", default=[], help="name=path of saved prediction rows")
    parser.add_argument("--latency-records", type=int, default=0, help="time this many single requests")
    args = parser.parse_args()
    records: list[TrainingRecord] = [record for path in args.data for record in load_records(path)]
    args.out.mkdir(parents=True, exist_ok=True)

    latency: dict[str, float] = {}
    predictions: dict[str, list[PredictionRow]] = {
        name: load_rows(Path(path)) for name, path in _pairs(args.rows).items()
    }
    if args.run is not None:
        import torch

        from rein.eval.predict import measure_latency, predict_rows
        from rein.model.loader import load_decider

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        decider = load_decider(args.run, device, optional(load_env(), "HF_TOKEN"))
        predictions[REIN_MODEL] = predict_rows(decider, records, EVAL_BATCH_TOKENS, EVAL_MAX_RECORDS)
        if args.latency_records:
            latency = measure_latency(decider, records, args.latency_records)
    for name, url in _pairs(args.baseline_url).items():
        predictions[name] = asyncio.run(score_remote(url, records, args.baseline_concurrency))
    for name, rows in predictions.items():
        save_rows(args.out / f"rows_{name}.jsonl", rows)

    common = set.intersection(*({row.item for row in rows} for rows in predictions.values()))
    aligned = {name: [row for row in rows if row.item in common] for name, rows in predictions.items()}

    def tag_values(key: str) -> list[str]:
        return sorted({str(row.tags.get(key)) for rows in aligned.values() for row in rows if row.tags.get(key)})

    taxonomy = load_taxonomy()
    slices = suite_slices(
        tag_values("lang_conv"),
        tag_values("category"),
        tag_values("case_form"),
        taxonomy.holdout.domains,
        taxonomy.holdout.categories,
    )
    report = build_report(aligned, slices)
    (args.out / "report.md").write_text(report.to_markdown(), encoding="utf-8")
    summary = {
        "items": len(common),
        "dropped_items": {name: len(rows) - len(aligned[name]) for name, rows in predictions.items()},
        "latency": latency,
        "cells": {f"{s}|{m}": dict(v) for (s, m), v in report.cells.items()},
        "deltas": deltas(report, [name for name in aligned if name != REIN_MODEL]) if REIN_MODEL in aligned else {},
    }
    (args.out / "report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(report.to_markdown())


if __name__ == "__main__":
    main()
