"""A run's decisions through llama-server on its merged GGUF against the same run in PyTorch: the same records, the
same layout, the options' probabilities compared question by question."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from rein.control.backend import DeciderBackend
from rein.control.plan import DecisionBackend
from rein.control.server_decider import ServerDecisionBackend
from rein.model.loader import load_decider
from rein.schema import DecisionRequest
from rein.train.data import load_records

DEFAULT_RECORDS = 20
DIGITS = 4


def compare(
    reference: DecisionBackend, candidate: DecisionBackend, requests: Sequence[DecisionRequest]
) -> list[dict[str, Any]]:
    """One row per question: both backends' probabilities over the options, their largest gap, and whether they pick
    the same option."""
    rows: list[dict[str, Any]] = []
    for index, request in enumerate(requests):
        expected, actual = reference.decide(request), candidate.decide(request)
        for question_id, options in expected.items():
            keys = list(options)
            others = actual[question_id]
            mine = [options[key] for key in keys]
            theirs = [others[key] for key in keys]
            rows.append(
                {
                    "request": index,
                    "question": question_id,
                    "reference": [round(value, DIGITS) for value in mine],
                    "candidate": [round(value, DIGITS) for value in theirs],
                    "max_abs_diff": round(max(abs(a - b) for a, b in zip(mine, theirs, strict=True)), DIGITS),
                    "same_argmax": max(keys, key=options.__getitem__) == max(keys, key=others.__getitem__),
                }
            )
    return rows


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    diffs = [row["max_abs_diff"] for row in rows]
    return {
        "questions": len(rows),
        "same_argmax": sum(row["same_argmax"] for row in rows),
        "mean_max_abs_diff": round(sum(diffs) / len(diffs), DIGITS),
        "worst_max_abs_diff": max(diffs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare a run's GGUF decisions through llama-server with PyTorch.")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="records in the training format")
    parser.add_argument("--url", required=True, help="llama-server serving the run's merged GGUF")
    parser.add_argument("--records", type=int, default=DEFAULT_RECORDS)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    reference = DeciderBackend(load_decider(args.run, torch.device(args.device)))
    candidate = ServerDecisionBackend(args.url, args.run, reference.decider.encoder)
    rows = compare(reference, candidate, load_records(args.data)[: args.records])
    summary = summarize(rows)
    args.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
