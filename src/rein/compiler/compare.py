import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rein.compiler.evaluate import DIAGNOSIS_FILE
from rein.compiler.run import REPORT_FILE

HEADLINE = ("all", "target_rule", "other_rules", "type:noul", "type:choice", "type:score")
TARGET_PREFIX = "target_"
MISSING = float("nan")
CELL_WIDTH = 22


def _row(entry: dict[str, Any] | None) -> dict[str, float]:
    if entry is None:
        return {}
    firing = entry.get("firing", {})
    best = entry.get("recall_at_precision_90") or {}
    return {
        "best_recall": best.get("recall", MISSING),
        "best_threshold": best.get("threshold", MISSING),
        "n": entry["n"],
        "accuracy": entry["accuracy"],
        "coverage": entry["coverage_at_gate"],
        "gate_accuracy": entry["accuracy_at_gate"],
        "precision": firing.get("precision", MISSING),
        "recall": firing.get("recall", MISSING),
    }


def comparison(runs: Sequence[Path], layer: int, position: str) -> dict[str, Any]:
    """The same slices of several compiler runs side by side, with each run's false fires at silence."""
    table: dict[str, Any] = {"runs": [str(run) for run in runs], "slices": {}, "false_fire": {}}
    for run in runs:
        diagnosis = json.loads((run / DIAGNOSIS_FILE).read_text(encoding="utf-8"))[str(layer)][position]
        report = json.loads((run / REPORT_FILE).read_text(encoding="utf-8"))
        table["false_fire"][str(run)] = report["splits"]["development"][str(layer)].get("false_fire_at_gate")
        for name, entry in diagnosis.items():
            table["slices"].setdefault(name, {})[str(run)] = _row(entry)
    return table


def _cell(row: dict[str, float]) -> str:
    return f"{row['accuracy']:6.3f} p{row['precision']:5.2f} r{row['recall']:5.2f}"


def _print_curves(table: dict[str, Any]) -> None:
    """Per slice: the most recall reachable while precision stays at or above 90%, and the threshold that gives it."""
    runs = table["runs"]
    print(
        f"{'slice (recall at precision >= 0.9, threshold)':52s} " + " ".join(f"{Path(r).name[-24:]:>24s}" for r in runs)
    )
    names = [name for name in HEADLINE if name in table["slices"]]
    names += sorted(name for name in table["slices"] if name.startswith(TARGET_PREFIX))
    for name in names:
        cells = []
        for run in runs:
            row = table["slices"][name].get(run, {})
            recall, threshold = row.get("best_recall", MISSING), row.get("best_threshold", MISSING)
            cells.append(f"{recall:5.2f} @ {threshold:4.2f}" if row else "")
        print(f"{name[:52]:52s} " + " ".join(f"{cell:>24s}" for cell in cells))


def _print(table: dict[str, Any]) -> None:
    runs = table["runs"]
    names = [Path(run).parent.name for run in runs]
    print(
        "false fire at silence: "
        + ", ".join(f"{n} {table['false_fire'][r]:.5f}" for n, r in zip(names, runs, strict=True))
    )
    ordered = [name for name in HEADLINE if name in table["slices"]]
    ordered += sorted(name for name in table["slices"] if name.startswith(TARGET_PREFIX))
    print(f"{'slice':40s} " + " ".join(f"{n[:18]:>30s}" for n in names))
    for name in ordered:
        cells = []
        for run in runs:
            row = table["slices"][name].get(run, {})
            cells.append(_cell(row) if row else " " * CELL_WIDTH)
        n = next((int(table["slices"][name][run]["n"]) for run in runs if table["slices"][name].get(run)), 0)
        print(f"{name[:34]:34s} {n:5d} " + " ".join(f"{cell:>30s}" for cell in cells))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the slices of several compiler runs at one layer.")
    parser.add_argument("--runs", type=Path, nargs="+", required=True, help="compiler output directories")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--position", default="nothink:reply_start")
    parser.add_argument("--curves", action="store_true", help="print recall at 90% precision per slice")
    args = parser.parse_args()
    table = comparison(args.runs, args.layer, args.position)
    (_print_curves if args.curves else _print)(table)


if __name__ == "__main__":
    main()
