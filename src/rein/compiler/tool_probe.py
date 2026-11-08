"""Where in Qwen3.5 the tool decision can be read so that it carries to agents never seen (DR-026).

Force ("the latest message needs one of my tools") and veto ("I have tools and none is needed") detectors are fitted
at several blocks on the synthetic records (reply openings, silences as negatives), their temperature and gate chosen on
the synthetic calibration split, and scored at reply openings on three sets: synthetic development records, every turn
of long end-to-end conversations with real agents, and the hand-written gold set. The ranking AUC tells a detector
that ranks well but sits at the wrong gate on new data from one that does not separate the cases at all.
"""

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from rein.compiler import category_metrics as m
from rein.compiler.category_probe import CALIBRATION, DEVELOPMENT, TRAIN, Row, fit_labelled, table_rows
from rein.compiler.table import StateTable
from rein.genview import constants as gc
from rein.genview.build import load_all_records

FORCE, VETO = "force", "veto"
ACTIONS = (FORCE, VETO)
OPENING_KINDS = (gc.REPLY_START,)
REPORT_FILE = "tool_probe.json"
SCORES_FILE = "tool_scores.json"
GOLD_SPLIT = "gold"


def tool_label(action: str, fired: int | None) -> int:
    """Force: a tool rule fires. Veto: the agent has a tool rule and it stays quiet."""
    return int(fired == 1) if action == FORCE else int(fired == 0)


def labelled(rows: Sequence[Row], action: str, split: str, quiet: Sequence[Row] = ()) -> tuple[list[Row], torch.Tensor]:
    picked = [row for row in rows if row["split"] == split]
    silences = [row for row in quiet if row["split"] == split]
    labels = [tool_label(action, row["labels"].get(gc.TOOL_CATEGORY)) for row in picked] + [0] * len(silences)
    return [*picked, *silences], torch.tensor(labels, dtype=torch.long)


def e2e_rows(table: StateTable, labels: Mapping[str, Mapping[str, Any]]) -> list[Row]:
    """Turn openings of the end-to-end sessions, labelled from the script: a tool is needed, or none is."""
    rows = []
    for row, entry in enumerate(table.index):
        label = labels.get(entry["record_id"])
        if label is None or entry["kind"] != gc.REPLY_START:
            continue
        fired = int(label["tool"] is not None)
        rows.append({"row": row, "split": "e2e", "record_id": entry["record_id"], "labels": {gc.TOOL_CATEGORY: fired}})
    return rows


def summary(probs: torch.Tensor, labels: torch.Tensor, gate: float | None) -> dict[str, Any]:
    rates = m.rates(probs, labels)
    return {
        "n": rates["n"],
        "positives": rates["positives"],
        "auc": m.ranking_auc(probs, labels),
        "at_gate": m.at_gate(probs, labels, gate),
        "recall_at_precision_90": rates["recall_at_precision_90"],
    }


COMBINATIONS = {
    "any_block": lambda stacked: stacked.max(0).values,
    "vote": lambda stacked: stacked.mean(0),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Tool detectors by block, scored on agents never seen.")
    parser.add_argument("--synthetic", type=Path, required=True, help="states of the synthetic records")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--categories", type=Path, required=True)
    parser.add_argument("--e2e", type=Path, required=True, help="states of the end-to-end turn openings")
    parser.add_argument("--e2e-labels", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True, help="states of the gold set")
    parser.add_argument("--gold-data", type=Path, required=True)
    parser.add_argument("--gold-categories", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    synthetic = StateTable.load(args.synthetic)
    e2e = StateTable.load(args.e2e, synthetic.layers)
    gold = StateTable.load(args.gold, synthetic.layers)
    rows, quiet = table_rows(
        synthetic, load_all_records(args.data), json.loads(args.categories.read_text()), OPENING_KINDS
    )
    turn_rows = e2e_rows(e2e, json.loads(args.e2e_labels.read_text(encoding="utf-8")))
    gold_rows, _ = table_rows(
        gold, load_all_records(args.gold_data), json.loads(args.gold_categories.read_text()), OPENING_KINDS
    )
    report: dict[str, Any] = {"layers": {}, "combined": {}}
    probabilities: dict[str, dict[str, list[torch.Tensor]]] = {action: {} for action in ACTIONS}
    labels_of: dict[str, dict[str, torch.Tensor]] = {action: {} for action in ACTIONS}
    for column, layer in enumerate(synthetic.layers):
        states = synthetic.states[:, column].to(device)
        sets = {
            CALIBRATION: (states, rows, CALIBRATION),
            DEVELOPMENT: (states, rows, DEVELOPMENT),
            "e2e": (e2e.states[:, column].to(device), turn_rows, "e2e"),
            GOLD_SPLIT: (gold.states[:, column].to(device), gold_rows, GOLD_SPLIT),
        }
        entry: dict[str, Any] = {}
        for action in ACTIONS:
            detector = fit_labelled(states, labelled(rows, action, TRAIN, quiet), labelled(rows, action, CALIBRATION))
            entry[action] = {"gate": detector.gate}
            for name, (set_states, set_rows, split) in sets.items():
                chosen, labels = labelled(set_rows, action, split)
                probs = detector.probabilities(set_states[[row["row"] for row in chosen]])
                probabilities[action].setdefault(name, []).append(probs)
                labels_of[action][name] = labels
                if name != CALIBRATION:
                    entry[action][name] = summary(probs, labels, detector.gate)
        report["layers"][str(layer)] = entry
        print(
            json.dumps(
                {"layer": layer, **{a: {k: v["auc"] for k, v in entry[a].items() if k != "gate"} for a in ACTIONS}}
            ),
            flush=True,
        )
    for action in ACTIONS:
        for rule, combine in COMBINATIONS.items():
            joined = {name: combine(torch.stack(values)) for name, values in probabilities[action].items()}
            gate = m.gate(joined[CALIBRATION], labels_of[action][CALIBRATION])
            report["combined"].setdefault(action, {})[rule] = {
                "gate": gate,
                **{
                    name: summary(probs, labels_of[action][name], gate)
                    for name, probs in joined.items()
                    if name != CALIBRATION
                },
            }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / REPORT_FILE).write_text(json.dumps(report, indent=2), encoding="utf-8")
    scores = {
        action: {
            name: {
                "records": [row["record_id"] for row in labelled(set_rows, action, split)[0]],
                "by_layer": [probs.tolist() for probs in probabilities[action][name]],
            }
            for name, (set_rows, split) in {"e2e": (turn_rows, "e2e"), GOLD_SPLIT: (gold_rows, GOLD_SPLIT)}.items()
        }
        for action in ACTIONS
    }
    (args.out / SCORES_FILE).write_text(json.dumps({"layers": list(synthetic.layers), **scores}), encoding="utf-8")


if __name__ == "__main__":
    main()
