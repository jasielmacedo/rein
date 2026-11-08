"""Generic detectors, one per rule category: the test of fixed neurons that serve any agent (DR-026).

Each detector reads only Qwen's state where the reply opens, in both thinking modes, with the agent's rules in the
context, and learns whether a rule of its category fires there, across every agent of the training split. It never
receives the rule text as an input. A record without a rule of the category counts as "does not fire". Agents never
cross splits, so the development split measures agents and rules the detector never saw. The temperature and the gate
a checkpoint would embed are chosen on the calibration split.
"""

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from rein.compiler import category_metrics as m
from rein.compiler.constants import NO_NULL
from rein.compiler.model import null_option_index
from rein.compiler.oracle_probe import choose_l2, fit_probe
from rein.compiler.table import StateTable
from rein.genview import constants as gc
from rein.genview.build import NO_THINKING, load_all_records
from rein.genview.reasoning import ANSWER_START, REASONING
from rein.genview.scenarios import ScenarioIndex
from rein.schema import TrainingRecord

TRAIN, CALIBRATION, DEVELOPMENT = "train", "calibration", "development"
DECISION_KINDS = (gc.REPLY_START, ANSWER_START)
SCORED_KINDS = (*DECISION_KINDS, REASONING)
SILENCES_PER_DECISION = 1
SCALE_FLOOR = 1e-6
REPORT_FILE = "category_probe.json"
PROBABILITIES_FILE = "category_probabilities.json"
COMPILER_POSITION = f"{NO_THINKING}:{gc.REPLY_START}"
NEUTRAL_FORM = "neutral"
MAIN_LANGUAGES = ("en", "pt-BR")
MIXED_MARK = "+"
UNKNOWN_LANGUAGE, OTHER_LANGUAGE, MIXED_LANGUAGE = "unknown", "other", "mixed"
DOOR_SEPARATOR = "="
PLAY_THRESHOLD = 0.5

Row = dict[str, Any]


def fires(record: TrainingRecord, question_id: str) -> bool | None:
    """Whether one rule fires on the record, or None when it has no label or no "does not fire" option (scores)."""
    question = record.questions[question_id]
    index = question.label_index()
    null = null_option_index(question.option_keys())
    if index is None or null == NO_NULL:
        return None
    return index != null


def category_labels(record: TrainingRecord, categories: Mapping[str, str]) -> dict[str, int]:
    """Per category of the taxonomy: 1 if some rule of that category fires on the record."""
    labels: dict[str, int] = {}
    for question_id, category in categories.items():
        if question_id in record.questions and (fired := fires(record, question_id)) is not None:
            labels[category] = max(labels.get(category, 0), int(fired))
    return labels


def record_categories(records: Mapping[str, TrainingRecord], index: ScenarioIndex) -> dict[str, dict[str, str]]:
    """Each record's rules with their categories, as the scenario that produced them declared."""
    found = {}
    for record_id, record in records.items():
        info = index.lookup(record)
        if info is not None:
            found[record_id] = {qid: info.categories[qid] for qid in record.questions if qid in info.categories}
    return found


def compiler_fire_probabilities(
    predictions: Path, categories: Mapping[str, Mapping[str, str]]
) -> dict[str, dict[str, float]]:
    """Per record and category, the per-rule compiler's chance that some rule of the category fires (its top rule)."""
    found: dict[str, dict[str, float]] = {}
    with predictions.open(encoding="utf-8") as handle:
        for line in handle:
            entry = json.loads(line) if line.strip() else None
            if entry is None or entry["position"] != COMPILER_POSITION:
                continue
            category = categories.get(entry["record_id"], {}).get(entry["question_id"])
            null = null_option_index(entry["option_keys"])
            if category is None or null == NO_NULL:
                continue
            chances = found.setdefault(entry["record_id"], {})
            chances[category] = max(chances.get(category, 0.0), 1.0 - entry["probs"][null])
    return found


def table_rows(
    table: StateTable,
    records: Mapping[str, TrainingRecord],
    categories: Mapping[str, Mapping[str, str]],
    kinds: Sequence[str] = SCORED_KINDS,
) -> tuple[list[Row], list[Row]]:
    """Scored rows of `kinds` (with the record's labels per category) and silence rows."""
    rows: list[Row] = []
    quiet: list[Row] = []
    for row, entry in enumerate(table.index):
        record = records.get(entry["record_id"])
        if record is None or record.id not in categories:
            continue
        item = {
            "row": row,
            "split": record.split,
            "record_id": record.id,
            "position": f"{entry['mode']}:{entry['kind']}",
        }
        if entry["kind"] in kinds:
            rows.append({**item, "kind": entry["kind"], "labels": category_labels(record, categories[record.id])})
        elif entry["kind"] == gc.SILENCE:
            quiet.append(item)
    return rows, quiet


def by_position(rows: Sequence[Row], split: str) -> dict[str, list[Row]]:
    grouped: dict[str, list[Row]] = {}
    for row in rows:
        if row["split"] == split:
            grouped.setdefault(row["position"], []).append(row)
    return grouped


def _labels(rows: Sequence[Row], category: str) -> torch.Tensor:
    return torch.tensor([row["labels"].get(category, 0) for row in rows], dtype=torch.long)


def _states(states: torch.Tensor, rows: Sequence[Row]) -> torch.Tensor:
    return states[[row["row"] for row in rows]]


@dataclass(frozen=True)
class Detector:
    """A linear read of the standardized state, with its temperature and its gate. With a bias channel `c` (where
    `c . x` is about 1), the centering and the intercept ride on `c . x`, so the read is exactly a neuron's."""

    weights: torch.Tensor
    mean: torch.Tensor
    scale: torch.Tensor
    temperature: float = 1.0
    gate: float | None = None
    channel: torch.Tensor | None = None

    def inputs(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """The standardized state and the column the intercept multiplies (None: a column of ones)."""
        x = states.float()
        if self.channel is None:
            return (x - self.mean) / self.scale, None
        level = x @ self.channel
        return x / self.scale - level[:, None] * (self.mean / self.scale), level

    def logits(self, states: torch.Tensor) -> torch.Tensor:
        features, bias = self.inputs(states)
        column = torch.ones(len(features), device=features.device) if bias is None else bias
        return torch.cat([features, column[:, None]], dim=1) @ self.weights.T

    def probabilities(self, states: torch.Tensor) -> torch.Tensor:
        if not len(states):
            return torch.zeros(0)
        return torch.softmax(self.logits(states) / self.temperature, dim=-1)[:, 1].cpu()

    def neuron_form(self) -> tuple[torch.Tensor, float]:
        """The yes-minus-no read as `direction . x + offset * (c . x)`, the form `threshold_gate` takes."""
        if self.channel is None:
            raise ValueError("only a detector fitted on the bias channel is a neuron's read")
        difference = self.weights[1] - self.weights[0]
        direction = difference[:-1] / self.scale
        return direction, float(difference[-1] - direction @ self.mean)


def fit_labelled(
    states: torch.Tensor,
    train: tuple[Sequence[Row], torch.Tensor],
    calibration: tuple[Sequence[Row], torch.Tensor],
    channel: torch.Tensor | None = None,
) -> Detector:
    """Fits on the training rows, then picks the temperature and the gate on the calibration rows."""
    rows, labels = train
    x = _states(states, rows).float()
    labels = labels.to(x.device)
    shell = Detector(torch.zeros(0), x.mean(0), x.std(0).clamp(min=SCALE_FLOOR), channel=channel)
    features, bias = shell.inputs(x)
    raw = replace(shell, weights=fit_probe(features, labels, 2, choose_l2(features, labels, 2, bias), bias))
    calibration_rows, calibration_labels = calibration
    picked = _states(states, calibration_rows)
    tempered = replace(raw, temperature=m.temperature(raw.logits(picked).cpu(), calibration_labels))
    return replace(tempered, gate=m.gate(tempered.probabilities(picked), calibration_labels))


def fit_detector(states: torch.Tensor, rows: Sequence[Row], quiet: Sequence[Row], category: str) -> Detector:
    """Fits on the training decisions plus as many training silences, then picks the temperature and the gate on the
    calibration decisions."""
    train = [row for row in rows if row["split"] == TRAIN and row["kind"] in DECISION_KINDS]
    quiet_train = [row for row in quiet if row["split"] == TRAIN][: len(train) * SILENCES_PER_DECISION]
    labels = torch.cat([_labels(train, category), torch.zeros(len(quiet_train), dtype=torch.long)])
    calibration = [row for row in rows if row["split"] == CALIBRATION and row["kind"] in DECISION_KINDS]
    return fit_labelled(states, ([*train, *quiet_train], labels), (calibration, _labels(calibration, category)))


def _scored(probs: torch.Tensor, labels: torch.Tensor, threshold: float | None) -> dict[str, Any]:
    return {
        **m.rates(probs, labels),
        "at_gate": m.at_gate(probs, labels, threshold),
        "ece": m.calibration_error(probs, labels),
    }


def position_report(
    detector: Detector, states: torch.Tensor, rows: Sequence[Row], category: str
) -> tuple[dict[str, Any], torch.Tensor]:
    """Every record at one position, and the records that carry a rule of the category."""
    probs, labels = detector.probabilities(_states(states, rows)), _labels(rows, category)
    present = torch.tensor([category in row["labels"] for row in rows], dtype=torch.bool)
    report = {
        "all": _scored(probs, labels, detector.gate),
        "with_the_category": _scored(probs[present], labels[present], detector.gate),
    }
    return report, probs


def score_category(
    detector: Detector,
    states: torch.Tensor,
    rows: Sequence[Row],
    quiet: Sequence[Row],
    category: str,
    doors: Mapping[str, tuple[torch.Tensor, list[Row]]],
) -> tuple[dict[str, Any], dict[str, list[float]]]:
    """The detector on the development split, per position, on its silences, and on the other doors' states."""
    report: dict[str, Any] = {"temperature": detector.temperature, "gate": detector.gate, "positions": {}, "doors": {}}
    probabilities: dict[str, list[float]] = {}
    for position, picked in by_position(rows, DEVELOPMENT).items():
        report["positions"][position], probs = position_report(detector, states, picked, category)
        probabilities[position] = probs.tolist()
    silences = detector.probabilities(_states(states, [row for row in quiet if row["split"] == DEVELOPMENT]))
    report["silence_false_fire"] = m.false_fires(silences, detector.gate)
    for name, (door_states, door_rows) in doors.items():
        report["doors"][name] = {
            position: position_report(detector, door_states, picked, category)[0]
            for position, picked in by_position(door_rows, DEVELOPMENT).items()
        }
    return report, probabilities


def compiler_category(rows: Sequence[Row], category: str, chances: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    """The per-rule compiler on the same decisions; a record without a rule of the category cannot fire."""
    dev = by_position(rows, DEVELOPMENT).get(COMPILER_POSITION, [])
    probs = torch.tensor([chances.get(row["record_id"], {}).get(category, 0.0) for row in dev])
    labels = _labels(dev, category)
    present = torch.tensor([category in row["labels"] for row in dev], dtype=torch.bool)
    return {"all": m.rates(probs, labels), "with_the_category": m.rates(probs[present], labels[present])}


def quiet_records(
    dev: Mapping[str, Sequence[Row]],
    probabilities: Mapping[str, Mapping[str, Sequence[float]]],
    gates: Mapping[str, float | None],
    records: Mapping[str, TrainingRecord],
) -> dict[str, Any]:
    """Records where no rule fires at all, and the plain conversations among them: how often any detector fires."""

    def any_fire(position: str, indices: Sequence[int]) -> float:
        fired = [
            any(gate is not None and probabilities[category][position][i] >= gate for category, gate in gates.items())
            for i in indices
        ]
        return sum(fired) / max(1, len(fired))

    report = {}
    for position, picked in dev.items():
        quiet = [i for i, row in enumerate(picked) if not any(row["labels"].values())]
        plain = [i for i in quiet if (records[picked[i]["record_id"]].meta or {}).get("case_form") == NEUTRAL_FORM]
        report[position] = {
            "records": len(quiet),
            "any_fire": any_fire(position, quiet),
            "neutral_records": len(plain),
            "neutral_any_fire": any_fire(position, plain),
        }
    return report


def language_group(code: str | None) -> str:
    if not code:
        return UNKNOWN_LANGUAGE
    if MIXED_MARK in code:
        return MIXED_LANGUAGE
    return code if code in MAIN_LANGUAGES else OTHER_LANGUAGE


def _pooled(pairs: Sequence[tuple[float, int, float]]) -> dict[str, Any]:
    fired = [prob >= gate for prob, _, gate in pairs]
    labels = [label for _, label, _ in pairs]
    hits = sum(f and bool(label) for f, label in zip(fired, labels, strict=True))
    return {
        "n": len(pairs),
        "positives": sum(labels),
        "precision": hits / max(1, sum(fired)),
        "recall": hits / max(1, sum(labels)),
        "accuracy": sum(f == bool(label) for f, label in zip(fired, labels, strict=True)) / max(1, len(pairs)),
    }


def languages(
    dev: Mapping[str, Sequence[Row]],
    probabilities: Mapping[str, Mapping[str, Sequence[float]]],
    gates: Mapping[str, float | None],
    records: Mapping[str, TrainingRecord],
) -> dict[str, Any]:
    """Every category pooled at its gate, by the language of the rules and of the conversation."""
    report = {}
    for position, picked in dev.items():
        groups: dict[str, list[tuple[float, int, float]]] = {}
        for i, row in enumerate(picked):
            meta = records[row["record_id"]].meta or {}
            keys = (
                f"rules:{language_group(meta.get('lang_rule'))}",
                f"conversation:{language_group(meta.get('lang_conv'))}",
            )
            for category, label in row["labels"].items():
                gate = gates.get(category)
                if gate is None:
                    continue
                for key in keys:
                    groups.setdefault(key, []).append((probabilities[category][position][i], label, gate))
        report[position] = {key: _pooled(pairs) for key, pairs in sorted(groups.items())}
    return report


def write_categories(data: Path, synthetic: Path, out: Path) -> None:
    mapping = record_categories(load_all_records(data), ScenarioIndex.from_directory(synthetic))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(mapping), encoding="utf-8")
    print(json.dumps({"records": len(mapping), "written": str(out)}), flush=True)


def _summary(category: str, entry: Mapping[str, Any]) -> str:
    positions = {}
    for position, scored in entry["positions"].items():
        mine = scored["with_the_category"]
        best = mine["recall_at_precision_90"]
        positions[position] = {
            "accuracy": round(mine["accuracy"], 3),
            "recall_at_precision_90": best and round(best["recall"], 3),
            "at_gate": mine["at_gate"] and {k: round(v, 3) for k, v in mine["at_gate"].items() if k != "fired"},
            "ece": mine["ece"] and round(mine["ece"], 3),
        }
    return json.dumps(
        {"category": category, "temperature": round(entry["temperature"], 3), "gate": entry["gate"], **positions}
    )


def probe(args: argparse.Namespace) -> None:
    records = load_all_records(args.data)
    categories: dict[str, dict[str, str]] = json.loads(args.categories.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    table = StateTable.load(args.states, [args.layer])
    states = table.states[:, 0].to(device)
    rows, quiet = table_rows(table, records, categories)
    doors: dict[str, tuple[torch.Tensor, list[Row]]] = {}
    for spec in args.door:
        name, _, folder = spec.partition(DOOR_SEPARATOR)
        door_table = StateTable.load(Path(folder), [args.layer])
        doors[name] = (door_table.states[:, 0].to(device), table_rows(door_table, records, categories)[0])
    chances = compiler_fire_probabilities(args.compiler_predictions, categories) if args.compiler_predictions else None
    dev = by_position(rows, DEVELOPMENT)
    report: dict[str, Any] = {
        "development_rows": {position: len(picked) for position, picked in dev.items()},
        "silences": len(quiet),
        "categories": {},
    }
    probabilities: dict[str, dict[str, list[float]]] = {}
    gates: dict[str, float | None] = {}
    for category in sorted({category for row in rows for category in row["labels"]}):
        detector = fit_detector(states, rows, quiet, category)
        entry, probabilities[category] = score_category(detector, states, rows, quiet, category, doors)
        entry["train_positives"] = int(
            sum(
                row["labels"].get(category, 0)
                for row in rows
                if row["split"] == TRAIN and row["kind"] in DECISION_KINDS
            )
        )
        if chances is not None:
            entry["compiler"] = compiler_category(rows, category, chances)
        report["categories"][category] = entry
        gates[category] = detector.gate
        print(_summary(category, entry), flush=True)
    report["quiet_records"] = quiet_records(dev, probabilities, gates, records)
    report["languages"] = languages(dev, probabilities, gates, records)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / REPORT_FILE).write_text(json.dumps(report, indent=2), encoding="utf-8")
    positions = {position: [row["record_id"] for row in picked] for position, picked in dev.items()}
    (args.out / PROBABILITIES_FILE).write_text(
        json.dumps({"records": positions, "probabilities": probabilities}), encoding="utf-8"
    )
    print(json.dumps({"quiet_records": report["quiet_records"], "written": str(args.out)}), flush=True)


def rule_in_play(
    saved: Mapping[str, Any],
    records: Mapping[str, TrainingRecord],
    categories: Mapping[str, Mapping[str, str]],
    threshold: float = PLAY_THRESHOLD,
) -> dict[str, dict[str, float]]:
    """Per position, the accuracy on the category of each development record's rule in play (the rule its case was
    written for), from the saved probabilities of a probe run."""
    found: dict[str, dict[str, float]] = {}
    for position, record_ids in saved["records"].items():
        count = hits = 0
        for index, record_id in enumerate(record_ids):
            record = records[record_id]
            category = (record.meta or {}).get("category")
            labels = category_labels(record, categories[record_id])
            if category not in labels or category not in saved["probabilities"]:
                continue
            count += 1
            hits += int((saved["probabilities"][category][position][index] >= threshold) == bool(labels[category]))
        found[position] = {"n": count, "accuracy": hits / max(1, count)}
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description="One generic detector per rule category, on cached states.")
    commands = parser.add_subparsers(dest="command", required=True)
    mapping = commands.add_parser("map", help="write each record's rules with their categories")
    mapping.add_argument("--data", type=Path, required=True)
    mapping.add_argument("--synthetic", type=Path, required=True, help="the scenarios that declared the categories")
    mapping.add_argument("--out", type=Path, required=True)
    probing = commands.add_parser("probe", help="fit and score one detector per category")
    probing.add_argument("--states", type=Path, required=True)
    probing.add_argument("--data", type=Path, required=True)
    probing.add_argument("--categories", type=Path, required=True, help="record id to {rule id: category}")
    probing.add_argument("--layer", type=int, required=True)
    probing.add_argument("--out", type=Path, required=True)
    probing.add_argument("--door", action="append", default=[], help="NAME=STATES: also score another door's states")
    probing.add_argument("--compiler-predictions", type=Path, default=None, help="the per-rule compiler, to compare")
    playing = commands.add_parser("play", help="accuracy on the rule in play, from a probe run's probabilities")
    playing.add_argument("--probabilities", type=Path, required=True)
    playing.add_argument("--data", type=Path, required=True)
    playing.add_argument("--categories", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "map":
        write_categories(args.data, args.synthetic, args.out)
    elif args.command == "play":
        saved = json.loads(args.probabilities.read_text(encoding="utf-8"))
        categories = json.loads(args.categories.read_text(encoding="utf-8"))
        print(json.dumps(rule_in_play(saved, load_all_records(args.data), categories), indent=2), flush=True)
    else:
        probe(args)


if __name__ == "__main__":
    main()
