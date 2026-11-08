import argparse
import json
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from rein.compiler.firing import firing, firing_curve, recall_at_precision
from rein.compiler.readings import load_readings
from rein.compiler.report import prediction_rows
from rein.compiler.run import CHANNELS_FILE, CONFIG_FILE, build_compiler, load_compiler_weights
from rein.compiler.table import Item, StateTable, table_items
from rein.compiler.tool_binary import (
    ANY_SUFFIX,
    EACH_SEPARATOR,
    is_derived,
    load_tool_rules,
    parent_question,
    with_tool_questions,
)
from rein.compiler.train import predict_logits
from rein.eval import constants as ec
from rein.eval.metrics import PredictionRow, summarize
from rein.genview.build import load_all_records

DIAGNOSIS_FILE = "diagnosis.json"
DEFAULT_SPLIT = "development"
MIN_SLICE = 20

Slice = Callable[[PredictionRow], bool]


def _target(row: PredictionRow) -> bool:
    return row.tags.get("target_rule") == row.question_id


def _type_is(kind: str, row: PredictionRow) -> bool:
    return row.question_type == kind


def _target_tag_is(key: str, value: str, row: PredictionRow) -> bool:
    return _target(row) and str(row.tags.get(key)) == value


def _derived_is(marker: str, row: PredictionRow) -> bool:
    """A yes/no version of the tool rule in play."""
    return marker in row.question_id and row.tags.get("target_rule") == parent_question(row.question_id)


def _derived_elsewhere(marker: str, row: PredictionRow) -> bool:
    """A yes/no version of a tool rule, in a record that tests another rule: where a firing is a false one."""
    return marker in row.question_id and row.tags.get("target_rule") != parent_question(row.question_id)


def _original(keep: Slice, row: PredictionRow) -> bool:
    """The slice over the sheet's own questions, leaving out the yes/no versions derived from tool rules."""
    return not is_derived(row.question_id) and keep(row)


def diagnostic_slices(rows: Sequence[PredictionRow]) -> dict[str, Slice]:
    """Where the points go: the rule in play against the others, question types, and for the rule in play its
    category and case form."""
    original: dict[str, Slice] = {
        "all": lambda row: True,
        "target_rule": _target,
        "other_rules": lambda row: not _target(row),
    }
    for kind in sorted({row.question_type for row in rows}):
        original[f"type:{kind}"] = partial(_type_is, kind)
    for key in ("category", "case_form"):
        for value in sorted({str(row.tags.get(key)) for row in rows}):
            original[f"target_{key}:{value}"] = partial(_target_tag_is, key, value)
    slices: dict[str, Slice] = {name: partial(_original, keep) for name, keep in original.items()}
    slices["tool:any_tool"] = partial(_derived_is, ANY_SUFFIX)
    slices["tool:each_tool"] = partial(_derived_is, EACH_SEPARATOR)
    slices["tool:any_tool_elsewhere"] = partial(_derived_elsewhere, ANY_SUFFIX)
    slices["tool:each_tool_elsewhere"] = partial(_derived_elsewhere, EACH_SEPARATOR)
    return slices


def diagnose(rows: Sequence[PredictionRow]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for name, keep in diagnostic_slices(rows).items():
        picked = [row for row in rows if keep(row)]
        if len(picked) >= MIN_SLICE:
            curve = firing_curve(picked)
            report[name] = {
                **summarize(picked),
                "firing": firing(picked, ec.DEFAULT_GATE),
                "curve": curve,
                "recall_at_precision_90": recall_at_precision(curve),
            }
    return report


def _items_by_position(items: Sequence[Item]) -> dict[str, list[Item]]:
    groups: dict[str, list[Item]] = {}
    for item in items:
        groups.setdefault(f"{item.mode}:{item.kind}", []).append(item)
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description="Break a trained compiler's development results down by slice.")
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--readings", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--out", type=Path, default=None, help="where the reports go; the compiler folder if omitted")
    parser.add_argument("--tool-binary", type=Path, default=None, help="add yes/no versions of each tool rule")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = json.loads((args.compiler / CONFIG_FILE).read_text(encoding="utf-8"))
    trained_layers: list[int] = config["layers"]
    output_layers: list[int] = config.get("output_layers", trained_layers)
    table = StateTable.load(args.states, trained_layers)
    reading_index, readings = load_readings(args.readings)
    records = with_tool_questions(load_all_records(args.data), load_tool_rules(args.tool_binary))
    decisions, _ = table_items(table, records, reading_index)
    compiler = build_compiler(config)
    load_compiler_weights(compiler, args.compiler)
    compiler.to(device).eval()
    channels = load_file(str(args.compiler / CHANNELS_FILE))["channels"].to(device)
    states, readings = table.states.to(device), readings.to(device)
    development = [item for item in decisions if item.split == args.split]
    diagnosis: dict[str, Any] = {}
    predictions: list[str] = []
    for position, items in sorted(_items_by_position(development).items()):
        logits, masks = predict_logits(compiler, readings, states, channels, items)
        for layer in args.layers:
            column = output_layers.index(layer)
            rows = prediction_rows(logits[:, column], masks, items, config["temperatures"][column])
            diagnosis.setdefault(str(layer), {})[position] = diagnose(rows)
            predictions += [_prediction_line(layer, position, row) for row in rows]
    out_dir = args.out or args.compiler
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / diagnosis_name(args.split)
    out.write_text(json.dumps(diagnosis, indent=2), encoding="utf-8")
    (out_dir / predictions_name(args.split)).write_text("".join(predictions), encoding="utf-8")
    print(json.dumps({"written": str(out), "layers": args.layers}), flush=True)


def diagnosis_name(split: str) -> str:
    return DIAGNOSIS_FILE if split == DEFAULT_SPLIT else f"diagnosis-{split}.json"


def predictions_name(split: str) -> str:
    return f"predictions-{split}.jsonl"


def _prediction_line(layer: int, position: str, row: PredictionRow) -> str:
    """One scored question, so later analyses can join the compiler's decisions with other measurements."""
    payload = {
        "layer": layer,
        "position": position,
        "record_id": row.record_id,
        "question_id": row.question_id,
        "option_keys": list(row.option_keys),
        "probs": list(row.probs),
        "label_index": row.label_index,
        "target_rule": row.tags.get("target_rule"),
    }
    return json.dumps(payload) + "\n"


if __name__ == "__main__":
    main()
