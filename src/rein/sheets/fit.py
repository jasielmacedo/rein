"""The per-sheet fit for an end-to-end sheet (DR-025): each compiled question's neurons, fitted on the sheet's examples.

Positions and their labels follow what each rule's neurons will do in the checkpoint:
- the reply opening (and the end of a reply in progress, for rules that watch the reply) carries the judged answer;
- positions inside Qwen's own reply keep the answer for steer rules, which act during the whole reply, and take the
  null option for rules that act once (a tool call at the opening, or where a reply narrates an action, which Qwen's
  own replies rarely do), so their neurons stay quiet inside ordinary replies;
- silence positions (earlier turns) take the null option for every rule.
Records split into fit and held-out by the same hash as the oracle; the sheet's script is held out entirely.
"""

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from rein.compiler.firing import firing
from rein.compiler.model import Compiler
from rein.compiler.oracle_probe import FOLDS, is_held_out
from rein.compiler.readings import load_readings
from rein.compiler.report import prediction_rows
from rein.compiler.run import CHANNELS_FILE, CONFIG_FILE, build_compiler, load_compiler_weights
from rein.compiler.sheet_fit import FITS, SheetExamples, SheetNeurons, choose_fit_l2, compiled_neurons
from rein.compiler.table import Item, StateTable, table_items
from rein.compiler.tool_binary import ANY_SUFFIX, parent_question, with_tool_questions
from rein.eval import constants as ec
from rein.genview import constants as gc
from rein.genview.build import load_all_records
from rein.genview.replies import REPLY
from rein.sheets.model import CallAction, Sheet, ToolAction, load_sheet

FULL_FIT_MIN_RECORDS = 24
YES_NO_OPTIONS = 2
NEURONS_FILE = "neurons.safetensors"
MANIFEST_FILE = "neurons.json"
REPORT_FILE = "fit.json"
SCRIPT_SPLIT = "gold"


def compiled_questions(sheet: Sheet) -> dict[str, str]:
    """The questions whose neurons go into the checkpoint, with their rule's action: a tool rule through its any-tool
    yes/no version, every other rule as written."""
    questions: dict[str, str] = {}
    for rule in sheet.rules:
        if isinstance(rule.action, ToolAction):
            questions[f"{rule.id}{ANY_SUFFIX}"] = rule.action.kind
        else:
            questions[rule.id] = rule.action.kind
    return questions


def _quiet(item: Item) -> Item:
    return replace(item, label=item.null, target=tuple(float(i == item.null) for i in range(len(item.option_keys))))


def labelled_items(items: Sequence[Item], actions: Mapping[str, str]) -> list[Item]:
    """Items of the compiled questions, each position labelled by what the rule's neurons do there.

    A rule that watches the reply decides at the end of a reply in progress, never at the opening; a tool rule decides
    at the opening and stays quiet inside the reply; a steer rule keeps its answer through the whole reply.
    """
    kept = []
    for item in items:
        action = actions.get(item.question_id)
        if action is None:
            continue
        if action == CallAction().kind:
            if item.kind == gc.REPLY_START:
                continue
            if item.kind == REPLY:
                item = _quiet(item)
        elif action == ToolAction().kind and item.kind in (REPLY, gc.CONTINUATION_END):
            item = _quiet(item)
        kept.append(item)
    return kept


def _features(states: torch.Tensor, items: Sequence[Item]) -> torch.Tensor:
    return states[[item.row for item in items]].float()


def fit_question(
    prior: SheetNeurons,
    train: Sequence[Item],
    quiet: Sequence[Item],
    states: torch.Tensor,
    channel: torch.Tensor,
    temperature: float,
) -> tuple[SheetNeurons, str, float]:
    """The question's neurons fitted on its training records: the full fit for yes/no questions and for lists with
    enough records, only the offsets for lists with few records (a full fit on 4 to 16 records hurt them)."""
    records = sorted({item.record_id for item in train})
    fold_of = {record: index % FOLDS for index, record in enumerate(records)}
    examples = SheetExamples(
        _features(states, train), torch.tensor([item.label for item in train]), _features(states, quiet), train[0].null
    )
    folds = (
        torch.tensor([fold_of[item.record_id] for item in train]),
        torch.tensor([fold_of.get(item.record_id, 0) for item in quiet], dtype=torch.long),
    )
    kind = "fit" if len(records) >= FULL_FIT_MIN_RECORDS or len(prior.offsets) == YES_NO_OPTIONS else "offsets"
    fit = FITS[kind]
    l2 = choose_fit_l2(prior, examples, folds, channel, temperature, fit)
    return fit(prior, examples, channel, temperature, l2), kind, l2


def _report(rows: Sequence[Any]) -> dict[str, Any]:
    from rein.eval.metrics import summarize

    return {**summarize(rows), "firing": firing(rows, ec.DEFAULT_GATE)} if rows else {}


def evaluate(
    neurons: SheetNeurons, items: Sequence[Item], states: torch.Tensor, channel: torch.Tensor, temperature: float
) -> dict[str, Any]:
    """Per position kind, the fitted neurons' answers on held-out items."""
    report: dict[str, Any] = {}
    for kind in sorted({item.kind for item in items}):
        picked = [item for item in items if item.kind == kind]
        logits = neurons.logits(_features(states, picked), channel)
        masks = torch.ones_like(logits, dtype=torch.bool)
        report[kind] = _report(prediction_rows(logits, masks, picked, temperature))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit the compiled neurons of an end-to-end sheet on its examples.")
    parser.add_argument("--sheet", type=Path, required=True)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--readings", type=Path, required=True)
    parser.add_argument(
        "--data", type=Path, required=True, help="the examples' train.jsonl and the script's gold.jsonl"
    )
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sheet = load_sheet(args.sheet)
    actions = compiled_questions(sheet)
    tool_rule = next((rule.id for rule in sheet.rules if isinstance(rule.action, ToolAction)), None)
    records = load_all_records(args.data)
    if tool_rule is not None:
        records = with_tool_questions(records, {record_id: tool_rule for record_id in records})
    config = json.loads((args.compiler / CONFIG_FILE).read_text(encoding="utf-8"))
    compiler = build_compiler(config)
    if not isinstance(compiler, Compiler):
        raise SystemExit("the per-sheet fit keeps the one-block neuron form: use a Compiler")
    load_compiler_weights(compiler, args.compiler)
    compiler.eval()
    channels = load_file(str(args.compiler / CHANNELS_FILE))["channels"].float()
    table = StateTable.load(args.states, config["layers"])
    reading_index, readings = load_readings(args.readings)
    decisions, silences = table_items(table, records, reading_index)
    items = labelled_items([item for item in decisions if item.label >= 0], actions)
    quiet = labelled_items(silences, actions)
    column, block = 0, config["layers"][0]
    temperature = float(config["temperatures"][column])
    states = table.states[:, column]
    channel = channels[column]
    tensors: dict[str, torch.Tensor] = {}
    manifest: dict[str, Any] = {"sheet": sheet.id, "block": block, "temperature": temperature, "questions": {}}
    report: dict[str, Any] = {}
    for question_id, action in actions.items():
        mine = [item for item in items if item.question_id == question_id]
        silent = [item for item in quiet if item.question_id == question_id]
        if not mine:
            report[question_id] = {"missing": True}
            continue
        prior = compiled_neurons(compiler, readings, mine[0].reading, column, len(mine[0].option_keys))
        fit_items = [item for item in mine if item.split != SCRIPT_SPLIT and not is_held_out(item.record_id)]
        fit_quiet = [item for item in silent if item.split != SCRIPT_SPLIT and not is_held_out(item.record_id)]
        held = [item for item in mine if item.split != SCRIPT_SPLIT and is_held_out(item.record_id)]
        script = [item for item in mine if item.split == SCRIPT_SPLIT]
        fitted, kind, l2 = fit_question(prior, fit_items, fit_quiet, states, channel, temperature)
        tensors[f"{question_id}.detectors"] = fitted.detectors.contiguous()
        tensors[f"{question_id}.offsets"] = fitted.offsets.contiguous()
        manifest["questions"][question_id] = {
            "rule": parent_question(question_id),
            "action": action,
            "option_keys": list(mine[0].option_keys),
            "null": mine[0].null,
            "fit": kind,
            "l2": l2,
            "records": len({item.record_id for item in fit_items}),
        }
        report[question_id] = {
            "fit": kind,
            "compiler": {
                "held_out": evaluate(prior, held, states, channel, temperature),
                "script": evaluate(prior, script, states, channel, temperature),
            },
            "fitted": {
                "held_out": evaluate(fitted, held, states, channel, temperature),
                "script": evaluate(fitted, script, states, channel, temperature),
            },
        }
        print(
            json.dumps(
                {"question": question_id, "fit": kind, "records": manifest["questions"][question_id]["records"]}
            ),
            flush=True,
        )
    tensors["channel"] = channel.contiguous()
    args.out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out / NEURONS_FILE))
    (args.out / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.out / REPORT_FILE).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "written": str(args.out),
                "questions": len(manifest["questions"]),
                "reading_kinds": sorted({gc.REPLY_START, REPLY}),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
