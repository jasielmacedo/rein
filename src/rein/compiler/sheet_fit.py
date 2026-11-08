"""Per-sheet fit: the compiler's neurons for one sheet, refined on that sheet's own examples.

Each option keeps the neuron form, a detector over the MLP input plus an offset along the bias channel, so a fitted
sheet compiles like any other. The fit starts at the compiler's numbers and λ‖W - W0‖² pulls it back toward them, with
λ chosen by cross-validation over the sheet's records. The silence positions of those records join the fit as the null
option, weighted as in the compiler's training, so the fit cannot buy recall with false fires. Like the compiler, it
runs outside the runtime, once per sheet.

On the oracle's rules, kept out of the compiler's training, it compares the compiler alone, the fit with a few to all
of each rule's training records, and the oracle's detectors trained from scratch, on the same held-out cases.
"""

import argparse
import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from rein.compiler.constants import ORACLE_PREFIX
from rein.compiler.evaluate import diagnose
from rein.compiler.firing import CURVE_THRESHOLDS, fire
from rein.compiler.model import Compiler
from rein.compiler.oracle_probe import FOLDS, KIND, LBFGS_STEPS, MODE, is_held_out, oracle_rows
from rein.compiler.reader import RuleReading
from rein.compiler.readings import load_readings
from rein.compiler.run import CHANNELS_FILE, CONFIG_FILE, build_compiler, load_compiler_weights
from rein.compiler.table import Item, StateTable, table_items
from rein.compiler.tool_binary import ANY_SUFFIX, TOOL_CATEGORY, is_derived, parent_question, with_tool_questions
from rein.compiler.train import TrainSettings
from rein.eval.metrics import PredictionRow
from rein.genview.build import load_all_records
from rein.schema import TrainingRecord

FIT_L2_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
SHOTS = (4, 8, 16)
ALL_RECORDS = "all"
SILENCE_WEIGHT = TrainSettings.silence_weight
MIN_SCALE = 1e-6
COMPILER, ORACLE = "compiler", "oracle"


@dataclass(frozen=True)
class SheetNeurons:
    """One question's neurons in one block: a detector per option (options, hidden) and an offset per option."""

    detectors: torch.Tensor
    offsets: torch.Tensor

    def logits(self, states: torch.Tensor, channel: torch.Tensor) -> torch.Tensor:
        return states @ self.detectors.T + (states @ channel)[:, None] * self.offsets


@dataclass(frozen=True)
class SheetExamples:
    """A question's examples in one block: decision states with their labels, silence states, and the null option."""

    states: torch.Tensor
    labels: torch.Tensor
    quiet: torch.Tensor
    null: int

    def take(self, decisions: torch.Tensor, quiet: torch.Tensor) -> "SheetExamples":
        return SheetExamples(self.states[decisions], self.labels[decisions], self.quiet[quiet], self.null)

    def scaled(self, scale: torch.Tensor) -> "SheetExamples":
        return SheetExamples(self.states / scale, self.labels, self.quiet / scale, self.null)


def compiled_neurons(compiler: Compiler, readings: RuleReading, reading: int, block: int, options: int) -> SheetNeurons:
    """The compiler's neurons for one question in one of its blocks."""
    with torch.no_grad():
        compiled = compiler(readings.select(torch.tensor([reading])))
    if compiled.detectors.shape[3] != 1:
        raise ValueError("the per-sheet fit keeps one neuron per option: use a compiler with one facet")
    return SheetNeurons(
        compiled.detectors[0, :options, block, 0].float(), compiled.offsets[0, :options, block, 0].float()
    )


def sheet_loss(
    neurons: SheetNeurons, examples: SheetExamples, channel: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Cross-entropy on the decisions plus, weighted as in compiler training, on the silences as the null option."""
    loss = F.cross_entropy(neurons.logits(examples.states, channel) / temperature, examples.labels)
    if len(examples.quiet) and examples.null >= 0:
        quiet_labels = torch.full((len(examples.quiet),), examples.null, dtype=torch.long)
        quiet_logits = neurons.logits(examples.quiet, channel) / temperature
        loss = loss + SILENCE_WEIGHT * F.cross_entropy(quiet_logits, quiet_labels)
    return loss


def fit_sheet(
    prior: SheetNeurons, examples: SheetExamples, channel: torch.Tensor, temperature: float, l2: float
) -> SheetNeurons:
    """LBFGS from the compiler's neurons on the sheet's loss plus λ‖W - W0‖².

    It optimizes over states scaled per dimension, which keeps the problem well conditioned, and maps the result back,
    so the penalty and the returned neurons are in the block's own units.
    """
    rows = torch.cat([examples.states, examples.quiet])
    scale = rows.std(0).clamp(min=MIN_SCALE) if len(rows) > 1 else torch.ones_like(channel)
    scaled, scaled_channel = examples.scaled(scale), channel * scale
    start = prior.detectors * scale
    detectors = start.clone().requires_grad_(True)
    offsets = prior.offsets.clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS([detectors, offsets], max_iter=LBFGS_STEPS, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = sheet_loss(SheetNeurons(detectors, offsets), scaled, scaled_channel, temperature)
        loss = loss + l2 * ((detectors - start) / scale).pow(2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return SheetNeurons(detectors.detach() / scale, offsets.detach())


def fit_offsets(
    prior: SheetNeurons, examples: SheetExamples, channel: torch.Tensor, temperature: float, l2: float
) -> SheetNeurons:
    """Only the offsets move, one threshold per option, pulled back to the compiler's by λ‖b - b0‖²: a handful of
    records cannot bend the detectors."""
    offsets = prior.offsets.clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS([offsets], max_iter=LBFGS_STEPS, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = sheet_loss(SheetNeurons(prior.detectors, offsets), examples, channel, temperature)
        loss = loss + l2 * (offsets - prior.offsets).pow(2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return SheetNeurons(prior.detectors, offsets.detach())


Fit = Callable[[SheetNeurons, SheetExamples, torch.Tensor, float, float], SheetNeurons]
FITS: dict[str, Fit] = {"fit": fit_sheet, "offsets": fit_offsets}


def choose_fit_l2(
    prior: SheetNeurons,
    examples: SheetExamples,
    folds: tuple[torch.Tensor, torch.Tensor],
    channel: torch.Tensor,
    temperature: float,
    fit: Fit = fit_sheet,
) -> float:
    """The λ whose fits on the other folds' records best explain each fold's records; `folds` gives each decision's
    and each silence's fold."""
    decision_folds, quiet_folds = folds
    scores = []
    for l2 in FIT_L2_GRID:
        total = 0.0
        for fold in range(FOLDS):
            kept, out = decision_folds != fold, decision_folds == fold
            if not kept.any() or not out.any():
                continue
            fitted = fit(prior, examples.take(kept, quiet_folds != fold), channel, temperature, l2)
            with torch.no_grad():
                total -= float(sheet_loss(fitted, examples.take(out, quiet_folds == fold), channel, temperature))
        scores.append(total)
    return FIT_L2_GRID[max(range(len(FIT_L2_GRID)), key=scores.__getitem__)]


def _order(record_id: str) -> int:
    return int(hashlib.sha256(record_id.encode("utf-8")).hexdigest(), 16)


def _rows(items: Sequence[Item], probs: torch.Tensor) -> list[PredictionRow]:
    return [
        PredictionRow(
            item.record_id,
            item.question_id,
            item.question_type,
            item.option_keys,
            tuple(prob.tolist()),
            item.label,
            item.tags,
        )
        for item, prob in zip(items, probs, strict=True)
    ]


@dataclass(frozen=True)
class QuestionCases:
    """One question's decision and silence items, split into the records the fit sees and the held-out ones."""

    train: list[Item]
    test: list[Item]
    train_quiet: list[Item]
    test_quiet: list[Item]

    @classmethod
    def split(cls, items: Sequence[Item], quiet: Sequence[Item]) -> "QuestionCases":
        return cls(
            [item for item in items if not is_held_out(item.record_id)],
            [item for item in items if is_held_out(item.record_id)],
            [item for item in quiet if not is_held_out(item.record_id)],
            [item for item in quiet if is_held_out(item.record_id)],
        )


@dataclass(frozen=True)
class Scored:
    decisions: list[PredictionRow]
    silences: list[PredictionRow]
    l2: float | None = None


def question_scores(
    prior: SheetNeurons, cases: QuestionCases, states: torch.Tensor, channel: torch.Tensor, temperature: float
) -> dict[str, Scored]:
    """Held-out rows for the compiler's neurons, and for each fit on the first 4, 8, 16 and all training records."""

    def features(picked: Sequence[Item]) -> torch.Tensor:
        return states[[item.row for item in picked]].float()

    def scored(neurons: SheetNeurons, l2: float | None = None) -> Scored:
        with torch.no_grad():
            decisions = torch.softmax(neurons.logits(features(cases.test), channel) / temperature, dim=-1)
            silences = torch.softmax(neurons.logits(features(cases.test_quiet), channel) / temperature, dim=-1)
        return Scored(_rows(cases.test, decisions), _rows(cases.test_quiet, silences), l2)

    results = {COMPILER: scored(prior)}
    records = sorted({item.record_id for item in cases.train}, key=_order)
    for shots in [*(count for count in SHOTS if count < len(records)), len(records)]:
        fold_of = {record: index % FOLDS for index, record in enumerate(records[:shots])}
        picked = [item for item in cases.train if item.record_id in fold_of]
        quiet = [item for item in cases.train_quiet if item.record_id in fold_of]
        examples = SheetExamples(
            features(picked), torch.tensor([item.label for item in picked]), features(quiet), picked[0].null
        )
        folds = (
            torch.tensor([fold_of[item.record_id] for item in picked]),
            torch.tensor([fold_of[item.record_id] for item in quiet], dtype=torch.long),
        )
        for kind, fit in FITS.items():
            l2 = choose_fit_l2(prior, examples, folds, channel, temperature, fit)
            name = f"{kind}_{shots}" if shots < len(records) else f"{kind}_{ALL_RECORDS}"
            results[name] = scored(fit(prior, examples, channel, temperature, l2), l2)
    return results


def _slice_of(row: PredictionRow) -> str:
    if is_derived(row.question_id):
        return "tool:any_tool" if ANY_SUFFIX in row.question_id else "tool:each_tool"
    return f"target_category:{row.tags.get('category')}"


def _fired_share(rows: Sequence[PredictionRow], threshold: float) -> float:
    outcomes = [fire(row, threshold) for row in rows]
    return sum(bool(outcome and outcome.fired) for outcome in outcomes) / max(1, len(rows))


def silence_report(rows: Sequence[PredictionRow]) -> dict[str, dict[str, float]]:
    """Share of held-out silence positions where the question would fire, per threshold, overall and per slice."""
    groups: dict[str, list[PredictionRow]] = {"all": list(rows)}
    for row in rows:
        groups.setdefault(_slice_of(row), []).append(row)
    return {
        name: {"n": float(len(picked)), **{str(t): _fired_share(picked, t) for t in CURVE_THRESHOLDS}}
        for name, picked in sorted(groups.items())
    }


def oracle_tool_rules(records: Mapping[str, TrainingRecord], rules: Sequence[Mapping[str, str]]) -> dict[str, str]:
    """For each oracle record, its tool rule, so the yes/no versions can be derived from it."""
    tools = {(rule["scenario_id"], rule["rule_id"]) for rule in rules if rule["category"] == TOOL_CATEGORY}
    return {
        record.id: question_id
        for record in records.values()
        for question_id in record.questions
        if (record.scenario_id.removeprefix(ORACLE_PREFIX), question_id) in tools
    }


def _studied(item: Item) -> bool:
    """The record's own rule, or a yes/no version of it."""
    return item.tags.get("target_rule") == parent_question(item.question_id)


def _l2_summary(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {"min": ordered[0], "median": ordered[len(ordered) // 2], "max": ordered[-1]} if ordered else {}


def layer_comparison(
    compiler: Compiler,
    readings: RuleReading,
    items: Sequence[Item],
    quiet: Sequence[Item],
    states: torch.Tensor,
    column: int,
    channel: torch.Tensor,
    temperature: float,
) -> dict[str, Any]:
    """Every way of getting a question's neurons, scored on the same held-out cases of one block."""
    grouped: dict[int, list[Item]] = {}
    for item in items:
        grouped.setdefault(item.reading, []).append(item)
    grouped_quiet: dict[int, list[Item]] = {}
    for item in quiet:
        grouped_quiet.setdefault(item.reading, []).append(item)
    scores: dict[str, list[Scored]] = {}
    block_states = states[:, column]
    started = time.monotonic()
    for done, (reading, question_items) in enumerate(grouped.items(), start=1):
        cases = QuestionCases.split(question_items, grouped_quiet.get(reading, []))
        if cases.train and cases.test:
            prior = compiled_neurons(compiler, readings, reading, column, len(question_items[0].option_keys))
            for name, result in question_scores(prior, cases, block_states, channel, temperature).items():
                scores.setdefault(name, []).append(result)
        progress = {"question": done, "of": len(grouped), "seconds": round(time.monotonic() - started, 1)}
        print(json.dumps(progress), flush=True)
    decisions = {name: [row for result in results for row in result.decisions] for name, results in scores.items()}
    decisions[ORACLE] = oracle_rows(items, states, column)
    print(json.dumps({"oracle": "done", "seconds": round(time.monotonic() - started, 1)}), flush=True)
    shared = {(row.record_id, row.question_id) for row in decisions[ORACLE]}
    report: dict[str, Any] = {}
    for name, rows in decisions.items():
        results = scores.get(name, [])
        report[name] = {
            "diagnosis": diagnose(rows),
            "on_oracle_cases": diagnose([row for row in rows if (row.record_id, row.question_id) in shared]),
            "silence_false_fire": silence_report([row for result in results for row in result.silences]),
            "l2": _l2_summary([result.l2 for result in results if result.l2 is not None]),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit each oracle rule's neurons from the compiler's, and compare.")
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--readings", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--rules", type=Path, required=True, help="the oracle's rules.json")
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--tool-binary", action="store_true", help="also fit the yes/no versions of the tool rules")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    records = load_all_records(args.data)
    if args.tool_binary:
        rules = json.loads(args.rules.read_text(encoding="utf-8"))
        records = with_tool_questions(records, oracle_tool_rules(records, rules))
    config = json.loads((args.compiler / CONFIG_FILE).read_text(encoding="utf-8"))
    compiler = build_compiler(config)
    if not isinstance(compiler, Compiler):
        raise SystemExit("the per-sheet fit keeps the one-block neuron form: use a Compiler, not a circuit")
    load_compiler_weights(compiler, args.compiler)
    compiler.eval()
    channels = load_file(str(args.compiler / CHANNELS_FILE))["channels"].float()
    table = StateTable.load(args.states, config["layers"])
    reading_index, readings = load_readings(args.readings)
    decisions, silences = table_items(table, records, reading_index)
    items = [
        item for item in decisions if item.mode == MODE and item.kind == KIND and item.label >= 0 and _studied(item)
    ]
    quiet = [item for item in silences if _studied(item)]
    report: dict[str, Any] = {
        "questions": len({item.reading for item in items}),
        "items": len(items),
        "silences": len(quiet),
        "layers": {},
    }
    for column, layer in enumerate(table.layers):
        report["layers"][str(layer)] = layer_comparison(
            compiler, readings, items, quiet, table.states, column, channels[column], config["temperatures"][column]
        )
        print(json.dumps({"layer": layer, "done": True}), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"written": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
