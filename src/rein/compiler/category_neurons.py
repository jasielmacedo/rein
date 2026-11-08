"""The checkpoint's generic neurons (DR-026): for each rule category, detectors fitted once on every agent's states and
then kept, unchanged, for any agent. Each detector is a neuron's exact read (its centering and its intercept ride on the
bias channel), with the temperature and the gate chosen on the calibration split.

- steer: fires where the reply opens and on every reply token while a rule of its category fires;
- force: fires only where the reply opens and a tool rule fires, so that the reply opens with `<tool_call>`;
- veto: fires only where the reply opens, for an agent whose tool rule does not fire (most calls open the reply,
  and a veto on every reply token moved the whole residual stream when nothing fired).
All of them stay quiet on the prompt (the silence positions) and inside the reasoning.
"""

import argparse
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from rein.compiler import category_metrics as m
from rein.compiler.category_probe import (
    CALIBRATION,
    DEVELOPMENT,
    TRAIN,
    Detector,
    Row,
    fit_labelled,
    table_rows,
)
from rein.compiler.table import StateTable
from rein.genview import constants as gc
from rein.genview.build import load_all_records
from rein.genview.reasoning import ANSWER_START, REASONING
from rein.genview.replies import REPLY
from rein.neurons.bias import bias_channel, channel_stats

STEER, FORCE, VETO = "steer", "force", "veto"
TOOL_CATEGORY = "tool_trigger"
STEER_CATEGORIES = (
    "locked_attribute",
    "break_character",
    "prompt_injection",
    "lore_consistency",
    "business_limit",
    "npc_secret",
    "age_rating",
    "format_rule",
)
OPENING_KINDS = (gc.REPLY_START, ANSWER_START)
NEURON_KINDS = (*OPENING_KINDS, REPLY, REASONING)
CHANNEL_SAMPLES = 20_000
SEED = 0
NEURONS_FILE = "category_neurons.safetensors"
MANIFEST_FILE = "category_neurons.json"
REPORT_FILE = "category_neurons_report.json"
EVALUATION_FILE = "category_neurons_evaluation.json"


def action_label(action: str, kind: str, fired: int | None) -> int:
    """What the neuron of `action` must do at one position (1 fire, 0 stay quiet); `fired` is None for an agent
    without a rule of the category."""
    if kind not in (*OPENING_KINDS, REPLY):
        return 0
    if action == STEER:
        return int(fired == 1)
    if action == FORCE:
        return int(kind in OPENING_KINDS and fired == 1)
    return int(kind in OPENING_KINDS and fired == 0)


def detector_name(category: str, action: str) -> str:
    return f"{category}:{action}"


def wanted(steer: Sequence[str]) -> list[tuple[str, str]]:
    return [(category, STEER) for category in steer] + [(TOOL_CATEGORY, FORCE), (TOOL_CATEGORY, VETO)]


def labelled(
    rows: Sequence[Row], quiet: Sequence[Row], category: str, action: str, split: str
) -> tuple[list[Row], torch.Tensor]:
    """A split's rows and silences with the action's labels."""
    picked = [row for row in rows if row["split"] == split]
    silences = [row for row in quiet if row["split"] == split]
    labels = [action_label(action, row["kind"], row["labels"].get(category)) for row in picked]
    return [*picked, *silences], torch.tensor(labels + [0] * len(silences), dtype=torch.long)


def floored(detector: Detector, action: str, floor: float) -> Detector:
    """A steer or force neuron fires at every position its gate passes, prompt tokens included: a floor keeps it from
    firing on what calibration let through. The veto keeps its own gate."""
    if action == VETO or detector.gate is None:
        return detector
    return replace(detector, gate=max(detector.gate, floor))


def channel_of(states: torch.Tensor, rows: Sequence[Row]) -> torch.Tensor:
    """The bias channel over a sample of the training rows: `c . x` is about 1 at every position the neurons see."""
    sample = random.Random(SEED).sample(list(rows), min(CHANNEL_SAMPLES, len(rows)))
    return bias_channel(states[[row["row"] for row in sample]].float().cpu()).to(states.device)


def kind_report(detector: Detector, states: torch.Tensor, rows: Sequence[Row], labels: torch.Tensor) -> dict[str, Any]:
    """The detector on the development rows, position by position (silences included, where it must stay quiet)."""
    report: dict[str, Any] = {}
    positions = sorted({row["position"] for row in rows})
    for position in positions:
        index = [i for i, row in enumerate(rows) if row["position"] == position]
        probs = detector.probabilities(states[[rows[i]["row"] for i in index]])
        picked = labels[index]
        report[position] = {
            **m.rates(probs, picked),
            "at_gate": m.at_gate(probs, picked, detector.gate),
            "ece": m.calibration_error(probs, picked),
        }
    return report


def wrong_fires(
    detectors: Mapping[str, Detector],
    states: torch.Tensor,
    rows: Sequence[Row],
    quiet: Sequence[Row],
    split: str = DEVELOPMENT,
) -> dict[str, Any]:
    """On records of `split` where no rule fires at all: how often any steer or force neuron fires, per position."""
    acting = {name: detector for name, detector in detectors.items() if not name.endswith(f":{VETO}")}
    calm = [row for row in rows if row["split"] == split and not any(row["labels"].values())]
    calm += [row for row in quiet if row["split"] == split]
    report: dict[str, Any] = {}
    for position in sorted({row["position"] for row in calm}):
        picked = [row for row in calm if row["position"] == position]
        chunk = states[[row["row"] for row in picked]]
        fired = torch.zeros(len(picked), dtype=torch.bool)
        for detector in acting.values():
            if detector.gate is not None:
                fired |= detector.probabilities(chunk) >= detector.gate
        report[position] = {"rows": len(picked), "any_fire": float(fired.float().mean()) if len(picked) else 0.0}
    return report


def save(out: Path, layer: int, channel: torch.Tensor, detectors: Mapping[str, Detector], report: Any) -> None:
    out.mkdir(parents=True, exist_ok=True)
    tensors = {"channel": channel.cpu().contiguous()}
    manifest: dict[str, Any] = {"block": layer, "detectors": {}}
    for name, detector in detectors.items():
        direction, offset = detector.neuron_form()
        tensors[f"{name}.direction"] = direction.cpu().contiguous()
        category, action = name.split(":")
        manifest["detectors"][name] = {
            "category": category,
            "action": action,
            "offset": offset,
            "temperature": detector.temperature,
            "gate": detector.gate,
        }
    save_file(tensors, str(out / NEURONS_FILE))
    (out / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / REPORT_FILE).write_text(json.dumps(report, indent=2), encoding="utf-8")


def load_detectors(folder: Path) -> dict[str, Detector]:
    """The saved neurons as detectors with the very same read: no centering, the offset on the bias channel."""
    manifest = json.loads((folder / MANIFEST_FILE).read_text(encoding="utf-8"))
    tensors = load_file(str(folder / NEURONS_FILE))
    channel = tensors["channel"].float()
    hidden = channel.shape[0]
    detectors = {}
    for name, info in manifest["detectors"].items():
        yes = torch.cat([tensors[f"{name}.direction"].float(), torch.tensor([info["offset"]])])
        weights = torch.stack([torch.zeros_like(yes), yes])
        detectors[name] = Detector(
            weights, torch.zeros(hidden), torch.ones(hidden), info["temperature"], info["gate"], channel
        )
    return detectors


def evaluate(args: argparse.Namespace, states: torch.Tensor, rows: list[Row], quiet: list[Row]) -> None:
    """Saved neurons on other states (another door, another template): development split only, nothing refitted."""
    detectors = {name: _on(detector, states.device) for name, detector in load_detectors(args.neurons).items()}
    report: dict[str, Any] = {"detectors": {}}
    for name, detector in detectors.items():
        category, action = name.split(":")
        picked_rows, picked_labels = labelled(rows, quiet, category, action, args.split)
        report["detectors"][name] = {"positions": kind_report(detector, states, picked_rows, picked_labels)}
    report["wrong_fires"] = wrong_fires(detectors, states, rows, quiet, args.split)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / EVALUATION_FILE).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"wrong_fires": report["wrong_fires"], "written": str(args.out)}), flush=True)


def _on(detector: Detector, device: torch.device) -> Detector:
    assert detector.channel is not None
    return Detector(
        detector.weights.to(device),
        detector.mean.to(device),
        detector.scale.to(device),
        detector.temperature,
        detector.gate,
        detector.channel.to(device),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit the checkpoint's generic neurons, one per category and action.")
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--categories", type=Path, required=True, help="record id to {rule id: category}")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steer", nargs="*", default=list(STEER_CATEGORIES))
    parser.add_argument("--neurons", type=Path, default=None, help="evaluate these saved neurons instead of fitting")
    parser.add_argument("--gate-floor", type=float, default=0.0, help="lowest gate for the neurons that act")
    parser.add_argument("--split", default=DEVELOPMENT, help="the split an evaluation reads")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = load_all_records(args.data)
    categories: dict[str, dict[str, str]] = json.loads(args.categories.read_text(encoding="utf-8"))
    table = StateTable.load(args.states, [args.layer])
    states = table.states[:, 0].to(device)
    rows, quiet = table_rows(table, records, categories, NEURON_KINDS)
    if args.neurons is not None:
        evaluate(args, states, rows, quiet)
        return
    channel = channel_of(states, [row for row in [*rows, *quiet] if row["split"] == TRAIN])
    stats = channel_stats(channel.cpu(), states[[row["row"] for row in rows[:CHANNEL_SAMPLES]]].float().cpu())
    report: dict[str, Any] = {"channel": {"mean": stats.mean, "variation": stats.variation}, "detectors": {}}
    detectors: dict[str, Detector] = {}
    for category, action in wanted(args.steer):
        name = detector_name(category, action)
        train = labelled(rows, quiet, category, action, TRAIN)
        calibration = labelled(rows, quiet, category, action, CALIBRATION)
        detector = floored(fit_labelled(states, train, calibration, channel), action, args.gate_floor)
        detectors[name] = detector
        development_rows, development_labels = labelled(rows, quiet, category, action, DEVELOPMENT)
        report["detectors"][name] = {
            "temperature": detector.temperature,
            "gate": detector.gate,
            "train_positives": int(train[1].sum()),
            "positions": kind_report(detector, states, development_rows, development_labels),
        }
        print(json.dumps({"detector": name, "temperature": detector.temperature, "gate": detector.gate}), flush=True)
    report["wrong_fires"] = wrong_fires(detectors, states, rows, quiet)
    print(json.dumps({"wrong_fires": report["wrong_fires"]}), flush=True)
    save(args.out, args.layer, channel, detectors, report)
    print(json.dumps({"written": str(args.out), "detectors": len(detectors)}), flush=True)


if __name__ == "__main__":
    main()
