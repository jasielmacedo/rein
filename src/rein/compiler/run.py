import argparse
import json
import random
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from rein.compiler import constants as c
from rein.compiler.circuit import CircuitCompiler
from rein.compiler.holdout import held_out_readings, load_rule_pairs
from rein.compiler.model import Compiler
from rein.compiler.readings import load_readings
from rein.compiler.report import layer_report, layer_temperatures
from rein.compiler.table import Item, StateTable, table_items
from rein.compiler.tool_binary import load_tool_rules, with_tool_questions
from rein.compiler.train import RuleModel, TrainSettings, predict_logits, train_compiler
from rein.genview.build import load_all_records
from rein.neurons.bias import bias_channel

COMPILER_FILE = "compiler.safetensors"
CHANNELS_FILE = "channels.safetensors"
CONFIG_FILE = "compiler.json"
ENCODER_PARTS = ("query_in", "option_in", "out_norm")
FLAT_ENCODER_BLOCK = re.compile(r"^encoder\.(\d+)\.")
REPORT_FILE = "report.json"
DEFAULT_CHANNEL_SAMPLES = 20000
CALIBRATION = "calibration"
EVALUATED_SPLITS = ("development", "test")


def _by_split(items: list[Item], split: str) -> list[Item]:
    return [item for item in items if item.split == split]


def channels_from(states: torch.Tensor, rows: list[int], samples: int, seed: int) -> torch.Tensor:
    picked = sorted(random.Random(seed).sample(rows, min(samples, len(rows))))
    chosen = states[picked].float()
    return torch.stack([bias_channel(chosen[:, layer]) for layer in range(states.shape[1])])


def build_compiler(architecture: dict[str, Any]) -> RuleModel:
    """The facet compiler, or a two-block circuit when `circuit_units` is set (its layers are the two blocks)."""
    if architecture.get("circuit_units"):
        first, second = architecture["layers"]
        return CircuitCompiler(
            architecture["hidden_size"], (first, second), architecture["circuit_units"], rank=architecture["rank"]
        )
    return Compiler(
        architecture["hidden_size"],
        architecture["layers"],
        rank=architecture["rank"],
        facets=architecture.get("facets", 1),
    )


def load_compiler_weights(compiler: RuleModel, folder: Path) -> None:
    """Loads a saved compiler, also one saved before the rule encoder became its own module."""
    stored = load_file(str(folder / COMPILER_FILE))
    compiler.load_state_dict({_current_name(key): value for key, value in stored.items()})


def _current_name(key: str) -> str:
    if key.split(".", 1)[0] in ENCODER_PARTS:
        return f"encoder.{key}"
    return FLAT_ENCODER_BLOCK.sub(r"encoder.blocks.\1.", key)


def _kept(item: Item, modes: list[str], kinds: list[str]) -> bool:
    return (not modes or item.mode in modes) and (not kinds or item.kind in kinds)


def _print_summary(report: dict[str, Any]) -> None:
    for layer, entry in report.items():
        for position, values in entry["positions"].items():
            summary = values["summary"]
            hard = values["slices"].get("words_same_intent_different", {}).get("accuracy", float("nan"))
            print(
                json.dumps(
                    {
                        "layer": int(layer),
                        "position": position,
                        "n": int(summary["n"]),
                        "accuracy": round(summary["accuracy"], 4),
                        "ece": round(summary["ece"], 4),
                        "hard_negative": round(hard, 4),
                        "false_fire": round(entry.get("false_fire_at_gate", float("nan")), 4),
                        "channel_variation": round(entry.get("bias_channel", {}).get("variation", float("nan")), 4),
                    }
                ),
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the rule compiler on stored Qwen3.5 states and report it.")
    parser.add_argument("--states", type=Path, nargs="+", required=True)
    parser.add_argument("--readings", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--extra-data", type=Path, nargs="*", default=[], help="more record folders, merged in")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="*", default=None)
    parser.add_argument("--epochs", type=int, default=TrainSettings.epochs)
    parser.add_argument("--rows-per-batch", type=int, default=TrainSettings.rows_per_batch)
    parser.add_argument("--channel-samples", type=int, default=DEFAULT_CHANNEL_SAMPLES)
    parser.add_argument("--rank", type=int, default=c.DETECTOR_RANK)
    parser.add_argument("--facets", type=int, default=1, help="neurons per firing option in each block")
    parser.add_argument("--circuit-units", type=int, default=0, help="feature neurons of a two-block circuit")
    parser.add_argument("--modes", nargs="*", default=[], help="keep only these view modes")
    parser.add_argument("--kinds", nargs="*", default=[], help="keep only these decision positions")
    parser.add_argument("--tool-binary", type=Path, default=None, help="add yes/no versions of each tool rule")
    parser.add_argument("--exclude-rules", type=Path, default=None, help="rule list kept out of training entirely")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    table = StateTable.load_many(args.states, args.layers)
    reading_index, readings = load_readings(args.readings)
    merged = {k: v for folder in (args.data, *args.extra_data) for k, v in load_all_records(folder).items()}
    records = with_tool_questions(merged, load_tool_rules(args.tool_binary))
    decisions, silences = table_items(table, records, reading_index)
    excluded = held_out_readings(records.values(), load_rule_pairs(args.exclude_rules), reading_index)
    decisions = [item for item in decisions if _kept(item, args.modes, args.kinds) and item.reading not in excluded]
    silences = [item for item in silences if _kept(item, args.modes, []) and item.reading not in excluded]
    train_rows = sorted({item.row for item in _by_split(decisions + silences, "train")})
    settings = TrainSettings(epochs=args.epochs, rows_per_batch=args.rows_per_batch)
    channels = channels_from(table.states, train_rows, args.channel_samples, settings.seed).to(device)
    states = table.states.to(device)
    readings = readings.to(device)
    architecture = {
        "hidden_size": int(states.shape[-1]),
        "layers": list(table.layers),
        "rank": args.rank,
        "facets": args.facets,
        "circuit_units": args.circuit_units,
    }
    compiler = build_compiler(architecture).to(device)
    outputs = compiler.output_blocks
    output_columns = [table.layers.index(block) for block in outputs]
    print(
        json.dumps({"layers": list(table.layers), "decisions": len(decisions), "silences": len(silences)}), flush=True
    )

    train_compiler(
        compiler,
        readings,
        states,
        channels,
        _by_split(decisions, "train"),
        _by_split(silences, "train"),
        settings,
        log=lambda entry: print(json.dumps(entry), flush=True),
    )
    calibration = _by_split(decisions, CALIBRATION)
    cal_logits, cal_masks = predict_logits(compiler, readings, states, channels, calibration)
    temperatures = layer_temperatures(cal_logits, cal_masks, calibration)
    report: dict[str, Any] = {"layers": list(outputs), "temperatures": temperatures, "splits": {}}
    for split in EVALUATED_SPLITS:
        items = _by_split(decisions, split)
        if not items:
            continue
        logits, masks = predict_logits(compiler, readings, states, channels, items)
        quiet = _by_split(silences, split)
        quiet_logits, quiet_masks = predict_logits(compiler, readings, states, channels, quiet)
        rows = sorted({item.row for item in items + quiet})
        report["splits"][split] = layer_report(
            outputs,
            logits,
            masks,
            items,
            temperatures,
            silence=(quiet_logits, quiet_masks, quiet),
            states=table.states[rows][:, output_columns],
            channels=channels.cpu()[output_columns],
        )
        _print_summary(report["splits"][split])

    args.out.mkdir(parents=True, exist_ok=True)
    save_file({name: value.cpu() for name, value in compiler.state_dict().items()}, str(args.out / COMPILER_FILE))
    save_file({"channels": channels.cpu()}, str(args.out / CHANNELS_FILE))
    config = {
        **architecture,
        "output_layers": list(outputs),
        "modes": args.modes,
        "kinds": args.kinds,
        "reader_layer": json.loads((args.readings / "readings.json").read_text(encoding="utf-8"))["layer"],
        "temperatures": temperatures,
        "train": asdict(settings),
    }
    (args.out / CONFIG_FILE).write_text(json.dumps(config, indent=2), encoding="utf-8")
    (args.out / REPORT_FILE).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
