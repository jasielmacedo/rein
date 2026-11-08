"""Builds the generic checkpoint (DR-026): Qwen3.5-4B with the category neurons in the added slots of their block and
the rules chat template, so that any runtime that loads Qwen3.5 serves it unchanged and every agent passes its own
rules in the request.

Each detector becomes a step pair (`rein.sheets.compile.gated_pair`) that writes its action once its calibrated
probability passes its gate: a steer detector writes its category's behavior vector (or the vector pooled over all
categories when its own is missing), the force detector `<tool_call>`'s direction, the veto detector the opposite.
"""

import argparse
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from rein.compiler.category_neurons import FORCE, MANIFEST_FILE, NEURONS_FILE, STEER, VETO
from rein.genview import constants as gc
from rein.neurons.checkpoint import expand_checkpoint
from rein.neurons.units import Neuron
from rein.sheets.compile import aligned, gated_pair, placed, tool_call_direction
from rein.sheets.steer_gpu import POOLED

CHAT_TEMPLATE_FILE = "chat_template.jinja"
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"
CHAT_TEMPLATE_KEY = "chat_template"
STRENGTH_SEPARATOR = "="


def category_neurons(
    manifest: Mapping[str, Any],
    tensors: Mapping[str, torch.Tensor],
    vectors: Mapping[str, torch.Tensor],
    steer: Mapping[str, float],
    tool: tuple[torch.Tensor, float, float],
) -> tuple[list[tuple[str, Neuron]], list[str]]:
    """The step pairs of every detector that has a gate, each with its role; and the detectors left out."""
    channel = tensors["channel"].float()
    direction, force, veto = tool
    unit = direction / direction.norm()
    neurons: list[tuple[str, Neuron]] = []
    skipped = []
    for name, info in manifest["detectors"].items():
        category, action = info["category"], info["action"]
        if info["gate"] is None or (action == STEER and category not in steer):
            skipped.append(name)
            continue
        if action == STEER:
            write = steer[category] * vectors.get(category, vectors[POOLED]).float()
        else:
            write = {FORCE: force * unit, VETO: -veto * unit}[action]
        pair = gated_pair(
            tensors[f"{name}.direction"].float(), info["offset"], channel, info["temperature"], info["gate"], write
        )
        neurons += [(f"{action}:{category}", neuron) for neuron in pair]
    return neurons, skipped


def install_template(template: Path, checkpoint: Path) -> None:
    """The rules template replaces Qwen's in the template file and in the tokenizer config (converters read either)."""
    text = template.read_text(encoding="utf-8")
    shutil.copyfile(template, checkpoint / CHAT_TEMPLATE_FILE)
    config_path = checkpoint / TOKENIZER_CONFIG_FILE
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if CHAT_TEMPLATE_KEY in config:
        config[CHAT_TEMPLATE_KEY] = text
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile the generic category neurons into a Qwen3.5 checkpoint.")
    parser.add_argument("--model", type=Path, required=True, help="the original Qwen3.5-4B snapshot")
    parser.add_argument("--neurons", type=Path, required=True, help="the folder `rein.compiler.category_neurons` wrote")
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--steer", nargs="+", required=True, help="category=strength, as the sweep chose")
    parser.add_argument("--force", type=float, required=True)
    parser.add_argument("--veto", type=float, required=True)
    parser.add_argument("--template", type=Path, default=Path(gc.RULES_TEMPLATE))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.neurons / MANIFEST_FILE).read_text(encoding="utf-8"))
    tensors = load_file(str(args.neurons / NEURONS_FILE))
    vectors = load_file(str(args.vectors))
    steer = {k: float(v) for k, v in (item.split(STRENGTH_SEPARATOR) for item in args.steer)}
    neurons, skipped = category_neurons(
        manifest, tensors, vectors, steer, (tool_call_direction(args.model), args.force, args.veto)
    )
    block = manifest["block"]
    expansion = expand_checkpoint(
        args.model,
        args.out,
        aligned(len(neurons)),
        placed(neurons, block),
        manifest={"detectors": manifest["detectors"], "steer": steer, "force": args.force, "veto": args.veto},
    )
    install_template(args.template, args.out)
    print(
        json.dumps(
            {
                "neurons": len(neurons),
                "added": expansion.added,
                "block": block,
                "skipped": skipped,
                "out": str(args.out),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
