"""Compiles fitted sheets into a standard Qwen3.5 checkpoint: each question's neurons become step pairs in the added
slots of their block, writing the rule's action into the residual stream (DR-024, DR-025).

- steer: a behavior vector, while the rule fires (the reply opening and every reply token after it);
- tool: `<tool_call>`'s direction when the any-tool question is sure (force), and against it when it is sure there is
  no call (veto);
- call: `<tool_call>`'s direction where the reply narrates an action without performing it.
Everything else in the checkpoint stays bit for bit as Qwen3.5-4B shipped it.
"""

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from rein.compiler.constants import NULL_OPTIONS
from rein.neurons import constants as nc
from rein.neurons.checkpoint import PlacedNeuron, expand_checkpoint
from rein.neurons.units import Neuron, step_pair, threshold_gate
from rein.sheets.fit import MANIFEST_FILE, NEURONS_FILE

STEER_GATE = 0.85
FORCE_GATE = 0.8
VETO_GATE = 0.5
CALL_GATE = 0.85
STEP_GAIN = 12.0
STEP_WIDTH = 2.0
TOOL_CALL = "<tool_call>"


@dataclass(frozen=True)
class Strengths:
    """How hard each action writes, in units of the residual stream after the block (or of the steer vector)."""

    force: float
    veto: float
    steer: Mapping[str, float]


@dataclass(frozen=True)
class FittedQuestion:
    detectors: torch.Tensor
    offsets: torch.Tensor
    option_keys: tuple[str, ...]
    action: str
    rule: str


def load_fitted(directory: Path) -> tuple[dict[str, Any], dict[str, FittedQuestion], torch.Tensor]:
    manifest = json.loads((directory / MANIFEST_FILE).read_text(encoding="utf-8"))
    tensors = load_file(str(directory / NEURONS_FILE))
    questions = {
        question_id: FittedQuestion(
            tensors[f"{question_id}.detectors"].float(),
            tensors[f"{question_id}.offsets"].float(),
            tuple(info["option_keys"]),
            info["action"],
            info["rule"],
        )
        for question_id, info in manifest["questions"].items()
    }
    return manifest, questions, tensors["channel"].float()


def gated_pair(
    detector: torch.Tensor, offset: float, channel: torch.Tensor, temperature: float, gate: float, write: torch.Tensor
) -> tuple[Neuron, Neuron]:
    """A bounded soft step: about 0 below the gate, `write` once the calibrated probability passes it."""
    row = threshold_gate(detector, offset, channel, temperature, gate)
    return step_pair(row, channel, STEP_GAIN, STEP_WIDTH, 1.0 / STEP_WIDTH, write)


def _versus(question: FittedQuestion, option: int, against: int) -> tuple[torch.Tensor, float]:
    return question.detectors[option] - question.detectors[against], float(
        question.offsets[option] - question.offsets[against]
    )


def question_neurons(
    question: FittedQuestion,
    channel: torch.Tensor,
    temperature: float,
    tool_direction: torch.Tensor,
    strengths: Strengths,
    vectors: Mapping[str, torch.Tensor],
    steer_names: Mapping[str, str],
) -> list[tuple[str, Neuron]]:
    """The neurons of one fitted question, each with the role it plays."""
    null = next(i for i, key in enumerate(question.option_keys) if key in NULL_OPTIONS)
    firing = [i for i in range(len(question.option_keys)) if i != null]
    unit = tool_direction / tool_direction.norm()
    neurons: list[tuple[str, Neuron]] = []
    if question.action == "steer":
        name = steer_names[question.rule]
        detector, offset = _versus(question, firing[0], null)
        pair = gated_pair(detector, offset, channel, temperature, STEER_GATE, strengths.steer[name] * vectors[name])
        neurons += [(f"steer:{question.rule}", neuron) for neuron in pair]
    elif question.action == "tool":
        detector, offset = _versus(question, firing[0], null)
        neurons += [
            (f"force:{question.rule}", n)
            for n in gated_pair(detector, offset, channel, temperature, FORCE_GATE, strengths.force * unit)
        ]
        neurons += [
            (f"veto:{question.rule}", n)
            for n in gated_pair(-detector, -offset, channel, temperature, 1.0 - VETO_GATE, -strengths.veto * unit)
        ]
    else:
        for option in firing:
            detector, offset = _versus(question, option, null)
            pair = gated_pair(detector, offset, channel, temperature, CALL_GATE, strengths.force * unit)
            neurons += [(f"call:{question.rule}:{question.option_keys[option]}", n) for n in pair]
    return neurons


def placed(neurons: Sequence[tuple[str, Neuron]], block: int) -> list[PlacedNeuron]:
    return [
        PlacedNeuron(block=block, slot=slot, neuron=neuron, role=role) for slot, (role, neuron) in enumerate(neurons)
    ]


def aligned(count: int) -> int:
    return max(nc.NEURON_ALIGN, math.ceil(count / nc.NEURON_ALIGN) * nc.NEURON_ALIGN)


def tool_call_direction(model: Path) -> torch.Tensor:
    """`<tool_call>`'s direction in the residual stream, from the checkpoint's own embedding and final norm."""
    from transformers import AutoTokenizer

    from rein.neurons.checkpoint import shard_names
    from rein.neurons.units import token_direction

    token = AutoTokenizer.from_pretrained(model).convert_tokens_to_ids(TOOL_CALL)
    if not isinstance(token, int):
        raise ValueError(f"{TOOL_CALL} is not a single token")
    tensors: dict[str, torch.Tensor] = {}
    for shard in shard_names(model):
        loaded = load_file(str(model / shard))
        tensors.update(
            {
                k: v
                for k, v in loaded.items()
                if k.endswith("embed_tokens.weight")
                or k.endswith("language_model.norm.weight")
                or k == "model.norm.weight"
            }
        )
    embedding = next(v for k, v in tensors.items() if k.endswith("embed_tokens.weight"))
    final_norm = next(v for k, v in tensors.items() if k.endswith("norm.weight"))
    return token_direction(embedding[token], final_norm)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile fitted end-to-end sheets into an expanded Qwen3.5 checkpoint."
    )
    parser.add_argument("--model", type=Path, required=True, help="the original Qwen3.5-4B snapshot")
    parser.add_argument("--fits", type=Path, nargs="+", required=True, help="one per-sheet fit folder per sheet")
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--steer", nargs="+", required=True, help="vector=strength, the strength the sweep chose")
    parser.add_argument("--force", type=float, required=True)
    parser.add_argument("--veto", type=float, required=True)
    parser.add_argument("--steer-names", type=Path, required=True, help="JSON rule id to vector name")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    vectors = {name: tensor.float() for name, tensor in load_file(str(args.vectors)).items()}
    strengths = Strengths(args.force, args.veto, {k: float(v) for k, v in (item.split("=") for item in args.steer)})
    steer_names = json.loads(args.steer_names.read_text(encoding="utf-8"))
    direction = tool_call_direction(args.model)
    all_neurons: list[tuple[str, Neuron]] = []
    block = None
    for folder in args.fits:
        manifest, questions, channel = load_fitted(folder)
        block = manifest["block"]
        for question in questions.values():
            all_neurons += question_neurons(
                question, channel, manifest["temperature"], direction, strengths, vectors, steer_names
            )
    assert block is not None
    expansion = expand_checkpoint(
        args.model,
        args.out,
        aligned(len(all_neurons)),
        placed(all_neurons, block),
        manifest={
            "sheets": [str(folder) for folder in args.fits],
            "force": args.force,
            "veto": args.veto,
            "steer": strengths.steer,
        },
    )
    print(
        json.dumps({"neurons": len(all_neurons), "added": expansion.added, "block": block, "out": str(args.out)}),
        flush=True,
    )


if __name__ == "__main__":
    main()
