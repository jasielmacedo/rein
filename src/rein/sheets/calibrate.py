"""How hard the compiled tool neurons must write from their block: the smallest push along `<tool_call>`'s direction
that makes it the next token at a reply opening (force), and the smallest pull that keeps it off where Qwen would open
a call on its own (veto). Strengths are measured in units of the residual stream's median size after the block."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from rein.neurons.extract import load_views
from rein.sheets.compile import TOOL_CALL, tool_call_direction
from rein.sheets.steer_gpu import SteerHook

RATIOS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
TARGET_RATE = 0.99


@torch.no_grad()
def next_tokens(model: Any, prompts: Sequence[Sequence[int]], layer: int, push: torch.Tensor | None) -> list[int]:
    handle = model.model.layers[layer].register_forward_hook(SteerHook(push, 1.0)) if push is not None else None
    try:
        return [
            int(model(input_ids=torch.tensor([list(ids)], device=model.device)).logits[0, -1].argmax())
            for ids in prompts
        ]
    finally:
        if handle is not None:
            handle.remove()


@torch.no_grad()
def residual_norm(model: Any, prompts: Sequence[Sequence[int]], layer: int) -> float:
    norms = [
        model(input_ids=torch.tensor([list(ids)], device=model.device), output_hidden_states=True)
        .hidden_states[layer + 1][0, -1]
        .float()
        .norm()
        for ids in prompts
    ]
    return float(torch.stack(norms).median())


def smallest(rates: dict[str, float]) -> float | None:
    passing = [float(ratio) for ratio, rate in rates.items() if rate >= TARGET_RATE]
    return min(passing) if passing else None


def calibrate(
    model: Any, prompts: Sequence[Sequence[int]], token: int, direction: torch.Tensor, layer: int
) -> dict[str, Any]:
    unit = (direction / direction.norm()).to(model.device)
    size = residual_norm(model, prompts, layer)
    natural = next_tokens(model, prompts, layer, None)
    opened = [ids for ids, top in zip(prompts, natural, strict=True) if top == token]
    force, veto = {}, {}
    for ratio in RATIOS:
        pushed = next_tokens(model, prompts, layer, ratio * size * unit)
        force[str(ratio)] = sum(top == token for top in pushed) / len(prompts)
        if opened:
            pulled = next_tokens(model, opened, layer, -ratio * size * unit)
            veto[str(ratio)] = sum(top != token for top in pulled) / len(opened)
    return {
        "prompts": len(prompts),
        "opened_by_qwen": len(opened),
        "residual_norm": size,
        "force": force,
        "veto": veto,
        "force_ratio": smallest(force),
        "veto_ratio": smallest(veto),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate the tool neurons' strengths from their block.")
    parser.add_argument("--views", type=Path, nargs="+", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()
    token = AutoTokenizer.from_pretrained(args.model).convert_tokens_to_ids(TOOL_CALL)
    if not isinstance(token, int):
        raise ValueError(f"{TOOL_CALL} is not a single token")
    direction = tool_call_direction(Path(snapshot_download(args.model)))
    views = [view for path in args.views for view in load_views(path) if view.mode == "nothink"]
    prompts = [list(view.ids[: view.prompt_length]) for view in views]
    report = calibrate(model, prompts, token, direction, args.layer)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
