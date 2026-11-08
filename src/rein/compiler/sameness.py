"""Group 2 of the acceptance numbers (DR-026): the generic checkpoint behaves as Qwen3.5-4B when nothing fires.

- the KL divergence of the checkpoint's next-token distribution from Qwen's, per token, on development conversations
  where no rule fires (each read with Qwen's own reply, under the rules block), and on general text;
- both models' perplexity on general text (WikiText-2, test split), in windows;
- how often the checkpoint's own neurons fire on those conversations, read from the MLP input of their block.
"""

import argparse
import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from rein.compiler.category_neurons import MANIFEST_FILE, NEURONS_FILE, VETO
from rein.compiler.category_probe import DEVELOPMENT, category_labels
from rein.genview.build import NO_THINKING, load_all_records
from rein.neurons.extract import load_views
from rein.neurons.units import threshold_gate

CONVERSATIONS = 300
WINDOW = 2048
SEED = 0
REPORT_FILE = "sameness.json"
NONE_FIRED, ONLY_VETO, ACTING = "no_neuron_fired", "only_the_veto_fired", "a_steer_or_force_fired"
TOOL_CALL = "<tool_call>"
CALL_OPENING, TEXT_OPENING = "reply_opens_a_call", "reply_opens_with_text"


def quiet_views(views: Path, data: Path, categories: Path, count: int) -> list[Any]:
    """Development views (non-thinking, with Qwen's reply) of records where no rule of any category fires."""
    records = load_all_records(data)
    mapping = json.loads(categories.read_text(encoding="utf-8"))
    picked = [
        view
        for view in load_views(views)
        if view.split == DEVELOPMENT
        and view.mode == NO_THINKING
        and view.record_id in mapping
        and not any(category_labels(records[view.record_id], mapping[view.record_id]).values())
    ]
    return random.Random(SEED).sample(picked, min(count, len(picked)))


def text_windows(path: Path, tokenizer: Any, window: int) -> list[list[int]]:
    ids = tokenizer(path.read_text(encoding="utf-8"), add_special_tokens=False)["input_ids"]
    return [ids[start : start + window] for start in range(0, len(ids) - window + 1, window)]


def gate_rows(neurons: Path) -> dict[str, torch.Tensor]:
    """Each detector's gate row: its pre-activation is at least 0 where the neuron fires."""
    manifest = json.loads((neurons / MANIFEST_FILE).read_text(encoding="utf-8"))
    tensors = load_file(str(neurons / NEURONS_FILE))
    channel = tensors["channel"].float()
    return {
        name: threshold_gate(
            tensors[f"{name}.direction"].float(), info["offset"], channel, info["temperature"], info["gate"]
        )
        for name, info in manifest["detectors"].items()
        if info["gate"] is not None
    }


class MlpInput:
    """Keeps the input of one block's MLP from the latest forward pass."""

    def __init__(self, block: Any) -> None:
        self.value: torch.Tensor | None = None
        self.handle = block.mlp.register_forward_pre_hook(self._keep)

    def _keep(self, module: Any, args: Any) -> None:
        self.value = args[0].detach()


@torch.no_grad()
def compare(
    original: Any, checkpoint: Any, sequences: Sequence[Sequence[int]], starts: Sequence[int]
) -> dict[str, list[float]]:
    """Per sequence: mean KL over every position and over the positions from `start` on (the reply)."""
    kl_all, kl_reply, kl_after, kl_prompt, loss_original, loss_checkpoint = [], [], [], [], [], []
    for ids, start in zip(sequences, starts, strict=True):
        tokens = torch.tensor([list(ids)], device=original.device)
        left = torch.log_softmax(original(input_ids=tokens).logits[0].float(), dim=-1)
        right = torch.log_softmax(checkpoint(input_ids=tokens).logits[0].float(), dim=-1)
        divergence = (left.exp() * (left - right)).sum(-1)
        kl_all.append(float(divergence.mean()))
        kl_reply.append(float(divergence[max(0, start - 1) :].mean()))
        kl_after.append(float(divergence[start:].mean()) if start < len(divergence) else 0.0)
        kl_prompt.append(float(divergence[: max(1, start - 1)].mean()))
        targets = tokens[0, 1:]
        loss_original.append(float(-left[:-1].gather(1, targets[:, None]).mean()))
        loss_checkpoint.append(float(-right[:-1].gather(1, targets[:, None]).mean()))
    return {
        "kl": kl_all,
        "kl_reply": kl_reply,
        "kl_reply_after_opening": kl_after,
        "kl_prompt": kl_prompt,
        "loss_original": loss_original,
        "loss_checkpoint": loss_checkpoint,
    }


@torch.no_grad()
def neuron_fires(
    checkpoint: Any,
    block: int,
    rows: dict[str, torch.Tensor],
    sequences: Sequence[Sequence[int]],
    starts: Sequence[int],
) -> tuple[dict[str, Any], list[list[str]]]:
    """Per detector, the share of conversations where it fires on the reply and on the prompt, the share of
    conversations where any steer or force neuron fires anywhere, and each conversation's firing neurons."""
    capture = MlpInput(checkpoint.model.layers[block])
    on_reply: dict[str, int] = dict.fromkeys(rows, 0)
    on_prompt: dict[str, int] = dict.fromkeys(rows, 0)
    any_acting = 0
    fired_names: list[list[str]] = []
    try:
        for ids, start in zip(sequences, starts, strict=True):
            checkpoint(input_ids=torch.tensor([list(ids)], device=checkpoint.device))
            assert capture.value is not None
            inputs = capture.value[0].float()
            acting = False
            names = []
            for name, row in rows.items():
                fired = inputs @ row.to(inputs.device) >= 0
                reply, prompt = bool(fired[start - 1 :].any()), bool(fired[: start - 1].any())
                on_reply[name] += reply
                on_prompt[name] += prompt
                acting |= (reply or prompt) and not name.endswith(f":{VETO}")
                if reply or prompt:
                    names.append(name)
            any_acting += acting
            fired_names.append(names)
    finally:
        capture.handle.remove()
    count = max(1, len(sequences))
    report = {
        "conversations": len(sequences),
        "any_steer_or_force": any_acting / count,
        "on_reply": {name: value / count for name, value in on_reply.items()},
        "on_prompt": {name: value / count for name, value in on_prompt.items()},
    }
    return report, fired_names


def by_firing(kl: Sequence[float], fired: Sequence[Sequence[str]]) -> dict[str, Any]:
    """The KL of the sequences where no neuron fired, where only the veto fired, and where a steer or force fired."""
    groups: dict[str, list[float]] = {NONE_FIRED: [], ONLY_VETO: [], ACTING: []}
    for value, names in zip(kl, fired, strict=True):
        if not names:
            groups[NONE_FIRED].append(value)
        elif all(name.endswith(f":{VETO}") for name in names):
            groups[ONLY_VETO].append(value)
        else:
            groups[ACTING].append(value)
    return {name: {"count": len(values), **_summary(values)} for name, values in groups.items()}


def by_opening(
    kl: Sequence[float], sequences: Sequence[Sequence[int]], starts: Sequence[int], call_id: int
) -> dict[str, Any]:
    """The reply KL split by how Qwen's reply opens: with a tool call or with text."""
    groups: dict[str, list[float]] = {CALL_OPENING: [], TEXT_OPENING: []}
    for value, ids, start in zip(kl, sequences, starts, strict=True):
        opens_a_call = start < len(ids) and ids[start] == call_id
        groups[CALL_OPENING if opens_a_call else TEXT_OPENING].append(value)
    return {name: {"count": len(values), **_summary(values)} for name, values in groups.items() if values}


def _summary(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean": sum(ordered) / max(1, len(ordered)),
        "median": ordered[len(ordered) // 2] if ordered else 0.0,
        "max": ordered[-1] if ordered else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Does the checkpoint behave as Qwen3.5-4B when nothing fires?")
    parser.add_argument("--original", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--neurons", type=Path, default=None, help="the folder `rein.compiler.category_neurons` wrote")
    parser.add_argument("--views", type=Path, required=True, help="block door views with Qwen's own replies")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--categories", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True, help="general text, for perplexity")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    def load(name: str | Path) -> Any:
        return AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda").eval()

    original, checkpoint = load(args.original), load(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.original)
    views = quiet_views(args.views, args.data, args.categories, CONVERSATIONS)
    sequences, starts = [view.ids for view in views], [view.prompt_length for view in views]
    chats = compare(original, checkpoint, sequences, starts)
    windows = text_windows(args.text, tokenizer, WINDOW)
    text = compare(original, checkpoint, windows, [1] * len(windows))
    call_id = tokenizer.convert_tokens_to_ids(TOOL_CALL)
    assert isinstance(call_id, int)
    report: dict[str, Any] = {
        "conversations": {
            "count": len(sequences),
            "kl_per_token": _summary(chats["kl"]),
            "kl_per_reply_token": _summary(chats["kl_reply"]),
            "kl_per_reply_token_by_opening": by_opening(chats["kl_reply"], sequences, starts, call_id),
            "kl_per_prompt_token": _summary(chats["kl_prompt"]),
            "kl_after_the_opening_by_opening": by_opening(chats["kl_reply_after_opening"], sequences, starts, call_id),
        },
        "general_text": {
            "windows": len(windows),
            "kl_per_token": _summary(text["kl"]),
            "perplexity_original": float(torch.tensor(text["loss_original"]).mean().exp()),
            "perplexity_checkpoint": float(torch.tensor(text["loss_checkpoint"]).mean().exp()),
        },
    }
    if args.neurons is not None:
        manifest = json.loads((args.neurons / MANIFEST_FILE).read_text(encoding="utf-8"))
        rows = gate_rows(args.neurons)
        chat_fires, chat_fired = neuron_fires(checkpoint, manifest["block"], rows, sequences, starts)
        text_fires, text_fired = neuron_fires(checkpoint, manifest["block"], rows, windows, [1] * len(windows))
        report["fires_on_quiet_conversations"] = chat_fires
        report["fires_on_general_text"] = text_fires
        report["reply_kl_by_firing"] = by_firing(chats["kl_reply"], chat_fired)
        report["general_text_kl_by_firing"] = by_firing(text["kl"], text_fired)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / REPORT_FILE).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
