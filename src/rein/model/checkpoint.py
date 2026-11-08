import json
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel, set_peft_model_state_dict
from safetensors.torch import load_file, save_file
from torch import nn

from rein.model.run_meta import (  # noqa: F401  (re-exported for the modules that read runs through here)
    CONFIG_FILE,
    FORMAT_VERSION,
    MERGED_FORMAT_VERSION,
    SOFT_FILE,
    SUPPORTED_FORMATS,
    read_meta,
)
from rein.model.soft import SoftPrompt

HEAD_FILE = "head.safetensors"
ADAPTER_DIR = "adapter"


def save_checkpoint(
    directory: Path,
    peft_model: PeftModel | None,
    head: nn.Module,
    meta: dict[str, Any],
    soft: SoftPrompt | None = None,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    if peft_model is not None:
        peft_model.save_pretrained(str(directory / ADAPTER_DIR))
    save_head(directory, head)
    if soft is not None:
        tensors = {"weight": soft.weight.detach().cpu().contiguous()}
        if soft.basis.shape[0]:
            tensors["basis"] = soft.basis.detach().cpu().contiguous()
        save_file(tensors, directory / SOFT_FILE)
    payload = {"format_version": FORMAT_VERSION, **meta}
    (directory / CONFIG_FILE).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def save_head(directory: Path, head: nn.Module) -> None:
    save_file(
        {name: tensor.detach().cpu().contiguous() for name, tensor in head.state_dict().items()}, directory / HEAD_FILE
    )


def load_weights(
    directory: Path,
    peft_model: PeftModel | None,
    head: nn.Module,
    device: torch.device,
    soft: SoftPrompt | None = None,
) -> None:
    if peft_model is not None:
        adapter = load_file(directory / ADAPTER_DIR / "adapter_model.safetensors", device=str(device))
        unexpected = getattr(set_peft_model_state_dict(peft_model, adapter), "unexpected_keys", [])
        if unexpected:
            raise ValueError(f"adapter has unexpected keys: {unexpected[:5]}")
    head.load_state_dict(load_file(directory / HEAD_FILE, device=str(device)))
    if soft is not None:
        tensors = load_file(directory / SOFT_FILE, device=str(device))
        with torch.no_grad():
            soft.weight.copy_(tensors["weight"])
        if "basis" in tensors:
            soft.set_basis(tensors["basis"])
