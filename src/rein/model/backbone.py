from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn
from transformers import AutoModelForCausalLM
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

if TYPE_CHECKING:
    from peft import PeftModel

LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass(frozen=True)
class LoraSettings:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    targets: tuple[str, ...] = LORA_TARGETS


def load_text_backbone(
    repo: str, revision: str | None = None, dtype: torch.dtype = torch.bfloat16, token: str | None = None
) -> Qwen3_5TextModel:
    """Text decoder of a Qwen3.5 checkpoint, without the vision tower and without the lm_head."""
    causal_lm = AutoModelForCausalLM.from_pretrained(repo, revision=revision, dtype=dtype, token=token)
    backbone = causal_lm.model
    if not isinstance(backbone, Qwen3_5TextModel):
        raise TypeError(f"{repo} did not load a Qwen3.5 text model: {type(backbone).__name__}")
    del causal_lm
    return backbone


def freeze(backbone: Qwen3_5TextModel) -> Qwen3_5TextModel:
    for param in backbone.parameters():
        param.requires_grad_(False)
    return backbone


def attach_lora(backbone: Qwen3_5TextModel, settings: LoraSettings) -> "PeftModel":
    """Injects LoRA in place; the returned wrapper is only used to save, load and merge the adapter."""
    from peft import LoraConfig, get_peft_model

    freeze(backbone)
    config = LoraConfig(
        r=settings.rank,
        lora_alpha=settings.alpha,
        lora_dropout=settings.dropout,
        target_modules=list(settings.targets),
        bias="none",
    )
    return get_peft_model(backbone, config)


def truncate_backbone(backbone: Qwen3_5TextModel, readout_layer: int | None) -> Qwen3_5TextModel:
    """Keeps the first `readout_layer` decoder blocks: the head reads a middle layer and later blocks never run."""
    total = len(backbone.layers)
    if readout_layer is None or readout_layer == total:
        return backbone
    if not 1 <= readout_layer < total:
        raise ValueError(f"readout_layer must be between 1 and {total}, got {readout_layer}")
    backbone.layers = nn.ModuleList(list(backbone.layers)[:readout_layer])
    backbone.config.num_hidden_layers = readout_layer
    backbone.config.layer_types = list(backbone.config.layer_types or [])[:readout_layer]
    return backbone


def placed(backbone: Qwen3_5TextModel, device: torch.device) -> Qwen3_5TextModel:
    """Moves a frozen backbone to `device` in evaluation mode."""
    nn.Module.to(backbone, device)
    return backbone.eval()
