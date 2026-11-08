"""A tiny random Qwen3.5 checkpoint with the real key layout (sharded, with an MTP block), for tests and smoke runs."""

import json
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

from rein.neurons import constants as c

TINY_HIDDEN = 32
TINY_LAYER_TYPES = ("linear_attention", "full_attention", "linear_attention", "full_attention")
TINY_VISION = {
    "depth": 1,
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_heads": 2,
    "out_hidden_size": TINY_HIDDEN,
    "num_position_embeddings": 64,
}
MTP_SHARD = "model-mtp.safetensors"
SINGLE_FILE = "model.safetensors"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")


def tiny_text_settings(vocab_size: int) -> dict[str, object]:
    return {
        "vocab_size": vocab_size,
        "hidden_size": TINY_HIDDEN,
        "intermediate_size": 2 * TINY_HIDDEN,
        "num_hidden_layers": len(TINY_LAYER_TYPES),
        "layer_types": list(TINY_LAYER_TYPES),
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "linear_conv_kernel_dim": 4,
        "tie_word_embeddings": True,
    }


def _add_mtp_shard(directory: Path, seed: int) -> None:
    """The real checkpoint carries one MTP block with its own MLP; transformers ignores it on load."""
    generator = torch.Generator().manual_seed(seed)
    inner = 2 * TINY_HIDDEN
    tensors = {
        "mtp.layers.0.mlp.gate_proj.weight": torch.randn(inner, TINY_HIDDEN, generator=generator),
        "mtp.layers.0.mlp.up_proj.weight": torch.randn(inner, TINY_HIDDEN, generator=generator),
        "mtp.layers.0.mlp.down_proj.weight": torch.randn(TINY_HIDDEN, inner, generator=generator),
        "mtp.fc.weight": torch.randn(TINY_HIDDEN, 2 * TINY_HIDDEN, generator=generator),
    }
    save_file(tensors, str(directory / MTP_SHARD), metadata=c.SAFETENSORS_METADATA)
    index_path = directory / c.INDEX_FILE
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else _single_file_index(directory)
    index["weight_map"].update(dict.fromkeys(tensors, MTP_SHARD))
    index_path.write_text(json.dumps(index), encoding="utf-8")


def _single_file_index(directory: Path) -> dict[str, Any]:
    """An index for a checkpoint small enough to fit one file, so a second shard can join it."""
    with safe_open(str(directory / SINGLE_FILE), framework="pt") as reader:
        names = reader.keys()
    return {"metadata": {}, "weight_map": dict.fromkeys(names, SINGLE_FILE)}


def tiny_qwen_checkpoint(
    directory: Path,
    vocab_size: int,
    shard_size: str,
    seed: int = 0,
    tokenizer_from: Path | None = None,
    tokenizer_files: Iterable[str] = TOKENIZER_FILES,
) -> Path:
    config = Qwen3_5Config(
        text_config=tiny_text_settings(vocab_size), vision_config=TINY_VISION, tie_word_embeddings=True
    )
    torch.manual_seed(seed)
    Qwen3_5ForConditionalGeneration(config).save_pretrained(directory, max_shard_size=shard_size)
    _add_mtp_shard(directory, seed)
    if tokenizer_from is not None:
        for name in tokenizer_files:
            if (tokenizer_from / name).exists():
                shutil.copy2(tokenizer_from / name, directory / name)
    return directory
