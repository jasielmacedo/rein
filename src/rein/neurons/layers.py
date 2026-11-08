"""Inserts a group of new decoder layers into a Qwen3.5 checkpoint (block expansion), the place where a trained rule
module lives (DR-027 follow-up, option A).

The new layers are copies of the group just before them, with every output projection at zero, so each one adds
nothing to the residual stream: the checkpoint computes exactly what Qwen does until they are trained. A group as long
as `full_attention_interval` keeps llama.cpp's layer pattern, which marks a layer as full attention when its 1-based
index is a multiple of the interval.
"""

import copy
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from rein.neurons import constants as c
from rein.neurons.checkpoint import _copy_other_files, shard_names

LAYER_KEY = re.compile(r"^(?P<prefix>model\.language_model\.layers\.)(?P<layer>\d+)(?P<rest>\..+)$")
ZEROED_OUTPUTS = (".linear_attn.out_proj.weight", ".self_attn.o_proj.weight", ".mlp.down_proj.weight")
INTERVAL = "full_attention_interval"
DEFAULT_INTERVAL = 4
LAYER_TYPES = "layer_types"
MLP_ONLY_LAYERS = "mlp_only_layers"
MANIFEST_FILE = "rein_layers.json"


@dataclass(frozen=True)
class Insertion:
    at: int
    group: int
    layers_before: int

    @property
    def layers_after(self) -> int:
        return self.layers_before + self.group

    @property
    def new_layers(self) -> range:
        return range(self.at, self.at + self.group)

    def moved(self, layer: int) -> int:
        """Where an original layer ends up."""
        return layer + self.group if layer >= self.at else layer


def _text(config: dict[str, Any]) -> dict[str, Any]:
    nested = config.get(c.TEXT_CONFIG)
    return nested if isinstance(nested, dict) else config


def plan_insertion(config: Mapping[str, Any], at: int, group: int | None = None) -> Insertion:
    text = _text(dict(config))
    size = group or int(text.get(INTERVAL, DEFAULT_INTERVAL))
    layers = int(text[c.NUM_HIDDEN_LAYERS])
    if not size <= at <= layers or at % size:
        raise ValueError(f"insert at a multiple of {size} between {size} and {layers}, got {at}")
    return Insertion(at=at, group=size, layers_before=layers)


def inserted_config(config: Mapping[str, Any], insertion: Insertion) -> dict[str, Any]:
    updated = copy.deepcopy(dict(config))
    text = _text(updated)
    text[c.NUM_HIDDEN_LAYERS] = insertion.layers_after
    types = list(text.get(LAYER_TYPES) or [])
    if types:
        copied = types[insertion.at - insertion.group : insertion.at]
        text[LAYER_TYPES] = types[: insertion.at] + copied + types[insertion.at :]
    if text.get(MLP_ONLY_LAYERS):
        text[MLP_ONLY_LAYERS] = [insertion.moved(layer) for layer in text[MLP_ONLY_LAYERS]]
    return updated


def _renamed(key: str, layer: int) -> str:
    match = LAYER_KEY.match(key)
    assert match is not None
    return f"{match['prefix']}{layer}{match['rest']}"


def insert_layers(source: Path, target: Path, at: int, group: int | None = None) -> Insertion:
    """Copies the checkpoint with a group of zero-output layers at `at`; every original tensor is kept bit for bit,
    under its new layer number when it sits after the insertion."""
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"{target} is not empty")
    config = json.loads((source / c.CONFIG_FILE).read_text(encoding="utf-8"))
    insertion = plan_insertion(config, at, group)
    copied = range(insertion.at - insertion.group, insertion.at)
    target.mkdir(parents=True, exist_ok=True)
    weight_map: dict[str, str] = {}
    total_bytes = 0
    for name in shard_names(source):
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(str(source / name), framework="pt") as reader:
            metadata = reader.metadata() or c.SAFETENSORS_METADATA
            names = reader.keys()
            for key in names:
                tensor = reader.get_tensor(key)
                match = LAYER_KEY.match(key)
                if match is None:
                    tensors[key] = tensor
                    continue
                layer = int(match["layer"])
                tensors[_renamed(key, insertion.moved(layer))] = tensor
                if layer in copied:
                    fresh = torch.zeros_like(tensor) if key.endswith(ZEROED_OUTPUTS) else tensor.clone()
                    tensors[_renamed(key, layer + insertion.group)] = fresh
        for key, tensor in tensors.items():
            weight_map[key] = name
            total_bytes += tensor.numel() * tensor.element_size()
        save_file(tensors, str(target / name), metadata=metadata)
    _copy_other_files(source, target)
    (target / c.CONFIG_FILE).write_text(
        json.dumps(inserted_config(config, insertion), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if (source / c.INDEX_FILE).exists():
        index = {"metadata": {"total_size": total_bytes}, "weight_map": dict(sorted(weight_map.items()))}
        (target / c.INDEX_FILE).write_text(json.dumps(index, indent=2), encoding="utf-8")
    manifest = {"inserted_at": insertion.at, "group": insertion.group, "copied_from": list(copied)}
    (target / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return insertion
