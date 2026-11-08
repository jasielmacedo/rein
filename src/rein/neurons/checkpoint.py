import copy
import json
import re
import shutil
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from rein.neurons import constants as c
from rein.neurons.units import Neuron

_MLP_KEY = re.compile(c.MLP_KEY_PATTERN)


@dataclass(frozen=True)
class PlacedNeuron:
    """A neuron written into one of the added slots of a decoder block (both 0-based)."""

    block: int
    slot: int
    neuron: Neuron
    role: str = ""


@dataclass(frozen=True)
class MlpKey:
    scope: str
    layer: int
    projection: str

    @property
    def is_mtp(self) -> bool:
        return self.scope == c.MTP_SCOPE


@dataclass(frozen=True)
class Expansion:
    base_size: int
    added: int
    hidden_size: int
    layers: int

    @property
    def size(self) -> int:
        return self.base_size + self.added


def parse_mlp_key(name: str) -> MlpKey | None:
    match = _MLP_KEY.match(name)
    if match is None:
        return None
    return MlpKey(match["scope"], int(match["layer"]), match["projection"])


def text_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = config.get(c.TEXT_CONFIG)
    return nested if isinstance(nested, Mapping) else config


def plan_expansion(config: Mapping[str, Any], added: int) -> Expansion:
    if added <= 0 or added % c.NEURON_ALIGN:
        raise ValueError(f"added neurons must be a positive multiple of {c.NEURON_ALIGN}, got {added}")
    text = text_config(config)
    return Expansion(
        base_size=int(text[c.INTERMEDIATE_SIZE]),
        added=added,
        hidden_size=int(text[c.HIDDEN_SIZE]),
        layers=int(text[c.NUM_HIDDEN_LAYERS]),
    )


def _validate(neurons: Sequence[PlacedNeuron], expansion: Expansion) -> None:
    seen: set[tuple[int, int]] = set()
    for placed in neurons:
        if not 0 <= placed.block < expansion.layers:
            raise ValueError(f"block {placed.block} is outside 0..{expansion.layers - 1}")
        if not 0 <= placed.slot < expansion.added:
            raise ValueError(f"slot {placed.slot} is outside 0..{expansion.added - 1}")
        if (placed.block, placed.slot) in seen:
            raise ValueError(f"two neurons share block {placed.block} slot {placed.slot}")
        seen.add((placed.block, placed.slot))
        for vector in (placed.neuron.gate, placed.neuron.up, placed.neuron.down):
            if tuple(vector.shape) != (expansion.hidden_size,):
                raise ValueError(f"neuron vectors must have shape ({expansion.hidden_size},)")
            if not torch.isfinite(vector).all():
                raise ValueError("neuron vectors must be finite")


def _expanded(
    tensor: torch.Tensor, key: MlpKey, expansion: Expansion, by_block: Mapping[int, Sequence[PlacedNeuron]]
) -> torch.Tensor:
    placed = () if key.is_mtp else by_block.get(key.layer, ())
    if key.projection == c.DOWN_PROJ:
        if tuple(tensor.shape) != (expansion.hidden_size, expansion.base_size):
            raise ValueError(f"unexpected down_proj shape {tuple(tensor.shape)}")
        extra = torch.zeros(expansion.hidden_size, expansion.added, dtype=tensor.dtype)
        for neuron in placed:
            extra[:, neuron.slot] = neuron.neuron.down.to(tensor.dtype)
        return torch.cat([tensor, extra], dim=1)
    if tuple(tensor.shape) != (expansion.base_size, expansion.hidden_size):
        raise ValueError(f"unexpected {key.projection} shape {tuple(tensor.shape)}")
    extra = torch.zeros(expansion.added, expansion.hidden_size, dtype=tensor.dtype)
    for neuron in placed:
        row = neuron.neuron.gate if key.projection == c.GATE_PROJ else neuron.neuron.up
        extra[neuron.slot] = row.to(tensor.dtype)
    return torch.cat([tensor, extra], dim=0)


def shard_names(directory: Path) -> list[str]:
    index = directory / c.INDEX_FILE
    if index.exists():
        weight_map: dict[str, str] = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        return sorted(set(weight_map.values()))
    return sorted(path.name for path in directory.glob(f"*{c.SAFETENSORS_SUFFIX}"))


def _copy_other_files(source: Path, target: Path) -> None:
    skipped = {c.CONFIG_FILE, c.INDEX_FILE, c.MANIFEST_FILE}
    for path in source.iterdir():
        if path.is_file() and path.suffix != c.SAFETENSORS_SUFFIX and path.name not in skipped:
            shutil.copy2(path, target / path.name)


def _write_config(config: Mapping[str, Any], expansion: Expansion, target: Path) -> None:
    updated = copy.deepcopy(dict(config))
    holder = updated[c.TEXT_CONFIG] if isinstance(updated.get(c.TEXT_CONFIG), dict) else updated
    holder[c.INTERMEDIATE_SIZE] = expansion.size
    (target / c.CONFIG_FILE).write_text(json.dumps(updated, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_index(source: Path, target: Path, total_bytes: int) -> None:
    index = source / c.INDEX_FILE
    if not index.exists():
        return
    payload = json.loads(index.read_text(encoding="utf-8"))
    payload.setdefault("metadata", {})["total_size"] = total_bytes
    (target / c.INDEX_FILE).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_manifest(
    target: Path, expansion: Expansion, neurons: Sequence[PlacedNeuron], extra: Mapping[str, Any] | None
) -> None:
    payload = {
        "version": c.MANIFEST_VERSION,
        "base_intermediate_size": expansion.base_size,
        "added_per_block": expansion.added,
        "neurons": [{"block": n.block, "slot": n.slot, "role": n.role} for n in neurons],
        **(dict(extra) if extra else {}),
    }
    (target / c.MANIFEST_FILE).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def expand_checkpoint(
    source: Path,
    target: Path,
    added: int,
    neurons: Sequence[PlacedNeuron] = (),
    manifest: Mapping[str, Any] | None = None,
) -> Expansion:
    """Copies a Qwen3.5 checkpoint with `added` more MLP neurons in every block, MTP included.

    Every original tensor is copied unchanged; the new neurons are zero except the placed ones.
    """
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"{target} is not empty")
    config: dict[str, Any] = json.loads((source / c.CONFIG_FILE).read_text(encoding="utf-8"))
    expansion = plan_expansion(config, added)
    _validate(neurons, expansion)
    by_block: dict[int, list[PlacedNeuron]] = {}
    for placed in neurons:
        by_block.setdefault(placed.block, []).append(placed)
    target.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    for name in shard_names(source):
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(str(source / name), framework="pt") as reader:
            metadata = reader.metadata() or c.SAFETENSORS_METADATA
            names = reader.keys()
            for key in names:
                tensor = reader.get_tensor(key)
                mlp = parse_mlp_key(key)
                tensors[key] = tensor if mlp is None else _expanded(tensor, mlp, expansion, by_block)
                total_bytes += tensors[key].numel() * tensors[key].element_size()
        save_file(tensors, str(target / name), metadata=metadata)
    _copy_other_files(source, target)
    _write_config(config, expansion, target)
    _write_index(source, target, total_bytes)
    _write_manifest(target, expansion, neurons, manifest)
    return expansion


def _key_to_shard(directory: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for name in shard_names(directory):
        with safe_open(str(directory / name), framework="pt") as reader:
            mapping.update(dict.fromkeys(reader.keys(), name))
    return mapping


def original_mismatches(source: Path, target: Path) -> list[str]:
    """Names of original tensors that the target does not carry bit for bit (empty when the expansion is faithful)."""
    target_shards = _key_to_shard(target)
    mismatches: list[str] = []
    with ExitStack() as stack:
        readers = {
            name: stack.enter_context(safe_open(str(target / name), framework="pt"))
            for name in set(target_shards.values())
        }
        for name in shard_names(source):
            with safe_open(str(source / name), framework="pt") as reader:
                names = reader.keys()
                for key in names:
                    if key not in target_shards:
                        mismatches.append(key)
                        continue
                    original = reader.get_tensor(key)
                    copied = readers[target_shards[key]].get_tensor(key)
                    if parse_mlp_key(key) is not None:
                        copied = copied[tuple(slice(0, size) for size in original.shape)]
                    if copied.dtype != original.dtype or not torch.equal(copied, original):
                        mismatches.append(key)
    return mismatches
