"""Writes a soft-token run into a plain copy of its base checkpoint: the trained rows into the embedding table's spare
rows, the soft tokens into the tokenizer, every other byte as it was. The copy converts to GGUF like any Qwen3.5
checkpoint and runs the chat exactly as the base does."""

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from rein.env import load_env, optional
from rein.model.run_meta import CONFIG_FILE, SOFT_FILE, read_meta, resolve
from rein.model.soft import add_soft_tokens, stored_rows

EMBEDDING_SUFFIX = "embed_tokens.weight"
INDEX_FILE = "model.safetensors.index.json"
SINGLE_FILE = "model.safetensors"
SHARD_SUFFIX = ".safetensors"
SOFT_META_FILE = "rein_soft.json"
LM_HEAD_KEY = "lm_head.weight"
CONFIG_JSON = "config.json"
TIE_KEY = "tie_word_embeddings"
TEXT_CONFIG = "text_config"


def embedding_shards(base: Path) -> dict[str, str]:
    """Shard file name to the embedding key it holds."""
    index = base / INDEX_FILE
    if index.exists():
        weight_map: dict[str, str] = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        pairs = [(shard, key) for key, shard in weight_map.items() if key.endswith(EMBEDDING_SUFFIX)]
    else:
        with safe_open(base / SINGLE_FILE, "pt") as handle:
            names = list(handle.keys())
        pairs = [(SINGLE_FILE, key) for key in names if key.endswith(EMBEDDING_SUFFIX)]
    if len(pairs) != 1:
        raise ValueError(f"expected one embedding table in {base}, found {[key for _, key in pairs]}")
    return dict(pairs)


def write_rows(source: Path, target: Path, key: str, ids: list[int], rows: torch.Tensor, untie: bool) -> int:
    """Writes the soft rows into the table; with `untie`, the untouched table also becomes the output head, so the
    chat's next-token logits stay exactly the base's. Returns the bytes the head added."""
    with safe_open(source, "pt") as handle:
        metadata = handle.metadata()
    tensors = load_file(source)
    table = tensors[key]
    if max(ids) >= table.shape[0]:
        raise ValueError(f"soft token id {max(ids)} is beyond the embedding table ({table.shape[0]} rows)")
    added = 0
    if untie:
        tensors[LM_HEAD_KEY] = table.clone()
        added = table.numel() * table.element_size()
    table[ids] = rows.to(table.dtype)
    save_file(tensors, target, metadata=metadata)
    return added


def untie_config(out: Path, shard: str, head_bytes: int) -> None:
    """Declares the separate output head in the config and the weight index."""
    config_path = out / CONFIG_JSON
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config[TIE_KEY] = False
    if isinstance(config.get(TEXT_CONFIG), dict):
        config[TEXT_CONFIG][TIE_KEY] = False
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    index_path = out / INDEX_FILE
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        index["weight_map"][LM_HEAD_KEY] = shard
        if "total_size" in index.get("metadata", {}):
            index["metadata"]["total_size"] += head_bytes
        index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")


def merge_soft_run(
    run: Path, out: Path, tokenizer_source: Path | str | None = None, untie: bool = False
) -> dict[str, Any]:
    meta = read_meta(run)
    counts = meta.get("soft_tokens") or {}
    if not (counts.get("front") or counts.get("tail")):
        raise ValueError(f"{run} has no soft tokens to merge")
    token = optional(load_env(), "HF_TOKEN")
    base = resolve(meta["base_repo"], token)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source or base, revision=meta.get("base_revision"), token=token)
    front, tail = add_soft_tokens(tokenizer, int(counts["front"]), int(counts["tail"]))
    ids = list(front + tail)
    rows = stored_rows(load_file(run / SOFT_FILE))
    if rows.shape[0] != len(ids):
        raise ValueError(f"{run} holds {rows.shape[0]} soft rows for {len(ids)} soft tokens")

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shards = embedding_shards(base)
    for item in base.iterdir():
        if not item.is_file() or item.name.endswith(SHARD_SUFFIX):
            continue
        shutil.copy(item, out / item.name)
    ((shard, key),) = shards.items()
    for item in base.glob(f"*{SHARD_SUFFIX}"):
        if item.name == shard:
            head_bytes = write_rows(item, out / item.name, key, ids, rows, untie)
        else:
            shutil.copy(item, out / item.name)
    if untie:
        untie_config(out, shard, head_bytes)
    tokenizer.save_pretrained(out)
    report = {
        "base": str(base),
        "run": run.name,
        "soft_tokens": counts,
        "ids": ids,
        "embedding": key,
        "untied_head": untie,
        "readout_layer": meta.get("readout_layer"),
        "temperature": meta.get("temperature"),
    }
    (out / SOFT_META_FILE).write_text(json.dumps(report, indent=2), encoding="utf-8")
    shutil.copy(run / CONFIG_FILE, out / CONFIG_FILE)
    return report


def differing_tensors(base: Path, out: Path, ids: list[int]) -> list[str]:
    """Names of tensors that differ between the base and the merged copy, outside the soft rows. An untied head
    counts as unchanged when it equals the base's whole embedding table."""
    differences = []
    for item in sorted(base.glob(f"*{SHARD_SUFFIX}")):
        before = load_file(item)
        after = load_file(out / item.name)
        head = after.pop(LM_HEAD_KEY, None)
        if head is not None:
            embedding = next(tensor for key, tensor in before.items() if key.endswith(EMBEDDING_SUFFIX))
            if not torch.equal(head, embedding):
                differences.append(LM_HEAD_KEY)
        if before.keys() != after.keys():
            differences.append(f"{item.name}: different tensor names")
            continue
        for key, tensor in before.items():
            other = after[key]
            if key.endswith(EMBEDDING_SUFFIX):
                keep = torch.ones(tensor.shape[0], dtype=torch.bool)
                keep[ids] = False
                tensor, other = tensor[keep], other[keep]
            if tensor.shape != other.shape or tensor.dtype != other.dtype or not torch.equal(tensor, other):
                differences.append(key)
    return differences


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a soft-token run into a plain copy of its base checkpoint.")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokenizer", default=None, help="tokenizer source when the base has none")
    parser.add_argument(
        "--untie", action="store_true", help="give the copy its own output head, the base's table untouched"
    )
    parser.add_argument("--verify", action="store_true", help="check every other tensor is identical to the base")
    args = parser.parse_args()
    report = merge_soft_run(args.run, args.out, args.tokenizer, args.untie)
    if args.verify:
        report["differing_tensors"] = differing_tensors(Path(report["base"]), args.out, report["ids"])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
