import argparse
import json
import time
from collections.abc import Callable, Iterator, Sequence
from functools import partial
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from rein.genview.build import ViewRow
from rein.model.backbone import load_text_backbone, placed, truncate_backbone
from rein.model.kernels import require_fast_kernels

INDEX_FILE = "index.jsonl"
META_FILE = "extraction.json"
SHARD_PATTERN = "states-{:05d}.npy"
STATE_DTYPE = np.float16
DEFAULT_TOKEN_BUDGET = 32768
DEFAULT_SHARD_ROWS = 8192
LOG_EVERY_BATCHES = 20


MLP_INPUT = "mlp"
MLP_HIDDEN = "mlp.down_proj"


def layer_part(backbone: Qwen3_5TextModel, layer: int, part: str) -> nn.Module:
    module: nn.Module = backbone.layers[layer]
    for name in part.split("."):
        module = cast(nn.Module, getattr(module, name))
    return module


class MlpInputCapture:
    """Keeps the input of a part of chosen layers (the MLP by default, or its down projection, which sees the MLP's
    hidden units) at chosen token positions while the backbone runs; with `detach` off the kept states stay in the
    graph, so a loss on them trains the layers before."""

    def __init__(
        self, backbone: Qwen3_5TextModel, layers: Sequence[int], detach: bool = True, part: str = MLP_INPUT
    ) -> None:
        self.layers = tuple(layers)
        self.detach = detach
        self.captured: dict[int, torch.Tensor] = {}
        self._rows = torch.zeros(0, dtype=torch.long)
        self._columns = torch.zeros(0, dtype=torch.long)
        self._handles = [
            layer_part(backbone, layer, part).register_forward_pre_hook(partial(self._keep, layer))
            for layer in self.layers
        ]

    def aim(self, rows: torch.Tensor, columns: torch.Tensor) -> None:
        self._rows, self._columns = rows, columns
        self.captured = {}

    def _keep(self, layer: int, module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        kept = args[0][self._rows, self._columns]
        self.captured[layer] = kept.detach() if self.detach else kept

    def stacked(self) -> torch.Tensor:
        return torch.stack([self.captured[layer] for layer in self.layers], dim=1)

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()


def view_length(view: ViewRow) -> int:
    return len(view.ids)


def length_batches[Item](
    items: Sequence[Item], token_budget: int, length: Callable[[Item], int]
) -> Iterator[list[Item]]:
    """Items sorted by length and grouped so that a padded batch stays within the token budget."""
    batch: list[Item] = []
    longest = 0
    for item in sorted(items, key=length):
        widest = max(longest, length(item))
        if batch and widest * (len(batch) + 1) > token_budget:
            yield batch
            batch, widest = [], length(item)
        batch.append(item)
        longest = widest
    if batch:
        yield batch


def batches(views: Sequence[ViewRow], token_budget: int) -> Iterator[list[ViewRow]]:
    return length_batches(views, token_budget, view_length)


def padded_ids(sequences: Sequence[Sequence[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Token ids padded on the right and their attention mask; causal layers never let a real token see the padding."""
    width = max(len(sequence) for sequence in sequences)
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(sequences), width), dtype=torch.long)
    for row, sequence in enumerate(sequences):
        ids[row, : len(sequence)] = torch.tensor(list(sequence))
        mask[row, : len(sequence)] = 1
    return ids, mask


def padded(batch: Sequence[ViewRow], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    return padded_ids([view.ids for view in batch], pad_id)


def _targets(batch: Sequence[ViewRow]) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    rows, columns, index = [], [], []
    for row, view in enumerate(batch):
        for kind, tokens in view.positions.items():
            for token in tokens:
                rows.append(row)
                columns.append(token)
                index.append(
                    {
                        "view_id": view.view_id,
                        "record_id": view.record_id,
                        "split": view.split,
                        "mode": view.mode,
                        "decision": view.decision,
                        "kind": kind,
                        "token": token,
                    }
                )
    return torch.tensor(rows), torch.tensor(columns), index


class StateWriter:
    """Writes captured states as float16 shards of shape (rows, layers, hidden) plus one JSON line per row.

    Shards close only between batches, so every view sits whole in one shard and a restart can skip finished views.
    """

    def __init__(self, out_dir: Path, shard_rows: int) -> None:
        self.out_dir = out_dir
        self.shard_rows = shard_rows
        out_dir.mkdir(parents=True, exist_ok=True)
        self.shard = len(list(out_dir.glob("states-*.npy")))
        self._states: list[np.ndarray] = []
        self._index: list[dict[str, Any]] = []

    def done_views(self) -> set[str]:
        path = self.out_dir / INDEX_FILE
        if not path.exists():
            return set()
        with path.open(encoding="utf-8") as handle:
            return {json.loads(line)["view_id"] for line in handle if line.strip()}

    def add(self, states: torch.Tensor, index: list[dict[str, Any]]) -> None:
        self._states.append(states.to(torch.float32).cpu().numpy().astype(STATE_DTYPE))
        self._index.extend(index)
        if sum(len(chunk) for chunk in self._states) >= self.shard_rows:
            self.flush()

    def flush(self) -> None:
        if not self._states:
            return
        name = SHARD_PATTERN.format(self.shard)
        temporary = self.out_dir / f"{name}.partial"
        with temporary.open("wb") as handle:
            np.save(handle, np.concatenate(self._states))
        temporary.replace(self.out_dir / name)
        with (self.out_dir / INDEX_FILE).open("a", encoding="utf-8") as handle:
            for row, entry in enumerate(self._index):
                handle.write(json.dumps({**entry, "shard": name, "row": row}) + "\n")
        self.shard += 1
        self._states, self._index = [], []


@torch.no_grad()
def extract(
    backbone: Qwen3_5TextModel,
    views: Sequence[ViewRow],
    layers: Sequence[int],
    writer: StateWriter,
    pad_id: int,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> int:
    """Runs every unfinished view once and stores the MLP inputs of `layers` at its named positions."""
    done = writer.done_views()
    pending = [view for view in views if view.view_id not in done]
    device = next(backbone.parameters()).device
    capture = MlpInputCapture(backbone, layers)
    started, tokens = time.monotonic(), 0
    try:
        for number, batch in enumerate(batches(pending, token_budget), start=1):
            ids, mask = padded(batch, pad_id)
            rows, columns, index = _targets(batch)
            capture.aim(rows.to(device), columns.to(device))
            backbone(input_ids=ids.to(device), attention_mask=mask.to(device), use_cache=False)
            writer.add(capture.stacked(), index)
            tokens += int(mask.sum())
            if number % LOG_EVERY_BATCHES == 0:
                rate = tokens / max(time.monotonic() - started, 1e-9)
                print(json.dumps({"batches": number, "tokens": tokens, "tokens_per_s": round(rate)}), flush=True)
        writer.flush()
    finally:
        capture.close()
    return len(pending)


def extract_to(
    backbone: Qwen3_5TextModel,
    views: Sequence[ViewRow],
    layers: Sequence[int],
    out: Path,
    model: str,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    shard_rows: int = DEFAULT_SHARD_ROWS,
) -> dict[str, Any]:
    """Stores the states of `layers` for every view under `out`, with the table's description beside them."""
    writer = StateWriter(out, shard_rows)
    meta = {"model": model, "layers": list(layers), "hidden_size": backbone.config.hidden_size, "views": len(views)}
    (out / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    pad_id = int(backbone.config.pad_token_id or 0)
    return {"extracted_views": extract(backbone, views, layers, writer, pad_id, token_budget), **meta}


def load_views(path: Path) -> list[ViewRow]:
    with path.open(encoding="utf-8") as handle:
        return [ViewRow.from_json(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Store the MLP inputs Qwen3.5 computes at the positions of each view.")
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="*", default=None, help="0-based layers; all when omitted")
    parser.add_argument("--token-budget", type=int, default=DEFAULT_TOKEN_BUDGET)
    parser.add_argument("--shard-rows", type=int, default=DEFAULT_SHARD_ROWS)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(require_fast_kernels(), flush=True)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    backbone = load_text_backbone(args.model, dtype=dtype)
    layers = list(range(len(backbone.layers))) if not args.layers else sorted(set(args.layers))
    backbone = placed(truncate_backbone(backbone, max(layers) + 1), device)
    views = load_views(args.views)
    report = extract_to(backbone, views, layers, args.out, args.model, args.token_budget, args.shard_rows)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
