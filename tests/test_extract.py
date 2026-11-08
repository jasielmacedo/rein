from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import Qwen3_5TextModel

from rein.compiler.table import StateTable
from rein.genview.build import ViewRow
from rein.neurons.extract import INDEX_FILE, StateWriter, batches, extract

LAYERS = (1, 3)
LENGTHS = (7, 12, 5, 9)
PAD_ID = 0
STATE_ATOL = 1e-3


def _views() -> list[ViewRow]:
    generator = torch.Generator().manual_seed(5)
    views = []
    for number, length in enumerate(LENGTHS):
        ids = tuple(torch.randint(1, 500, (length,), generator=generator).tolist())
        positions = {"reply_start": (length - 1,), "silence": (0, length // 2)}
        views.append(
            ViewRow(f"v{number}|nothink", f"r{number}", "train", "nothink", "reply_start", ids, length, positions)
        )
    return views


def _meta(directory: Path, hidden: int) -> None:
    (directory / "extraction.json").write_text(
        f'{{"model": "tiny", "layers": [{LAYERS[0]}, {LAYERS[1]}], "hidden_size": {hidden}, "views": 4}}',
        encoding="utf-8",
    )


def test_batches_respect_the_token_budget() -> None:
    for batch in batches(_views(), token_budget=24):
        assert max(len(view.ids) for view in batch) * len(batch) <= 24 or len(batch) == 1


def test_captured_states_do_not_depend_on_padding_or_batching(tiny_backbone: Qwen3_5TextModel, tmp_path: Path) -> None:
    together, alone = tmp_path / "together", tmp_path / "alone"
    extract(tiny_backbone, _views(), LAYERS, StateWriter(together, shard_rows=1000), PAD_ID, token_budget=10_000)
    extract(tiny_backbone, _views(), LAYERS, StateWriter(alone, shard_rows=3), PAD_ID, token_budget=1)
    hidden = tiny_backbone.config.hidden_size
    for directory in (together, alone):
        _meta(directory, hidden)
    first, second = StateTable.load(together), StateTable.load(alone)

    def keyed(table: StateTable) -> dict[tuple[str, str, int], torch.Tensor]:
        return {(row["view_id"], row["kind"], row["token"]): table.states[i] for i, row in enumerate(table.index)}

    a, b = keyed(first), keyed(second)
    assert set(a) == set(b) and len(a) == sum(1 + 2 for _ in LENGTHS)
    for key in a:
        torch.testing.assert_close(a[key].float(), b[key].float(), atol=STATE_ATOL, rtol=0)
    assert first.states.shape[1:] == (len(LAYERS), hidden)


def test_extraction_resumes_without_redoing_finished_views(tiny_backbone: Qwen3_5TextModel, tmp_path: Path) -> None:
    out = tmp_path / "states"
    assert extract(tiny_backbone, _views(), LAYERS, StateWriter(out, shard_rows=2), PAD_ID, token_budget=16) == 4
    assert extract(tiny_backbone, _views(), LAYERS, StateWriter(out, shard_rows=2), PAD_ID, token_budget=16) == 0
    lines = (out / INDEX_FILE).read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 12
    shapes = [np.load(path).shape for path in sorted(out.glob("states-*.npy"))]
    assert sum(shape[0] for shape in shapes) == 12


def test_a_missing_layer_is_refused(tmp_path: Path, tiny_backbone: Qwen3_5TextModel) -> None:
    extract(tiny_backbone, _views()[:1], LAYERS, StateWriter(tmp_path, shard_rows=10), PAD_ID)
    _meta(tmp_path, tiny_backbone.config.hidden_size)
    with pytest.raises(ValueError):
        StateTable.load(tmp_path, layers=[0])
