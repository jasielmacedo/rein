import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from rein.export.soft_merge import (
    INDEX_FILE,
    LM_HEAD_KEY,
    SOFT_META_FILE,
    differing_tensors,
    embedding_shards,
    merge_soft_run,
)
from rein.model.checkpoint import CONFIG_FILE, FORMAT_VERSION, SOFT_FILE
from rein.model.soft import soft_token_texts
from rein.neurons.tiny import TINY_HIDDEN, tiny_qwen_checkpoint
from tests.conftest import CHECKPOINT_SHARD, SEED, TINY_VOCAB, TOKENIZER_REPO

FRONT = 3
TAIL = 2


@pytest.fixture(scope="module")
def full_vocab_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tiny_qwen_checkpoint(tmp_path_factory.mktemp("tiny-full-vocab"), TINY_VOCAB, CHECKPOINT_SHARD, SEED)


def _soft_run(directory: Path, base: Path) -> tuple[Path, torch.Tensor]:
    run = directory / "soft-run"
    run.mkdir()
    rows = torch.randn(FRONT + TAIL, TINY_HIDDEN, generator=torch.Generator().manual_seed(SEED))
    save_file({"weight": rows}, run / SOFT_FILE)
    meta = {
        "format_version": FORMAT_VERSION,
        "base_repo": str(base),
        "soft_tokens": {"front": FRONT, "tail": TAIL},
        "readout_layer": None,
        "temperature": 1.0,
    }
    (run / CONFIG_FILE).write_text(json.dumps(meta), encoding="utf-8")
    return run, rows


def test_the_merge_changes_only_the_soft_rows_and_adds_the_tokens(full_vocab_checkpoint: Path, tmp_path: Path) -> None:
    run, rows = _soft_run(tmp_path, full_vocab_checkpoint)
    out = tmp_path / "merged"
    report = merge_soft_run(run, out, tokenizer_source=TOKENIZER_REPO)
    ids = report["ids"]
    assert len(ids) == FRONT + TAIL
    assert differing_tensors(full_vocab_checkpoint, out, ids) == []

    ((shard, key),) = embedding_shards(out).items()
    table = load_file(out / shard)[key]
    torch.testing.assert_close(table[ids], rows.to(table.dtype))

    tokenizer = AutoTokenizer.from_pretrained(out)
    front_texts, tail_texts = soft_token_texts(FRONT, TAIL)
    assert tokenizer.convert_tokens_to_ids(front_texts + tail_texts) == ids
    assert json.loads((out / SOFT_META_FILE).read_text(encoding="utf-8"))["ids"] == ids
    assert (out / "config.json").exists()


def test_an_untied_merge_keeps_the_base_table_as_the_output_head(full_vocab_checkpoint: Path, tmp_path: Path) -> None:
    run, rows = _soft_run(tmp_path, full_vocab_checkpoint)
    out = tmp_path / "untied"
    report = merge_soft_run(run, out, tokenizer_source=TOKENIZER_REPO, untie=True)
    ids = report["ids"]
    assert report["untied_head"] is True
    assert differing_tensors(full_vocab_checkpoint, out, ids) == []

    ((shard, key),) = embedding_shards(out).items()
    tensors = load_file(out / shard)
    base_table = load_file(full_vocab_checkpoint / shard)[key]
    torch.testing.assert_close(tensors[LM_HEAD_KEY], base_table)
    torch.testing.assert_close(tensors[key][ids], rows.to(tensors[key].dtype))

    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert config["tie_word_embeddings"] is False
    assert config.get("text_config", {}).get("tie_word_embeddings", False) is False
    index = json.loads((out / INDEX_FILE).read_text(encoding="utf-8"))
    assert index["weight_map"][LM_HEAD_KEY] == shard
    assert index["metadata"]["total_size"] == (
        json.loads((full_vocab_checkpoint / INDEX_FILE).read_text(encoding="utf-8"))["metadata"]["total_size"]
        + base_table.numel() * base_table.element_size()
    )
