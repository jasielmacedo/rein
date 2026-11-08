from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from rein.genview.build import ViewRow
from rein.genview.reasoning import ANSWER_START, REASONING, merge_reasoning
from rein.neurons import constants as c
from rein.neurons.checkpoint import expand_checkpoint
from rein.neurons.portability import forcing_checkpoint, kl_stats, last_layer_stats
from tests.conftest import CHECKPOINT_VOCAB

FORCED_TOKEN = 77
SEQUENCES = [[5, 9, 41, 3, 200, 17], [8, 8, 120, 64, 3], [400, 2, 33, 71, 18, 90, 11]]
CLOSE_ID, GAP_ID = 99, 98


def _load(directory: Path) -> torch.nn.Module:
    return AutoModelForCausalLM.from_pretrained(directory, dtype=torch.float32).eval()


def test_an_always_on_neuron_forces_its_token_at_every_position(tiny_checkpoint: Path, tmp_path: Path) -> None:
    original = _load(tiny_checkpoint)
    stats = last_layer_stats(original.model, SEQUENCES)
    layer = len(original.model.layers) - 1
    target = tmp_path / "forced"
    forcing_checkpoint(
        tiny_checkpoint,
        target,
        token="tok",
        token_id=FORCED_TOKEN,
        embedding=original.model.embed_tokens.weight.detach(),
        final_norm_weight=original.model.norm.weight.detach(),
        stats=stats,
        layer=layer,
    )
    forced = _load(target)
    with torch.no_grad():
        for ids in SEQUENCES:
            logits = forced(input_ids=torch.tensor([ids])).logits[0]
            assert torch.all(logits.argmax(-1) == FORCED_TOKEN)
            assert not torch.all(original(input_ids=torch.tensor([ids])).logits[0].argmax(-1) == FORCED_TOKEN)


def test_kl_is_zero_against_a_zero_expansion(tiny_checkpoint: Path, tmp_path: Path) -> None:
    expand_checkpoint(tiny_checkpoint, tmp_path / "zero", c.NEURON_ALIGN)
    stats = kl_stats(_load(tiny_checkpoint), _load(tmp_path / "zero"), SEQUENCES, starts=[2, 2, 3])
    assert stats["kl_max"] < 1e-6
    assert stats["top1_agreement"] == 1.0
    assert stats["tokens"] == sum(len(ids) - start for ids, start in zip(SEQUENCES, [2, 2, 3], strict=True))
    assert max(max(ids) for ids in SEQUENCES) < CHECKPOINT_VOCAB


def _thinking_view() -> ViewRow:
    return ViewRow("r|think", "r", "train", "think", "reply_start", (1, 2, 3, 4), 4, {"reply_start": (3,)})


def test_closed_reasoning_marks_the_answer_start() -> None:
    merged = merge_reasoning(_thinking_view(), [10, 11, 12, CLOSE_ID, 55], CLOSE_ID, [GAP_ID])
    assert merged.ids == (1, 2, 3, 4, 10, 11, 12, CLOSE_ID, GAP_ID)
    assert merged.positions[ANSWER_START] == (8,)
    assert all(4 <= index < 7 for index in merged.positions[REASONING])
    assert merged.positions["reply_start"] == (3,)


def test_unfinished_reasoning_has_no_answer_start() -> None:
    merged = merge_reasoning(_thinking_view(), [10, 11, 12], CLOSE_ID, [GAP_ID])
    assert merged.ids == (1, 2, 3, 4, 10, 11, 12)
    assert ANSWER_START not in merged.positions
    assert len(merged.positions[REASONING]) == 2
