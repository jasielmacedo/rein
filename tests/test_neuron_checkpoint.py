import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM

from rein.neurons import constants as c
from rein.neurons.checkpoint import PlacedNeuron, expand_checkpoint, original_mismatches, shard_names
from rein.neurons.tiny import tiny_qwen_checkpoint
from rein.neurons.units import Neuron, mlp_contribution
from tests.conftest import CHECKPOINT_VOCAB, TINY_HIDDEN

ADDED = c.NEURON_ALIGN
BATCH, LENGTH = 2, 12
LOGIT_ATOL = 1e-5
NEURON_ATOL = 1e-4
LAST_BLOCK = 3


def _ids() -> torch.Tensor:
    return torch.randint(0, CHECKPOINT_VOCAB, (BATCH, LENGTH), generator=torch.Generator().manual_seed(1))


def _load(directory: Path) -> torch.nn.Module:
    return AutoModelForCausalLM.from_pretrained(directory, dtype=torch.float32).eval()


def _tensor(directory: Path, key: str) -> torch.Tensor:
    for name in shard_names(directory):
        with safe_open(str(directory / name), framework="pt") as reader:
            names = reader.keys()
            if key in names:
                return reader.get_tensor(key)
    raise KeyError(key)


def _random_neurons(count: int) -> list[Neuron]:
    generator = torch.Generator().manual_seed(2)
    return [Neuron(*(torch.randn(TINY_HIDDEN, generator=generator) for _ in range(3))) for _ in range(count)]


def test_1_expansion_keeps_every_original_tensor_bit_for_bit(tiny_checkpoint: Path, tmp_path: Path) -> None:
    target = tmp_path / "expanded"
    expansion = expand_checkpoint(tiny_checkpoint, target, ADDED)
    assert original_mismatches(tiny_checkpoint, target) == []

    before = json.loads((tiny_checkpoint / c.CONFIG_FILE).read_text(encoding="utf-8"))
    after = json.loads((target / c.CONFIG_FILE).read_text(encoding="utf-8"))
    assert after[c.TEXT_CONFIG][c.INTERMEDIATE_SIZE] == expansion.base_size + ADDED
    after[c.TEXT_CONFIG][c.INTERMEDIATE_SIZE] = before[c.TEXT_CONFIG][c.INTERMEDIATE_SIZE]
    assert after == before

    size = expansion.size
    assert tuple(_tensor(target, "mtp.layers.0.mlp.gate_proj.weight").shape) == (size, TINY_HIDDEN)
    assert tuple(_tensor(target, "mtp.layers.0.mlp.down_proj.weight").shape) == (TINY_HIDDEN, size)
    added_rows = _tensor(target, "model.language_model.layers.0.mlp.up_proj.weight")[expansion.base_size :]
    assert torch.count_nonzero(added_rows) == 0


def test_2_zero_neurons_leave_the_outputs_unchanged(tiny_checkpoint: Path, tmp_path: Path) -> None:
    target = tmp_path / "expanded"
    expand_checkpoint(tiny_checkpoint, target, ADDED)
    ids = _ids()
    with torch.no_grad():
        original = _load(tiny_checkpoint)(input_ids=ids).logits
        expanded = _load(target)(input_ids=ids).logits
    torch.testing.assert_close(expanded, original, atol=LOGIT_ATOL, rtol=0)


def test_3_placed_neurons_add_exactly_what_their_math_says(tiny_checkpoint: Path, tmp_path: Path) -> None:
    neurons = _random_neurons(2)
    placed = [PlacedNeuron(block=LAST_BLOCK, slot=slot, neuron=neuron) for slot, neuron in enumerate(neurons)]
    target = tmp_path / "expanded"
    expand_checkpoint(tiny_checkpoint, target, ADDED, placed)

    original = _load(tiny_checkpoint)
    captured: dict[str, torch.Tensor] = {}
    layer = original.model.layers[LAST_BLOCK]
    layer.mlp.register_forward_pre_hook(lambda module, args: captured.update(mlp_input=args[0]))
    layer.register_forward_hook(lambda module, args, output: captured.update(block_output=output))
    ids = _ids()
    with torch.no_grad():
        original(input_ids=ids)
        block_output = captured["block_output"]
        block_output = block_output[0] if isinstance(block_output, tuple) else block_output
        shifted = block_output + mlp_contribution(captured["mlp_input"], neurons)
        expected = original.lm_head(original.model.norm(shifted))
        actual = _load(target)(input_ids=ids).logits
    torch.testing.assert_close(actual, expected, atol=NEURON_ATOL, rtol=0)


def test_manifest_lists_the_placed_neurons(tiny_checkpoint: Path, tmp_path: Path) -> None:
    neuron = _random_neurons(1)[0]
    target = tmp_path / "expanded"
    expand_checkpoint(
        tiny_checkpoint, target, ADDED, [PlacedNeuron(0, 5, neuron, role="test")], manifest={"sheet": "demo"}
    )
    manifest = json.loads((target / c.MANIFEST_FILE).read_text(encoding="utf-8"))
    assert manifest["added_per_block"] == ADDED
    assert manifest["neurons"] == [{"block": 0, "slot": 5, "role": "test"}]
    assert manifest["sheet"] == "demo"


@pytest.mark.parametrize("added", [0, 100, -256])
def test_expansion_rejects_sizes_that_break_k_quant_blocks(tiny_checkpoint: Path, tmp_path: Path, added: int) -> None:
    with pytest.raises(ValueError):
        expand_checkpoint(tiny_checkpoint, tmp_path / "expanded", added)


def test_expansion_rejects_bad_placements(tiny_checkpoint: Path, tmp_path: Path) -> None:
    neuron = _random_neurons(1)[0]
    wrong_shape = Neuron(torch.ones(TINY_HIDDEN + 1), neuron.up, neuron.down)
    not_finite = Neuron(torch.full((TINY_HIDDEN,), float("nan")), neuron.up, neuron.down)
    bad = [
        [PlacedNeuron(LAST_BLOCK + 1, 0, neuron)],
        [PlacedNeuron(0, ADDED, neuron)],
        [PlacedNeuron(0, 1, neuron), PlacedNeuron(0, 1, neuron)],
        [PlacedNeuron(0, 0, wrong_shape)],
        [PlacedNeuron(0, 0, not_finite)],
    ]
    for index, placements in enumerate(bad):
        with pytest.raises(ValueError):
            expand_checkpoint(tiny_checkpoint, tmp_path / f"bad-{index}", ADDED, placements)


def test_expansion_refuses_a_non_empty_target(tiny_checkpoint: Path, tmp_path: Path) -> None:
    target = tmp_path / "expanded"
    target.mkdir()
    (target / "keep.txt").write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError):
        expand_checkpoint(tiny_checkpoint, target, ADDED)


def test_a_single_file_checkpoint_gets_an_index_for_its_mtp_shard(tmp_path: Path) -> None:
    directory = tiny_qwen_checkpoint(tmp_path / "single", CHECKPOINT_VOCAB, shard_size="1GB")
    index = json.loads((directory / c.INDEX_FILE).read_text(encoding="utf-8"))
    assert set(index["weight_map"].values()) == {"model.safetensors", "model-mtp.safetensors"}
    expand_checkpoint(directory, tmp_path / "expanded", ADDED)
    assert original_mismatches(directory, tmp_path / "expanded") == []
