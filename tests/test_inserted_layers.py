import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM

from rein.neurons import constants as c
from rein.neurons.checkpoint import shard_names
from rein.neurons.layers import LAYER_TYPES, insert_layers
from rein.neurons.tiny import TINY_LAYER_TYPES, tiny_qwen_checkpoint

VOCAB = 64
SHARD = "100KB"
PERIOD = 2
TOKENS = 12


def _tensors(directory: Path) -> dict[str, torch.Tensor]:
    out = {}
    for name in shard_names(directory):
        with safe_open(str(directory / name), framework="pt") as reader:
            names = reader.keys()
            for key in names:
                out[key] = reader.get_tensor(key)
    return out


def test_zero_output_layers_leave_every_logit_unchanged(tmp_path: Path) -> None:
    source = tiny_qwen_checkpoint(tmp_path / "original", VOCAB, SHARD)
    target = tmp_path / "inserted"
    insertion = insert_layers(source, target, at=PERIOD, group=PERIOD)
    config = json.loads((target / c.CONFIG_FILE).read_text(encoding="utf-8"))["text_config"]
    assert config["num_hidden_layers"] == len(TINY_LAYER_TYPES) + PERIOD
    assert config[LAYER_TYPES] == [*TINY_LAYER_TYPES[:PERIOD], *TINY_LAYER_TYPES[:PERIOD], *TINY_LAYER_TYPES[PERIOD:]]
    before, after = _tensors(source), _tensors(target)
    for key, tensor in before.items():
        moved = key
        if key.startswith("model.language_model.layers."):
            layer = int(key.split(".")[3])
            moved = key.replace(f".layers.{layer}.", f".layers.{insertion.moved(layer)}.", 1)
        assert torch.equal(after[moved], tensor)
    ids = torch.randint(0, VOCAB, (1, TOKENS), generator=torch.Generator().manual_seed(0))
    original = AutoModelForCausalLM.from_pretrained(source, dtype=torch.float32).eval()
    inserted = AutoModelForCausalLM.from_pretrained(target, dtype=torch.float32).eval()
    with torch.no_grad():
        assert torch.equal(original(input_ids=ids).logits, inserted(input_ids=ids).logits)
