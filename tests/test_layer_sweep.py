from transformers import Qwen3_5TextModel

from rein.eval.layer_sweep import collect, probe_layer
from rein.model.encode import RequestEncoder
from rein.train.toy import toy_records
from tests.devices import DEVICE


def test_sweep_probes_every_block(tiny_backbone: Qwen3_5TextModel, encoder: RequestEncoder) -> None:
    device = DEVICE
    train = collect(tiny_backbone, encoder, toy_records("train", 8), device)
    dev = collect(tiny_backbone, encoder, toy_records("development", 4), device)
    assert sorted(train.query) == list(range(1, len(tiny_backbone.layers) + 1))
    point = probe_layer(train, dev, 2, device)
    assert point["layer"] == 2.0
    assert 0.0 <= point["accuracy"] <= 1.0
