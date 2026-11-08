import json
from pathlib import Path

import torch
from transformers import PreTrainedTokenizerBase, Qwen3_5TextModel

from rein.model.backbone import LoraSettings, attach_lora
from rein.model.checkpoint import CONFIG_FILE, MERGED_FORMAT_VERSION, save_head
from rein.model.decider import Decider
from rein.model.encode import RequestEncoder
from rein.model.loader import load_merged
from rein.schema import DecisionRequest
from tests.conftest import TINY_HEAD_DIM, TINY_HEAD_HEADS
from tests.devices import DEVICE, PROB_ATOL

LORA_SCALE = 0.5
MAX_STATE_TOKENS = 512


def test_a_merged_export_loads_as_a_plain_qwen_and_decides_the_same(
    decider: Decider,
    tokenizer: PreTrainedTokenizerBase,
    roleplay_request: DecisionRequest,
    tmp_path: Path,
) -> None:
    peft_model = attach_lora(decider.backbone, LoraSettings())
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if "lora_B" in name:
                param.normal_(std=LORA_SCALE)
    decider.eval()
    expected = decider.decide(roleplay_request).probs

    merged = peft_model.merge_and_unload()
    assert isinstance(merged, Qwen3_5TextModel)
    merged.save_pretrained(tmp_path)
    tokenizer.save_pretrained(tmp_path)
    save_head(tmp_path, decider.head)
    meta = {
        "format_version": MERGED_FORMAT_VERSION,
        "head": {"dim": TINY_HEAD_DIM, "heads": TINY_HEAD_HEADS, "set_block": True},
        "max_state_tokens": MAX_STATE_TOKENS,
        "temperature": 1.0,
    }
    (tmp_path / CONFIG_FILE).write_text(json.dumps(meta), encoding="utf-8")

    restored = load_merged(tmp_path, DEVICE, dtype=torch.float32)
    assert isinstance(restored.encoder, RequestEncoder)
    assert not any("lora" in name for name, _ in restored.named_parameters())
    torch.testing.assert_close(restored.decide(roleplay_request).probs, expected, atol=PROB_ATOL, rtol=0)
