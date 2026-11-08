from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from rein.model import constants as mc
from rein.model.backbone import LoraSettings, attach_lora, freeze, load_text_backbone, truncate_backbone
from rein.model.checkpoint import HEAD_FILE, MERGED_FORMAT_VERSION, load_weights, read_meta
from rein.model.decider import Decider, Head
from rein.model.encode import RequestEncoder
from rein.model.head import build_head, layout_for
from rein.model.run_meta import resolve
from rein.model.soft import SoftPrompt, add_soft_tokens, input_embeddings


def _readout(meta: dict[str, Any]) -> str:
    return str(meta.get("readout", mc.READOUT_SET))


def _head(meta: dict[str, Any], backbone: Qwen3_5TextModel) -> Head:
    head_meta = meta["head"]
    return build_head(_readout(meta), backbone, head_meta["dim"], head_meta["heads"], head_meta["set_block"])


def _soft_ids(meta: dict[str, Any], tokenizer: Any) -> tuple[tuple[int, ...], tuple[int, ...]]:
    counts = meta.get("soft_tokens") or {"front": 0, "tail": 0}
    return add_soft_tokens(tokenizer, int(counts["front"]), int(counts["tail"]))


def build_encoder(meta: dict[str, Any], tokenizer: Any) -> RequestEncoder:
    """The encoder a saved run decides with: its layout, state budget and soft tokens."""
    front, tail = _soft_ids(meta, tokenizer)
    return RequestEncoder(
        tokenizer,
        max_state_tokens=meta["max_state_tokens"],
        layout=layout_for(_readout(meta)),
        soft_front=front,
        soft_tail=tail,
    )


def _soft(meta: dict[str, Any], tokenizer: Any, backbone: Qwen3_5TextModel) -> SoftPrompt | None:
    front, tail = _soft_ids(meta, tokenizer)
    return SoftPrompt(front + tail, input_embeddings(backbone)) if front or tail else None


def _adapter_on_state(meta: dict[str, Any]) -> bool:
    return meta.get("adapter_scope", mc.ADAPTER_STATE_AND_ROWS) == mc.ADAPTER_STATE_AND_ROWS


def default_dtype(device: torch.device) -> torch.dtype:
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def load_unmerged(
    run_dir: Path, device: torch.device, token: str | None = None, dtype: torch.dtype | None = None
) -> tuple[Decider, PeftModel | None, dict[str, Any]]:
    """A training run: base backbone, its adapter and soft tokens when it has them, head and temperature."""
    meta = read_meta(run_dir)
    dtype = dtype or default_dtype(device)
    tokenizer = AutoTokenizer.from_pretrained(meta["base_repo"], revision=meta.get("base_revision"), token=token)
    backbone = load_text_backbone(meta["base_repo"], meta.get("base_revision"), dtype=dtype, token=token)
    backbone = truncate_backbone(backbone, meta.get("readout_layer"))
    lora = meta.get("lora")
    peft_model = attach_lora(backbone, LoraSettings(lora["rank"], lora["alpha"], lora["dropout"])) if lora else None
    if peft_model is None:
        freeze(backbone)
    soft = _soft(meta, tokenizer, backbone)
    decider = Decider(
        backbone,
        _head(meta, backbone),
        build_encoder(meta, tokenizer),
        temperature=meta["temperature"],
        adapter_on_state=_adapter_on_state(meta),
        soft=soft,
    ).to(device)
    load_weights(run_dir, peft_model, decider.head, device, soft)
    return decider.eval(), peft_model, meta


def load_merged(directory: Path, device: torch.device, dtype: torch.dtype | None = None) -> Decider:
    """An exported Rein model: a plain Qwen3.5 text model with the adapter merged in, plus the decision head."""
    meta = read_meta(directory)
    dtype = dtype or default_dtype(device)
    tokenizer = AutoTokenizer.from_pretrained(directory)
    backbone = Qwen3_5TextModel.from_pretrained(directory, dtype=dtype)
    head = _head(meta, backbone)
    head.load_state_dict(load_file(directory / HEAD_FILE))
    decider = Decider(
        backbone,
        head,
        build_encoder(meta, tokenizer),
        temperature=meta["temperature"],
        adapter_on_state=_adapter_on_state(meta),
    )
    return decider.to(device).eval()


def load_decider(source: Path | str, device: torch.device, token: str | None = None) -> Decider:
    directory = resolve(source, token)
    if read_meta(directory)["format_version"] == MERGED_FORMAT_VERSION:
        return load_merged(directory, device)
    return load_unmerged(directory, device, token)[0]
