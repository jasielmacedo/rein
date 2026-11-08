"""The encoder a saved run decides with, built from its metadata alone: layout, state budget and soft tokens. Kept
apart from the loader so a serving client can encode requests without the training stack."""

from typing import Any

from rein.model import constants as mc
from rein.model.encode import RequestEncoder
from rein.model.head import layout_for
from rein.model.soft import add_soft_tokens


def readout_of(meta: dict[str, Any]) -> str:
    return str(meta.get("readout", mc.READOUT_SET))


def soft_ids_of(meta: dict[str, Any], tokenizer: Any) -> tuple[tuple[int, ...], tuple[int, ...]]:
    counts = meta.get("soft_tokens") or {"front": 0, "tail": 0}
    return add_soft_tokens(tokenizer, int(counts["front"]), int(counts["tail"]))


def build_encoder(meta: dict[str, Any], tokenizer: Any) -> RequestEncoder:
    front, tail = soft_ids_of(meta, tokenizer)
    return RequestEncoder(
        tokenizer,
        max_state_tokens=meta["max_state_tokens"],
        layout=layout_for(readout_of(meta)),
        soft_front=front,
        soft_tail=tail,
    )
