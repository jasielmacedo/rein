import argparse
import json
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from rein.env import load_env, optional
from rein.eval.metrics import PredictionRow, accuracy, expected_calibration_error
from rein.model.backbone import load_text_backbone
from rein.model.batch import DecisionBatch, collate
from rein.model.decider import masked_probabilities
from rein.model.encode import RequestEncoder
from rein.model.head import SetPointerHead
from rein.model.prefix import shared_state_forward
from rein.schema import TrainingRecord
from rein.train.data import load_records, token_batches
from rein.train.loss import decision_loss

PROBE_DIM = 128
PROBE_HEADS = 4
PROBE_EPOCHS = 6
PROBE_LR = 1e-3
PROBE_BATCH = 256
PROBE_BRIER_WEIGHT = 0.25
FEATURE_BATCH_TOKENS = 32768
FEATURE_MAX_RECORDS = 32
EN_PT = frozenset({"en", "pt-BR"})
DEFAULT_TRAIN_LIMIT = 6000
DEFAULT_DEV_LIMIT = 2000


@dataclass
class LayerFeatures:
    """Query and option hidden states at every block, for the questions of a record set."""

    query: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    options: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    masks: list[torch.Tensor] = field(default_factory=list)
    targets: list[torch.Tensor] = field(default_factory=list)
    labels: list[torch.Tensor] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)

    def stacked(self, layer: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        width = max(mask.shape[1] for mask in self.masks)

        def pad(tensors: Sequence[torch.Tensor], value: float | bool) -> torch.Tensor:
            return torch.cat(
                [
                    F.pad(t, (0, 0, 0, width - t.shape[1]) if t.dim() == 3 else (0, width - t.shape[1]), value=value)
                    for t in tensors
                ]
            )

        return (
            torch.cat(self.query[layer]),
            pad(self.options[layer], 0.0),
            pad(self.masks, False),
            pad(self.targets, 0.0),
            torch.cat(self.labels),
        )


def _capture(features: LayerFeatures, batch: DecisionBatch, layer: int, rows: torch.Tensor) -> None:
    features.query.setdefault(layer, []).append(rows[batch.query_row, batch.query_pos].half().cpu())
    features.options.setdefault(layer, []).append(rows[batch.option_row, batch.option_pos].half().cpu())


@torch.no_grad()
def collect(
    backbone: Qwen3_5TextModel, encoder: RequestEncoder, records: Sequence[TrainingRecord], device: torch.device
) -> LayerFeatures:
    features = LayerFeatures()
    encoded = [encoder.encode(record) for record in records]
    for indices in token_batches(encoded, FEATURE_BATCH_TOKENS, FEATURE_MAX_RECORDS, random.Random(0)):
        batch = collate([encoded[i] for i in indices], encoder.pad_id).to(device)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            shared_state_forward(backbone, batch, capture=partial(_capture, features, batch))
        features.masks.append(batch.option_mask.cpu())
        features.targets.append(batch.targets.cpu())
        features.labels.append(batch.label_index.cpu())
        features.languages += [str(records[i].meta.get("lang_conv")) for i in indices for _ in encoded[i].questions]
    return features


def _rows(probs: torch.Tensor, mask: torch.Tensor, labels: torch.Tensor, languages: list[str]) -> list[PredictionRow]:
    rows = []
    for index in range(probs.shape[0]):
        if labels[index] < 0:
            continue
        count = int(mask[index].sum())
        rows.append(
            PredictionRow(
                str(index),
                "q",
                "probe",
                tuple(str(k) for k in range(count)),
                tuple(probs[index, :count].tolist()),
                int(labels[index]),
                {"lang": languages[index]},
            )
        )
    return rows


def probe_layer(train: LayerFeatures, dev: LayerFeatures, layer: int, device: torch.device) -> dict[str, float]:
    query, options, mask, targets, _ = train.stacked(layer)
    head = SetPointerHead(hidden_size=query.shape[-1], dim=PROBE_DIM, heads=PROBE_HEADS).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=PROBE_LR)
    generator = torch.Generator().manual_seed(layer)
    for _ in range(PROBE_EPOCHS):
        for batch in torch.randperm(query.shape[0], generator=generator).split(PROBE_BATCH):
            logits = head(query[batch].to(device), options[batch].to(device), mask[batch].to(device))
            owner = torch.arange(batch.shape[0], device=device)
            loss = decision_loss(logits, mask[batch].to(device), targets[batch].to(device), owner, PROBE_BRIER_WEIGHT)
            optimizer.zero_grad(set_to_none=True)
            loss.total.backward()
            optimizer.step()
    head.eval()
    d_query, d_options, d_mask, _, d_labels = dev.stacked(layer)
    with torch.no_grad():
        logits = head(d_query.to(device), d_options.to(device), d_mask.to(device)).cpu()
    rows = _rows(masked_probabilities(logits, d_mask), d_mask, d_labels, dev.languages)
    core = [row for row in rows if row.tags["lang"] in EN_PT]
    other = [row for row in rows if row.tags["lang"] not in EN_PT]
    return {
        "layer": float(layer),
        "accuracy": accuracy(rows),
        "ece": expected_calibration_error(rows),
        "accuracy_en_pt": accuracy(core),
        "accuracy_other_languages": accuracy(other),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe every block of a frozen backbone for decision quality.")
    parser.add_argument("--base", required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=DEFAULT_TRAIN_LIMIT)
    parser.add_argument("--dev-limit", type=int, default=DEFAULT_DEV_LIMIT)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    token = optional(load_env(), "HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(args.base, token=token)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    backbone = load_text_backbone(args.base, dtype=dtype, token=token)
    cast(Any, backbone).to(device)
    backbone.eval()
    encoder = RequestEncoder(tokenizer)
    rng = random.Random(0)
    train_records = load_records(args.train)
    dev_records = load_records(args.dev)
    train = collect(backbone, encoder, rng.sample(train_records, min(args.train_limit, len(train_records))), device)
    dev = collect(backbone, encoder, rng.sample(dev_records, min(args.dev_limit, len(dev_records))), device)
    curve = [probe_layer(train, dev, layer, device) for layer in sorted(train.query)]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "layer_sweep.json").write_text(json.dumps({"base": args.base, "curve": curve}, indent=2), "utf-8")
    for point in curve:
        print(json.dumps(point))


if __name__ == "__main__":
    main()
