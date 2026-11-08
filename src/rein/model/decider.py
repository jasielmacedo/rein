from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from rein.model.batch import DecisionBatch, collate
from rein.model.encode import EncodedRequest, RequestEncoder
from rein.model.head import LabelHead, SetPointerHead
from rein.model.prefix import shared_state_forward
from rein.model.soft import SoftPrompt, input_embeddings
from rein.schema import DecisionRequest

Head = SetPointerHead | LabelHead


def masked_probabilities(logits: torch.Tensor, mask: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    scaled = (logits / temperature).masked_fill(~mask, torch.finfo(logits.dtype).min)
    return torch.softmax(scaled, dim=-1).masked_fill(~mask, 0.0)


@dataclass(frozen=True)
class DecisionOutput:
    question_ids: list[str]
    option_keys: list[tuple[str, ...]]
    logits: torch.Tensor
    probs: torch.Tensor
    mask: torch.Tensor

    def by_key(self) -> dict[str, dict[str, float]]:
        return {
            qid: dict(zip(keys, row[: len(keys)].tolist(), strict=True))
            for qid, keys, row in zip(self.question_ids, self.option_keys, self.probs, strict=True)
        }


class Decider(nn.Module):
    def __init__(
        self,
        backbone: Qwen3_5TextModel,
        head: Head,
        encoder: RequestEncoder,
        temperature: float = 1.0,
        checkpointing: bool = False,
        adapter_on_state: bool = True,
        soft: SoftPrompt | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.encoder = encoder
        self.temperature = temperature
        self.checkpointing = checkpointing
        self.adapter_on_state = adapter_on_state
        self.soft = soft
        if soft is not None:
            soft.attach(input_embeddings(backbone))

    @property
    def device(self) -> torch.device:
        return next(self.backbone.parameters()).device

    def logits(self, batch: DecisionBatch) -> torch.Tensor:
        rows = shared_state_forward(
            self.backbone,
            batch,
            checkpointing=self.checkpointing and self.training,
            adapter_on_state=self.adapter_on_state,
        )
        h_query = rows[batch.query_row, batch.query_pos]
        with torch.autocast(device_type=rows.device.type, enabled=False):
            if isinstance(self.head, LabelHead):
                return self.head(h_query.float(), batch.option_token_ids, batch.option_mask)
            h_options = rows[batch.option_row, batch.option_pos]
            return self.head(h_query.float(), h_options.float(), batch.option_mask)

    @torch.no_grad()
    def decide_encoded(self, encoded: Sequence[EncodedRequest], temperature: float | None = None) -> DecisionOutput:
        batch = collate(encoded, self.encoder.pad_id).to(self.device)
        logits = self.logits(batch)
        probs = masked_probabilities(
            logits, batch.option_mask, self.temperature if temperature is None else temperature
        )
        return DecisionOutput(
            question_ids=batch.question_ids,
            option_keys=batch.option_keys,
            logits=logits.cpu(),
            probs=probs.cpu(),
            mask=batch.option_mask.cpu(),
        )

    def decide(self, request: DecisionRequest, temperature: float | None = None) -> DecisionOutput:
        return self.decide_encoded([self.encoder.encode(request)], temperature)
