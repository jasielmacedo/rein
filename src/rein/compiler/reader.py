from collections.abc import Sequence
from dataclasses import dataclass

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from rein.model.batch import collate
from rein.model.encode import RequestEncoder
from rein.model.prefix import shared_state_forward
from rein.schema import DecisionRequest, Question, State

READING_KEY = "rule"


@dataclass(frozen=True)
class RuleReading:
    """The frozen Qwen's reading of each rule: the query vector and one vector per option, padded to the widest rule."""

    query: torch.Tensor
    options: torch.Tensor
    mask: torch.Tensor
    option_keys: tuple[tuple[str, ...], ...]

    def to(self, device: torch.device | str) -> "RuleReading":
        return RuleReading(self.query.to(device), self.options.to(device), self.mask.to(device), self.option_keys)

    def select(self, index: torch.Tensor, width: int | None = None) -> "RuleReading":
        """The readings at `index`, cut to their first `width` option slots when given."""
        keys = tuple(self.option_keys[int(i)] for i in index.tolist())
        columns = slice(0, width)
        return RuleReading(self.query[index], self.options[index][:, columns], self.mask[index][:, columns], keys)


class RuleReader:
    """Reads a rule with the frozen backbone through Rein's encoder: the base is the state, each option its own row.

    Every option row continues the same state at the same positions, so reordering the options reorders the reading
    and changes nothing else.
    """

    def __init__(self, backbone: Qwen3_5TextModel, encoder: RequestEncoder) -> None:
        self.backbone = backbone
        self.encoder = encoder

    @torch.no_grad()
    def read(self, rules: Sequence[tuple[Question, str]]) -> RuleReading:
        requests = [
            DecisionRequest(
                state=State(context=base),
                questions={READING_KEY: question.model_copy(update={"label": None, "target": None})},
            )
            for question, base in rules
        ]
        device = next(self.backbone.parameters()).device
        batch = collate([self.encoder.encode(request) for request in requests], self.encoder.pad_id).to(device)
        rows = shared_state_forward(self.backbone, batch)
        query = rows[batch.query_row, batch.query_pos].float()
        options = rows[batch.option_row, batch.option_pos].float() * batch.option_mask[..., None]
        return RuleReading(query.cpu(), options.cpu(), batch.option_mask.cpu(), tuple(batch.option_keys))
