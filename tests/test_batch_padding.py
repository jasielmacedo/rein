import pytest
import torch

from rein.model.batch import BatchPadding, collate
from rein.model.decider import Decider
from rein.schema import DecisionRequest
from tests.devices import DEVICE, GRAD_GLOBAL_RELATIVE_L2, PROB_ATOL

BUCKETS = BatchPadding(state_multiple=64, row_multiple=16, row_count_multiple=32)
WEIGHT_SEED = 3


def _logits_and_grads(decider: Decider, requests: list[DecisionRequest], padding: BatchPadding):
    encoded = [decider.encoder.encode(request) for request in requests]
    batch = collate(encoded, decider.encoder.pad_id, padding).to(DEVICE)
    decider.zero_grad(set_to_none=True)
    logits = decider.logits(batch)
    weights = torch.randn(logits.shape, generator=torch.Generator().manual_seed(WEIGHT_SEED)).to(DEVICE)
    (logits.masked_fill(~batch.option_mask, 0.0) * weights).sum().backward()
    grads = [param.grad.clone() for param in decider.parameters() if param.grad is not None]
    return logits.detach(), batch, grads


def test_bucketed_padding_rounds_the_shapes(
    decider: Decider, roleplay_request: DecisionRequest, tool_call_request: DecisionRequest
) -> None:
    encoded = [decider.encoder.encode(request) for request in (roleplay_request, tool_call_request)]
    batch = collate(encoded, decider.encoder.pad_id, BUCKETS)
    assert batch.state_ids.shape[1] % BUCKETS.state_multiple == 0
    assert batch.row_ids.shape[1] % BUCKETS.row_multiple == 0
    assert batch.row_ids.shape[0] % BUCKETS.row_count_multiple == 0


def test_bucketed_padding_changes_neither_outputs_nor_gradients(
    decider: Decider, roleplay_request: DecisionRequest, tool_call_request: DecisionRequest
) -> None:
    decider.to(DEVICE).train()
    requests = [roleplay_request, tool_call_request]
    plain_logits, plain_batch, plain_grads = _logits_and_grads(decider, requests, BatchPadding())
    padded_logits, padded_batch, padded_grads = _logits_and_grads(decider, requests, BUCKETS)

    assert padded_batch.state_ids.shape[1] > plain_batch.state_ids.shape[1]
    assert padded_batch.row_ids.shape[0] > plain_batch.row_ids.shape[0]
    mask = plain_batch.option_mask
    torch.testing.assert_close(padded_logits[mask], plain_logits[mask], atol=PROB_ATOL, rtol=0)
    diff = torch.sqrt(sum(((a - b) ** 2).sum() for a, b in zip(padded_grads, plain_grads, strict=True)))
    norm = torch.sqrt(sum((b**2).sum() for b in plain_grads))
    assert float(diff / norm) < GRAD_GLOBAL_RELATIVE_L2


def test_padding_multiples_must_be_positive() -> None:
    with pytest.raises(ValueError):
        BatchPadding(state_multiple=0)
