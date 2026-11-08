import pytest
import torch

from rein.model.batch import collate
from rein.model.decider import Decider
from rein.schema import DecisionRequest
from rein.train.loss import decision_loss

BRIER_WEIGHT = 0.25

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def test_forward_and_backward_never_sync_with_the_host(
    decider: Decider, roleplay_request: DecisionRequest, tool_call_request: DecisionRequest
) -> None:
    decider = decider.cuda().train()
    encoded = [decider.encoder.encode(roleplay_request), decider.encoder.encode(tool_call_request)]
    batch = collate(encoded, decider.encoder.pad_id).to("cuda")
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        logits = decider.logits(batch)
        loss = decision_loss(logits, batch.option_mask, batch.targets, batch.question_owner, BRIER_WEIGHT)
        loss.total.backward()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
