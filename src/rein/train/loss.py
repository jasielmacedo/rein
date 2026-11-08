from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LossParts:
    total: torch.Tensor
    ce: torch.Tensor
    brier: torch.Tensor


def decision_loss(
    logits: torch.Tensor,
    mask: torch.Tensor,
    targets: torch.Tensor,
    question_owner: torch.Tensor,
    brier_weight: float,
) -> LossParts:
    """Cross-entropy over each question's options plus a Brier calibration term.

    Each record weighs the same no matter how many questions it carries.
    """
    log_probs = torch.log_softmax(logits.masked_fill(~mask, torch.finfo(logits.dtype).min), dim=-1)
    ce_per_question = -(targets * log_probs.masked_fill(~mask, 0.0)).sum(-1)
    probs = log_probs.exp().masked_fill(~mask, 0.0)
    brier_per_question = (probs - targets).square().masked_fill(~mask, 0.0).sum(-1)
    record_slots = int(question_owner.shape[0])
    counts = torch.zeros(record_slots, dtype=logits.dtype, device=logits.device).scatter_add_(
        0, question_owner, torch.ones_like(ce_per_question)
    )
    weights = 1.0 / counts.clamp(min=1.0)[question_owner]
    records = (counts > 0).sum().to(logits.dtype)
    ce = (weights * ce_per_question).sum() / records
    brier = (weights * brier_per_question).sum() / records
    return LossParts(total=ce + brier_weight * brier, ce=ce, brier=brier)
