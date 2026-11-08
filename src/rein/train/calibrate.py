import math

import torch

TEMPERATURE_MIN = 0.25
TEMPERATURE_MAX = 4.0
TEMPERATURE_STEPS = 121


def temperature_grid() -> torch.Tensor:
    return torch.logspace(math.log10(TEMPERATURE_MIN), math.log10(TEMPERATURE_MAX), TEMPERATURE_STEPS)


def nll_at(logits: torch.Tensor, mask: torch.Tensor, labels: torch.Tensor, temperature: float) -> float:
    scaled = (logits / temperature).masked_fill(~mask, torch.finfo(logits.dtype).min)
    log_probs = torch.log_softmax(scaled, dim=-1)
    return float(-log_probs.gather(-1, labels[:, None]).mean())


def fit_temperature(
    logits: torch.Tensor, mask: torch.Tensor, labels: torch.Tensor, grid: torch.Tensor | None = None
) -> float:
    """Single temperature minimizing NLL over hard-labeled questions (labels < 0 are skipped)."""
    keep = labels >= 0
    if not torch.any(keep):
        raise ValueError("no labeled questions to fit a temperature on")
    logits, mask, labels = logits[keep].float(), mask[keep], labels[keep]
    candidates = temperature_grid() if grid is None else grid
    losses = [nll_at(logits, mask, labels, float(t)) for t in candidates]
    return float(candidates[min(range(len(losses)), key=losses.__getitem__)])
