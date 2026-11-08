from dataclasses import dataclass

import torch

from rein.neurons import constants as c


@dataclass(frozen=True)
class ChannelStats:
    mean: float
    std: float

    @property
    def variation(self) -> float:
        return self.std / abs(self.mean) if self.mean else float("inf")


def bias_channel(samples: torch.Tensor, ridge: float = c.DEFAULT_CHANNEL_RIDGE) -> torch.Tensor:
    """The functional with mean one and the least variance over `samples` (rows are MLP inputs).

    It solves min c'Sc subject to c'm = 1, so c = S^-1 m / (m' S^-1 m); the ridge is relative to the mean variance.
    """
    if samples.ndim != 2 or samples.shape[0] < 2:
        raise ValueError("bias_channel needs a 2-D tensor with at least two samples")
    data = samples.double()
    mean = data.mean(0)
    centered = data - mean
    covariance = centered.T @ centered / (data.shape[0] - 1)
    scale = covariance.diagonal().mean().clamp(min=torch.finfo(torch.float64).tiny)
    regularized = covariance + ridge * scale * torch.eye(data.shape[1], dtype=torch.float64)
    direction = torch.linalg.solve(regularized, mean)
    return (direction / (mean @ direction)).float()


def channel_stats(channel: torch.Tensor, samples: torch.Tensor) -> ChannelStats:
    values = samples.double() @ channel.double()
    return ChannelStats(mean=float(values.mean()), std=float(values.std()))
