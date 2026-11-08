from collections.abc import Iterable, Mapping

import torch
from torch import nn


class MasterWeights:
    """Float32 copies of bf16 parameters for the optimizer: Adam's small steps would round away in bf16 weights. The
    model keeps running in bf16, as every runtime will run it."""

    def __init__(self, params: Mapping[str, nn.Parameter]) -> None:
        self.names = tuple(params)
        self.params = list(params.values())
        self.master = [nn.Parameter(param.detach().float().clone()) for param in self.params]

    def pull_gradients(self) -> None:
        """Adds the model's gradients to the float32 copies, so several batches can add up before a step."""
        for master, param in zip(self.master, self.params, strict=True):
            if param.grad is not None:
                gradient = param.grad.float()
                master.grad = gradient if master.grad is None else master.grad + gradient
            param.grad = None

    @torch.no_grad()
    def push_weights(self) -> None:
        for master, param in zip(self.master, self.params, strict=True):
            param.copy_(master)

    @torch.no_grad()
    def load(self, values: Iterable[torch.Tensor]) -> None:
        for master, value in zip(self.master, values, strict=True):
            master.copy_(value)
        self.push_weights()
