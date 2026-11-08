import os
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from rein.model.decider import Decider

RESUME_DIR = "resume"
STATE_FILE = "state.pt"
PARTIAL_SUFFIX = ".partial"


@dataclass
class ResumePoint:
    """Where training stopped: the epoch, how many micro-batches of it ran, and the epoch's starting data RNG."""

    config_key: str
    epoch: int
    micro_done: int
    step: int
    epoch_rng_state: Any
    running: list[float] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)


def _trainable(decider: Decider) -> dict[str, torch.nn.Parameter]:
    return {name: param for name, param in decider.named_parameters() if param.requires_grad}


def save_resume(
    directory: Path,
    decider: Decider,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    point: ResumePoint,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "point": point.__dict__,
        "weights": {name: param.detach().cpu() for name, param in _trainable(decider).items()},
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    target = directory / STATE_FILE
    partial = target.with_name(target.name + PARTIAL_SUFFIX)
    torch.save(payload, partial)
    os.replace(partial, target)


def load_resume(
    directory: Path,
    decider: Decider,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config_key: str,
) -> ResumePoint | None:
    path = directory / STATE_FILE
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=True)
    point = ResumePoint(**payload["point"])
    if point.config_key != config_key:
        raise ValueError(f"{path} was written by a different training config; remove it to start over")
    trainable = _trainable(decider)
    if set(trainable) != set(payload["weights"]):
        raise ValueError(f"{path} does not match the trainable parameters of this model")
    with torch.no_grad():
        for name, param in trainable.items():
            param.copy_(payload["weights"][name])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    torch.set_rng_state(payload["torch_rng"])
    if payload["cuda_rng"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    return point


def epoch_rng(point: ResumePoint) -> random.Random:
    rng = random.Random()
    rng.setstate(point.epoch_rng_state)
    return rng


def clear_resume(directory: Path) -> None:
    shutil.rmtree(directory, ignore_errors=True)
