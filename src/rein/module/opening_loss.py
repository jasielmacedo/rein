"""The action where the reply starts. Where the turn needs a tool, the module learns the call's first tokens
(`<tool_call>` and the function name), so the reply opens with that call; where no call is wanted, it learns not to
open with `<tool_call>`. Rein-r1's decision is read at the same opening, and the KL inside the prompt keeps the reading
of the conversation as Qwen's.
"""

import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from rein.module.losses import (
    Stacked,
    Teacher,
    bypassed,
    decision_loss_of,
    divergence,
    last_hidden,
    log_probs_at,
    sequence_sums,
    stacked,
)
from rein.module.material import read_rows
from rein.module.targets import LabelTable
from rein.module.train import ModulePlace
from rein.neurons.extract import MlpInputCapture, load_views

CHANCE_CEILING = 1.0 - 1e-6


@dataclass(frozen=True)
class Opening:
    """A reply opening: with `calls` True the reply must open with `call` (the call's first tokens), with False it
    must not open with `call[0]`, and with None only the decision is learned there."""

    view_id: str
    record_id: str
    prompt: tuple[int, ...]
    call: tuple[int, ...] = ()
    calls: bool | None = None

    @property
    def width(self) -> int:
        return len(self.prompt) + len(self.call)


def load_openings(path: Path, views: Sequence[Path]) -> list[Opening]:
    """The openings of a targets file with their prompts from the view files; an opening without a view is left out."""
    prompts = {view.view_id: view.ids for views_path in views for view in load_views(views_path)}
    return [
        Opening(row["view_id"], row["record_id"], tuple(prompts[row["view_id"]]), tuple(row["call"]), row["calls"])
        for row in read_rows(path)
        if row["view_id"] in prompts
    ]


def opening_loss(
    backbone: Qwen3_5TextModel, hidden: torch.Tensor, batch: Stacked, openings: Sequence[Opening]
) -> torch.Tensor:
    """Minus the log-chance of the call's first tokens where the reply must open with them, and of not opening with
    the call's first token where no call is wanted, averaged over the openings with a target (zero without any)."""
    terms = []
    called = [index for index, opening in enumerate(openings) if opening.calls]
    if called:
        sums = sequence_sums(log_probs_at(backbone, hidden, batch.rows, batch.columns), batch, len(openings))
        terms.append(-sums[torch.tensor(called, device=sums.device)])
    held = [index for index, opening in enumerate(openings) if opening.calls is False]
    if held:
        rows = torch.tensor(held)
        log_probs = log_probs_at(backbone, hidden, rows, batch.openings[rows])
        avoided = torch.tensor([openings[index].call[0] for index in held], device=log_probs.device)
        chance = log_probs.gather(-1, avoided[:, None])[:, 0].exp().clamp(max=CHANCE_CEILING)
        terms.append(-torch.log1p(-chance))
    if not terms:
        return hidden.float().sum() * 0.0
    return torch.cat(terms).mean()


def opening_losses(
    backbone: Qwen3_5TextModel,
    place: ModulePlace,
    capture: MlpInputCapture,
    head: nn.Linear,
    openings: Sequence[Opening],
    table: LabelTable,
    brier_weight: float,
    rng: random.Random,
    teacher: Teacher | None = None,
) -> dict[str, torch.Tensor]:
    """The opening's action, the decision read at the opening and the KL at positions drawn from each prompt."""
    pieces = [(opening.prompt, opening.call if opening.calls else ()) for opening in openings]
    batch = stacked(pieces, int(backbone.config.pad_token_id or 0), rng, len(pieces))
    device = head.weight.device
    capture.aim(torch.arange(len(openings), device=device), batch.openings.to(device))
    with torch.no_grad(), bypassed(backbone, place.layers):
        reference = log_probs_at(backbone, last_hidden(backbone, batch), batch.prompt_rows, batch.prompt_columns)
    hidden = last_hidden(backbone, batch)
    return {
        "opening": opening_loss(backbone, hidden, batch, openings),
        "opening_prompt_kl": divergence(
            reference, log_probs_at(backbone, hidden, batch.prompt_rows, batch.prompt_columns)
        ),
        "opening_decision": decision_loss_of(head, capture, openings, table, teacher, brier_weight),
    }
