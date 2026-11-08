import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from rein.compiler.circuit import CircuitCompiler
from rein.compiler.model import Compiler
from rein.compiler.reader import RuleReading
from rein.compiler.table import Item
from rein.train.loss import decision_loss

Log = Callable[[dict[str, float]], None]
RuleModel = Compiler | CircuitCompiler


@dataclass(frozen=True)
class TrainSettings:
    epochs: int = 4
    rows_per_batch: int = 64
    learning_rate: float = 3.0e-4
    weight_decay: float = 0.01
    warmup_fraction: float = 0.05
    brier_weight: float = 0.25
    silence_weight: float = 0.5
    silence_per_batch: int = 256
    max_grad_norm: float = 1.0
    log_every: int = 50
    seed: int = 0


@dataclass(frozen=True)
class Batch:
    readings: torch.Tensor
    rows: torch.Tensor
    owners: torch.Tensor
    targets: torch.Tensor
    nulls: torch.Tensor


def make_batch(items: Sequence[Item], device: torch.device) -> Batch:
    """The batch only spans as many option slots as its widest rule, so a rare wide rule costs nothing elsewhere."""
    owners: dict[int, int] = {}
    targets = torch.zeros(len(items), max(len(item.option_keys) for item in items))
    for position, item in enumerate(items):
        owners.setdefault(item.row, len(owners))
        targets[position, : len(item.target)] = torch.tensor(item.target)
    return Batch(
        readings=torch.tensor([item.reading for item in items], device=device),
        rows=torch.tensor([item.row for item in items], device=device),
        owners=torch.tensor([owners[item.row] for item in items], device=device),
        targets=targets.to(device),
        nulls=torch.tensor([item.null for item in items], device=device),
    )


def batch_logits(
    compiler: RuleModel, readings: RuleReading, states: torch.Tensor, channels: torch.Tensor, batch: Batch
) -> tuple[torch.Tensor, torch.Tensor]:
    """Logits (items, output blocks, options) and the option mask; each rule is compiled once even if it repeats."""
    unique, inverse = torch.unique(batch.readings, return_inverse=True)
    reading = readings.select(unique, int(batch.targets.shape[1]))
    logits = compiler.item_logits(reading, inverse, states[batch.rows].float(), channels, batch.nulls)
    return logits, reading.mask[inverse]


def layered_loss(logits: torch.Tensor, mask: torch.Tensor, batch: Batch, brier_weight: float) -> torch.Tensor:
    parts = [
        decision_loss(logits[:, layer], mask, batch.targets[:, : mask.shape[1]], batch.owners, brier_weight).total
        for layer in range(logits.shape[1])
    ]
    return torch.stack(parts).mean()


def _rows(items: Sequence[Item]) -> dict[int, list[Item]]:
    grouped: dict[int, list[Item]] = {}
    for item in items:
        grouped.setdefault(item.row, []).append(item)
    return grouped


def _schedule(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def train_compiler(
    compiler: RuleModel,
    readings: RuleReading,
    states: torch.Tensor,
    channels: torch.Tensor,
    decisions: Sequence[Item],
    silences: Sequence[Item],
    settings: TrainSettings,
    log: Log = lambda entry: None,
) -> None:
    """Cross-entropy over each rule's options plus Brier at decision positions, and the null option where the rule
    must stay silent. States, readings and channels must already sit on the compiler's device."""
    device = channels.device
    rng = random.Random(settings.seed)
    by_row = _rows(decisions)
    rows = list(by_row)
    batches_per_epoch = max(1, math.ceil(len(rows) / settings.rows_per_batch))
    total = batches_per_epoch * settings.epochs
    optimizer = torch.optim.AdamW(compiler.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _schedule(step, total, int(settings.warmup_fraction * total))
    )
    step = 0
    compiler.train()
    for epoch in range(settings.epochs):
        rng.shuffle(rows)
        for start in range(0, len(rows), settings.rows_per_batch):
            items = [item for row in rows[start : start + settings.rows_per_batch] for item in by_row[row]]
            batch = make_batch(items, device)
            logits, mask = batch_logits(compiler, readings, states, channels, batch)
            loss = layered_loss(logits, mask, batch, settings.brier_weight)
            silence_loss = torch.zeros((), device=device)
            if silences and settings.silence_weight > 0:
                quiet = make_batch(rng.sample(list(silences), min(settings.silence_per_batch, len(silences))), device)
                quiet_logits, quiet_mask = batch_logits(compiler, readings, states, channels, quiet)
                silence_loss = layered_loss(quiet_logits, quiet_mask, quiet, brier_weight=0.0)
            total_loss = loss + settings.silence_weight * silence_loss
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(compiler.parameters(), settings.max_grad_norm)
            optimizer.step()
            scheduler.step()
            step += 1
            if step % settings.log_every == 0 or step == total:
                log({"epoch": epoch, "step": step, "loss": loss.item(), "silence_loss": silence_loss.item()})
    compiler.eval()


@torch.no_grad()
def predict_logits(
    compiler: RuleModel,
    readings: RuleReading,
    states: torch.Tensor,
    channels: torch.Tensor,
    items: Sequence[Item],
    batch_items: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Logits (items, layers, options) and masks on the CPU, padded to the widest rule of `items`."""
    width = max((len(item.option_keys) for item in items), default=0)
    logits, masks = [], []
    for start in range(0, len(items), batch_items):
        batch = make_batch(items[start : start + batch_items], channels.device)
        part, mask = batch_logits(compiler, readings, states, channels, batch)
        extra = width - part.shape[-1]
        logits.append(F.pad(part.cpu(), (0, extra), value=torch.finfo(part.dtype).min))
        masks.append(F.pad(mask.cpu(), (0, extra), value=False))
    return torch.cat(logits), torch.cat(masks)
