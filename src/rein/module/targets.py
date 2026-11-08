"""Where the rule module learns: each view's decision positions, with the record's label per rule category, and as many
of its silences, where nothing may fire. The labels are the generic detectors' (DR-026), so both read the same task."""

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from rein.compiler.category_probe import DECISION_KINDS, SILENCES_PER_DECISION, category_labels
from rein.genview import constants as gc
from rein.genview.build import ViewRow
from rein.schema import TrainingRecord

OPTIONS = 2


@dataclass(frozen=True)
class LabelTable:
    categories: tuple[str, ...]
    by_record: Mapping[str, tuple[int, ...]]

    @property
    def quiet(self) -> tuple[int, ...]:
        return (0,) * len(self.categories)


@dataclass(frozen=True)
class Targets:
    rows: torch.Tensor
    columns: torch.Tensor
    labels: torch.Tensor

    def __len__(self) -> int:
        return len(self.rows)


def label_table(records: Mapping[str, TrainingRecord], categories: Mapping[str, Mapping[str, str]]) -> LabelTable:
    """The categories found in the labels and, per record, 1 where some rule of that category fires."""
    found = {
        record_id: category_labels(records[record_id], rules)
        for record_id, rules in categories.items()
        if record_id in records
    }
    names = tuple(sorted({name for labels in found.values() for name in labels}))
    return LabelTable(names, {rid: tuple(labels.get(name, 0) for name in names) for rid, labels in found.items()})


def labelled_views(views: Sequence[ViewRow], table: LabelTable, split: str) -> list[ViewRow]:
    return [view for view in views if view.split == split and view.record_id in table.by_record]


def batch_targets(batch: Sequence[ViewRow], table: LabelTable, rng: random.Random) -> Targets:
    """Every decision position of each view with its record's labels, plus as many of its silences, all quiet."""
    rows: list[int] = []
    columns: list[int] = []
    labels: list[tuple[int, ...]] = []
    for row, view in enumerate(batch):
        decisions = [token for kind in DECISION_KINDS for token in view.positions.get(kind, ())]
        silences = list(view.positions.get(gc.SILENCE, ()))
        picked = rng.sample(silences, min(len(silences), len(decisions) * SILENCES_PER_DECISION))
        labelled = [(token, table.by_record[view.record_id]) for token in decisions]
        labelled += [(token, table.quiet) for token in picked]
        for token, label in labelled:
            rows.append(row)
            columns.append(token)
            labels.append(label)
    return Targets(torch.tensor(rows), torch.tensor(columns), torch.tensor(labels, dtype=torch.long))


def option_targets(labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Each (position, category) as a two-option question: its target (one-hot for a label, the two chances for a
    teacher's probability) and the position that owns it, so every position weighs the same in the loss whatever the
    number of categories."""
    flat = labels.reshape(-1)
    owners = torch.arange(len(labels), device=labels.device).repeat_interleave(labels.shape[1])
    if flat.is_floating_point():
        return torch.stack([1.0 - flat, flat], dim=-1), owners
    return torch.nn.functional.one_hot(flat, OPTIONS).float(), owners
