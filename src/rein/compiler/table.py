import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rein.compiler.model import null_option_index
from rein.genview.keys import rule_key
from rein.neurons.extract import INDEX_FILE, META_FILE
from rein.schema import TrainingRecord

SILENCE_KIND = "silence"


@dataclass(frozen=True)
class StateTable:
    """Every stored MLP input, as one (rows, layers, hidden) tensor, with one index entry per row."""

    states: torch.Tensor
    layers: tuple[int, ...]
    index: list[dict[str, Any]]

    @classmethod
    def load_many(cls, directories: Sequence[Path], layers: Sequence[int] | None = None) -> "StateTable":
        """Several extractions as one table, rows in the order of `directories`."""
        tables = [cls.load(directory, layers) for directory in directories]
        if len({table.layers for table in tables}) != 1:
            raise ValueError("every extraction must provide the same layers")
        return cls(
            states=torch.cat([table.states for table in tables]),
            layers=tables[0].layers,
            index=[row for table in tables for row in table.index],
        )

    @classmethod
    def load(cls, directory: Path, layers: Sequence[int] | None = None) -> "StateTable":
        stored = tuple(json.loads((directory / META_FILE).read_text(encoding="utf-8"))["layers"])
        wanted = tuple(layers) if layers else stored
        missing = sorted(set(wanted) - set(stored))
        if missing:
            raise ValueError(f"layers {missing} were not extracted")
        keep = [stored.index(layer) for layer in wanted]
        by_shard: dict[str, list[dict[str, Any]]] = {}
        with (directory / INDEX_FILE).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    entry = json.loads(line)
                    by_shard.setdefault(entry["shard"], []).append(entry)
        chunks, index = [], []
        for shard, entries in by_shard.items():
            array = np.load(directory / shard, mmap_mode="r")
            chunks.append(torch.from_numpy(np.ascontiguousarray(array[:, keep, :])))
            index.extend(sorted(entries, key=lambda entry: entry["row"]))
        return cls(torch.cat(chunks), wanted, index)


@dataclass(frozen=True)
class Item:
    """One rule asked at one stored position: what the compiler must answer there."""

    row: int
    reading: int
    question_id: str
    question_type: str
    option_keys: tuple[str, ...]
    label: int
    target: tuple[float, ...]
    null: int
    record_id: str
    split: str
    mode: str
    kind: str
    tags: Mapping[str, Any] = field(default_factory=dict)


def _one_hot(index: int, width: int) -> tuple[float, ...]:
    return tuple(1.0 if position == index else 0.0 for position in range(width))


def table_items(
    table: StateTable, records: Mapping[str, TrainingRecord], reading_index: Mapping[str, int]
) -> tuple[list[Item], list[Item]]:
    """Decision items (each rule at its record's decision positions) and silence items (each rule where it must stay
    off, labelled with its null option)."""
    decisions: list[Item] = []
    silences: list[Item] = []
    for row, entry in enumerate(table.index):
        record = records.get(entry["record_id"])
        if record is None:
            continue
        silent = entry["kind"] == SILENCE_KIND
        for question_id, question in record.questions.items():
            reading = reading_index.get(rule_key(question))
            if reading is None:
                continue
            keys = question.option_keys()
            null = null_option_index(keys)
            if silent and null < 0:
                continue
            label = question.label_index()
            supervised = question.target_vector() or (_one_hot(label, len(keys)) if label is not None else None)
            if silent:
                target = _one_hot(null, len(keys))
            elif supervised is None:
                continue
            else:
                target = supervised
            item = Item(
                row=row,
                reading=reading,
                question_id=question_id,
                question_type=question.type,
                option_keys=keys,
                label=null if silent else (-1 if label is None else label),
                target=target,
                null=null,
                record_id=record.id,
                split=record.split,
                mode=entry["mode"],
                kind=entry["kind"],
                tags={"split": record.split, "source": record.source, **record.meta},
            )
            (silences if silent else decisions).append(item)
    return decisions, silences
