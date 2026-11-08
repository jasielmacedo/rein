"""Reply openings at the first user turn: every training conversation cut after its first user message, under the same
rules and tools. That is where Qwen calls tools nobody asked for the most, and the training conversations almost never
stop there (61 of 14,903 views). A cut conversation has no label of its own (the record's label is for its decisive
turn), so Rein-r1 labels it (`rein.module.teacher`) and only its confident decisions become opening targets
(`rein.module.openings`).

The cut records keep their ids and go to their own data folder, so the teacher reads them in place of the whole
conversations; their views are told apart by the `|first` suffix.
"""

import argparse
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from rein.compiler.category_probe import TRAIN
from rein.genview import constants as gc
from rein.genview.build import NO_THINKING, TRAIN_FILE, record_views
from rein.genview.keys import load_bases
from rein.genview.render import GeneratorView
from rein.genview.scenarios import ScenarioIndex
from rein.schema import TrainingRecord
from rein.train.data import load_records

FIRST = "first"
VIEWS_FILE = "views-first.jsonl"
RECORDS_DIR = "records"
USER = "user"


def first_turn(record: TrainingRecord) -> TrainingRecord | None:
    """The record cut after its first user message, with no reply in progress; None when it has no user message or
    already ends there."""
    conversation = record.state.conversation
    first = next((index for index, message in enumerate(conversation) if message.role == USER), None)
    if first is None or first == len(conversation) - 1:
        return None
    state = record.state.model_copy(update={"conversation": conversation[: first + 1], "draft": None})
    return record.model_copy(update={"state": state})


def is_first(view_id: str) -> bool:
    return view_id.endswith(f"|{FIRST}")


def cut_records(records: list[TrainingRecord], mapping: Mapping[str, object]) -> list[TrainingRecord]:
    """The training records with rule categories, cut at their first user turn."""
    cut = [first_turn(record) for record in records if record.split == TRAIN and record.id in mapping]
    return [record for record in cut if record is not None]


def write_relabelled(
    records: list[TrainingRecord], out: Path, views_file: str, suffix: str, args: argparse.Namespace
) -> int:
    """Changed records in their own folder (same ids, so the teacher reads them in place of the originals) and their
    reply openings rendered with the rules template, told apart by `suffix`; returns the number of views."""
    from transformers import AutoTokenizer

    folder = out / RECORDS_DIR
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / TRAIN_FILE).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")
    template = args.chat_template.read_text(encoding="utf-8")
    view = GeneratorView(AutoTokenizer.from_pretrained(gc.CHAT_TOKENIZER_REPO), chat_template=template)
    bases, index = load_bases(args.bases), ScenarioIndex.from_directory(args.synthetic)
    written = 0
    with (out / views_file).open("w", encoding="utf-8") as handle:
        for record in records:
            for row in record_views(record, bases, index, view, gc.PROMPT_BLOCK, (NO_THINKING,)):
                handle.write(replace(row, view_id=f"{row.view_id}|{suffix}").to_json() + "\n")
                written += 1
    return written


def build(args: argparse.Namespace) -> None:
    mapping = json.loads(args.categories.read_text(encoding="utf-8"))
    cut = cut_records(load_records(args.data / TRAIN_FILE), mapping)
    written = write_relabelled(cut, args.out, VIEWS_FILE, FIRST, args)
    print(json.dumps({"records": len(cut), "views": written, "out": str(args.out)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Training conversations cut at their first user turn, rendered.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--categories", type=Path, required=True)
    parser.add_argument("--bases", type=Path, required=True)
    parser.add_argument("--synthetic", type=Path, required=True)
    parser.add_argument("--chat-template", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
