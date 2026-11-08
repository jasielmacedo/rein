"""Rein-r1 as the teacher of the rule module, a first model making a second: the phase 1 decision model (Qwen3.5-4B
Base, a LoRA and a set-pointer head, run in PyTorch only to make training data) reads each view's conversation and
gives, per rule, the chance of each option. Per category, the chance that some rule of it fires (one minus its "does
not fire" option, the highest over the category's rules) becomes the module's soft label at that view's reply opening.

Long views carry ordinary turns before the record's own conversation. They are read back from the view's tokens: the
filler lines are known, and Qwen's replies sit between them.
"""

import argparse
import json
import random
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rein.compiler.constants import NO_NULL
from rein.compiler.model import null_option_index
from rein.genview import constants as gc
from rein.genview.build import NO_THINKING, ViewRow, load_all_records
from rein.module.material import is_long, read_rows
from rein.neurons.extract import load_views
from rein.schema import TrainingRecord
from rein.schema.conversation import Message

TEACHER_FILE = "teacher.jsonl"
BATCH_TOKENS = 32768
MAX_RECORDS = 32
SEED = 0
TURN = re.compile(r"<\|im_start\|>(user|assistant)\n(.*?)<\|im_end\|>", re.DOTALL)
EMPTY_THINKING = re.compile(r"^<think>\s*</think>\s*", re.DOTALL)


def filler_history(text: str, lines: Sequence[str]) -> list[Message]:
    """The filler turns of a long view's text: each known user line and the assistant reply that follows it."""
    turns = TURN.findall(text)
    history: list[Message] = []
    for index, line in enumerate(lines):
        role, user = turns[2 * index]
        reply_role, reply = turns[2 * index + 1]
        if role != "user" or reply_role != "assistant" or user.strip() != line.strip():
            raise ValueError(f"turn {index} of the long view does not match its filler line")
        history.append(Message(role="user", text=user.strip()))
        history.append(Message(role="assistant", text=EMPTY_THINKING.sub("", reply).strip()))
    return history


def with_history(record: TrainingRecord, history: Sequence[Message]) -> TrainingRecord:
    state = record.state.model_copy(update={"conversation": (*history, *record.state.conversation)})
    return record.model_copy(update={"state": state})


def category_fire(
    probabilities: Mapping[str, tuple[Sequence[str], Sequence[float]]], categories: Mapping[str, str]
) -> dict[str, float]:
    """Per category: the highest chance, over its rules, that the rule fires (rules without a null option skipped)."""
    fire: dict[str, float] = {}
    for question_id, (keys, probs) in probabilities.items():
        category, null = categories.get(question_id), null_option_index(keys)
        if category is None or null == NO_NULL:
            continue
        fire[category] = max(fire.get(category, 0.0), 1.0 - float(probs[null]))
    return fire


def likeliest_tool(
    probabilities: Mapping[str, tuple[Sequence[str], Sequence[float]]], categories: Mapping[str, str]
) -> tuple[str, float] | None:
    """The tool option with the highest chance over the record's tool rules (their "none" option left out); a tool
    rule's option keys are the tool names."""
    best: tuple[str, float] | None = None
    for question_id, (keys, probs) in probabilities.items():
        if categories.get(question_id) != gc.TOOL_CATEGORY:
            continue
        null = null_option_index(keys)
        for index, (key, chance) in enumerate(zip(keys, probs, strict=False)):
            if index != null and (best is None or float(chance) > best[1]):
                best = (str(key), float(chance))
    return best


def teacher_records(
    views: Sequence[ViewRow],
    records: Mapping[str, TrainingRecord],
    filler: Mapping[str, Sequence[str]],
    tokenizer: Any,
) -> dict[str, TrainingRecord]:
    """The record each view's reply opening shows, by view: as it is, or with the long view's filler turns first."""
    found: dict[str, TrainingRecord] = {}
    for view in views:
        record = records.get(view.record_id)
        if record is None or view.mode != NO_THINKING:
            continue
        lines = filler.get(view.record_id) if is_long(view.view_id) else None
        if lines:
            text = str(tokenizer.decode(list(view.ids), skip_special_tokens=False))
            record = with_history(record, filler_history(text, lines))
        found[view.view_id] = record
    return found


def label(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoTokenizer

    from rein.model.loader import load_decider
    from rein.train.data import token_batches

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = load_all_records(args.data)
    mapping: dict[str, dict[str, str]] = json.loads(args.categories.read_text(encoding="utf-8"))
    filler = {row["record_id"]: row["lines"] for path in args.filler for row in read_rows(path)}
    tokenizer = AutoTokenizer.from_pretrained(gc.CHAT_TOKENIZER_REPO)
    views = [view for path in args.views for view in load_views(path)]
    by_view = teacher_records(views, records, filler, tokenizer)
    done = {row["view_id"] for row in read_rows(args.out)}
    pending = [view_id for view_id in by_view if view_id not in done]
    decider = load_decider(args.teacher, device).eval()
    encoded = [decider.encoder.encode(by_view[view_id]) for view_id in pending]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad(), args.out.open("a", encoding="utf-8") as handle:
        batches = token_batches(encoded, BATCH_TOKENS, MAX_RECORDS, random.Random(SEED))
        for number, batch in enumerate(batches, start=1):
            output = decider.decide_encoded([encoded[index] for index in batch])
            rows = [(pending[index], question) for index in batch for question in encoded[index].questions]
            per_view: dict[str, dict[str, tuple[Sequence[str], Sequence[float]]]] = {}
            for (view_id, question), probs in zip(rows, output.probs, strict=True):
                keys = question.option_keys
                per_view.setdefault(view_id, {})[question.question_id] = (keys, probs[: len(keys)].tolist())
            for view_id, probabilities in per_view.items():
                record = by_view[view_id]
                rules = mapping.get(record.id, {})
                fire = category_fire(probabilities, rules)
                row = {"view_id": view_id, "record_id": record.id, "fire": fire}
                handle.write(json.dumps({**row, "tool": likeliest_tool(probabilities, rules)}) + "\n")
            handle.flush()
            print(json.dumps({"batches": number, "of": len(batches)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rein-r1's soft labels per category at each view's reply opening.")
    parser.add_argument("--teacher", required=True, help="the Rein-r1 checkpoint folder")
    parser.add_argument("--views", type=Path, nargs="+", required=True)
    parser.add_argument("--filler", type=Path, nargs="*", default=[], help="filler lines of the long views")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--categories", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    label(parser.parse_args())


if __name__ == "__main__":
    main()
