"""Training conversations retold the way other users write, so the module meets more ways of asking for the same
things. The writer is the teacher (the teacher endpoint) or, at the user's choice for its more natural writing, the
auditor's Qwen3.8-27B (`rein-auditor`). Every user message keeps what it asks for, claims and refers to; only the
wording changes, to one of a few generic styles. Assistant and narrator turns stay as they were.

A restyled conversation has no label of its own: Rein-r1 labels it (`rein.module.teacher`) and its confident decisions
become opening targets (`rein.module.openings`), as for the first-turn cuts. Nothing here reads the gold set or the
test sets.
"""

import argparse
import asyncio
import json
import random
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel

from rein.compiler.oracle_data import TEACHER_APP
from rein.datagen.audit import AUDITOR_APP, auditor_settings
from rein.datagen.generate import Stats, ask_json
from rein.genview.build import TRAIN_FILE
from rein.module.firstturn import USER, cut_records, write_relabelled
from rein.schema import TrainingRecord
from rein.teacher.client import TeacherClient
from rein.teacher.modal_ops import stop_app_containers
from rein.teacher.settings import TeacherSettings
from rein.train.data import load_records

RESTYLED = "restyled"
TEACHER, AUDITOR = "teacher", "auditor"
VIEWS_FILE = "views-restyled.jsonl"
STYLES = (
    "very short messages of a few words, little punctuation",
    "casual chat with slang, abbreviations and a few typos",
    "indirect: the user hints at what they want instead of asking for it",
    "the request tucked into small talk about something else",
    "role-play, with the user's actions narrated between asterisks",
    "formal and polite, in longer sentences",
)
RECORDS = 6000
SEED = 11
WORKERS = 16
COLD_START_S = 900.0
TEMPERATURE = 0.8
MAX_TOKENS = 2000

PROMPT = """Rewrite the user's messages of the conversation below in this style: {style}.
Each message must keep exactly what it asks for, claims and refers to, in the same language; change only the wording.
Keep the same number of user messages, in the same order, and leave the other turns alone.
Answer with JSON only: {{"messages": ["the first user message rewritten", "..."]}}

Conversation:
{conversation}"""


class Rewritten(BaseModel):
    messages: list[str]


def conversation_text(record: TrainingRecord) -> str:
    return "\n".join(f"{message.role}: {message.text}" for message in record.state.conversation)


def restyled_record(record: TrainingRecord, texts: Sequence[str]) -> TrainingRecord | None:
    """The record with its user messages replaced in order, or None when the rewrite does not line up with them."""
    conversation = list(record.state.conversation)
    users = [index for index, message in enumerate(conversation) if message.role == USER]
    if len(texts) != len(users) or any(not text.strip() for text in texts):
        return None
    for index, text in zip(users, texts, strict=True):
        conversation[index] = conversation[index].model_copy(update={"text": text.strip()})
    state = record.state.model_copy(update={"conversation": tuple(conversation), "draft": None})
    return record.model_copy(update={"state": state})


async def restyle(records: Sequence[TrainingRecord], seed: int, writer: str) -> list[TrainingRecord]:
    rng = random.Random(seed)
    plan = [(record, rng.choice(STYLES)) for record in records]
    stats = Stats()
    settings = auditor_settings() if writer == AUDITOR else TeacherSettings.from_env()
    async with TeacherClient(settings, max_in_flight=WORKERS) as client:
        await client.wait_healthy(COLD_START_S)

        async def one(record: TrainingRecord, style: str) -> TrainingRecord | None:
            prompt = PROMPT.format(style=style, conversation=conversation_text(record))
            answer = await ask_json(
                client, prompt, Rewritten, stats, "restyle", max_tokens=MAX_TOKENS, temperature=TEMPERATURE
            )
            return restyled_record(record, answer.messages) if answer is not None else None

        rewritten = await asyncio.gather(*(one(record, style) for record, style in plan))
    print(json.dumps({"teacher_rejected": dict(stats.rejected)}), flush=True)
    return [record for record in rewritten if record is not None]


def build(args: argparse.Namespace) -> None:
    mapping = json.loads(args.categories.read_text(encoding="utf-8"))
    pool = [record for record in load_records(args.data / TRAIN_FILE) if record.id in mapping]
    chosen = random.Random(args.seed).sample(pool, min(args.records, len(pool)))
    cut_ids = {record.id for record in cut_records(chosen, mapping)}
    try:
        restyled = asyncio.run(restyle(chosen, args.seed, args.writer))
    finally:
        app = AUDITOR_APP if args.writer == AUDITOR else TEACHER_APP
        print("stopped containers:", stop_app_containers(app), flush=True)
    written = write_relabelled(restyled, args.out, VIEWS_FILE, RESTYLED, args)
    counts = Counter("several user turns" if record.id in cut_ids else "one user turn" for record in restyled)
    print(json.dumps({"asked": len(chosen), "restyled": len(restyled), "views": written, **counts}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Training conversations retold in other user styles, rendered.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--categories", type=Path, required=True)
    parser.add_argument("--bases", type=Path, required=True)
    parser.add_argument("--synthetic", type=Path, required=True)
    parser.add_argument("--chat-template", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--records", type=int, default=RECORDS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--writer", choices=(TEACHER, AUDITOR), default=TEACHER)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
