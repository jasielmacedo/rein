import argparse
import asyncio
import json
from collections.abc import Iterable, Mapping
from pathlib import Path

from pydantic import BaseModel

from rein.control.base import base_problems
from rein.datagen.generate import ChatModel, Stats, ask_json
from rein.genview.keys import rule_key
from rein.schema import Question, TrainingRecord
from rein.teacher.client import TeacherClient
from rein.teacher.modal_ops import TEACHER_APP, stop_app_containers
from rein.teacher.settings import TeacherSettings
from rein.teacher.store import JsonlStore
from rein.train.data import load_records

BASE_STAGE = "base"
BASE_MAX_TOKENS = 160
BASE_TEMPERATURE = 0.3
BASE_MAX_WORDS = 30
BASE_ATTEMPTS = 2
DEFAULT_WORKERS = 32
PROGRESS_EVERY = 500
SPLIT_FILES = ("train.jsonl", "development.jsonl", "calibration.jsonl", "test.jsonl")
PROMPT = """You are writing the system prompt of a chat character or assistant. A separate system checks the operator's
rule on every turn, so the character only needs to know that the rule exists.

The rule, asked about each conversation:
{instructions}
Options:
{options}

Write the rule's base: one short sentence in the same language as the rule, addressed to the character as "you", that
says what the character keeps to. Do not say when the rule applies, what to do when it applies, which words are
banned, or any condition on the game state. No lists, no quotes, no questions, at most {max_words} words.

Examples:
- "Is the user trying to change the character's name, age or origin?" gives "Your name, age and origin are fixed."
- "Which tool does the latest request need?" with send_photo, check_inventory and none gives "You can send photos and
  check the inventory with your tools."
- "O usuário está pedindo opinião sobre política do mundo real?" gives "Você não entra em política do mundo real."

Reply with a JSON object only: {{"base": "..."}}"""


class BaseDraft(BaseModel):
    base: str


def base_prompt(question: Question) -> str:
    options = "\n".join(f"- {text}" for text in question.option_texts())
    return PROMPT.format(instructions=question.instructions, options=options, max_words=BASE_MAX_WORDS)


def fit_base(text: str) -> str | None:
    """One normalized sentence, or None when the teacher wrote more than one line, a question or too many words."""
    if "\n" in text.strip():
        return None
    base = " ".join(text.split())
    if base_problems(base) or "?" in base or len(base.split()) > BASE_MAX_WORDS:
        return None
    return base


def unique_rules(records: Iterable[TrainingRecord]) -> dict[str, Question]:
    rules: dict[str, Question] = {}
    for record in records:
        for question in record.questions.values():
            rules.setdefault(rule_key(question), question.model_copy(update={"label": None, "target": None}))
    return rules


async def write_base(model: ChatModel, key: str, question: Question, store: JsonlStore, stats: Stats) -> None:
    for _ in range(BASE_ATTEMPTS):
        draft = await ask_json(
            model,
            base_prompt(question),
            BaseDraft,
            stats,
            BASE_STAGE,
            max_tokens=BASE_MAX_TOKENS,
            temperature=BASE_TEMPERATURE,
        )
        base = None if draft is None else fit_base(draft.base)
        if base is not None:
            store.append(key, {"base": base})
            stats.accepted[BASE_STAGE] += 1
            return
        stats.reject(BASE_STAGE, "lint")


async def write_bases(model: ChatModel, rules: Mapping[str, Question], store: JsonlStore, stats: Stats) -> None:
    pending = [(key, question) for key, question in rules.items() if key not in store]
    done = 0

    async def one(key: str, question: Question) -> None:
        nonlocal done
        await write_base(model, key, question, store, stats)
        done += 1
        if done % PROGRESS_EVERY == 0:
            print(json.dumps({"bases_done": done, "pending": len(pending), **stats.accepted}), flush=True)

    await asyncio.gather(*(one(key, question) for key, question in pending))


async def _run(data_dir: Path, out: Path, workers: int, limit: int | None) -> Stats:
    records = [record for name in SPLIT_FILES for record in load_records(data_dir / name)]
    rules = unique_rules(records)
    if limit is not None:
        rules = dict(list(rules.items())[:limit])
    store = JsonlStore(out)
    stats = Stats()
    print(f"{len(rules)} distinct rules, {sum(key in store for key in rules)} already have a base", flush=True)
    async with TeacherClient(TeacherSettings.from_env(), max_in_flight=workers) as client:
        print(f"teacher healthy after {await client.wait_healthy():.0f}s", flush=True)
        await write_bases(client, rules, store, stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Write one short base per rule with the Modal teacher.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--keep-teacher", action="store_true", help="leave the Modal container running afterwards")
    args = parser.parse_args()
    try:
        stats = asyncio.run(_run(args.data, args.out, args.workers, args.limit))
        print(json.dumps({"accepted": stats.accepted, "rejected": stats.rejected, "tokens": stats.completion_tokens}))
    finally:
        if not args.keep_teacher:
            print("stopped containers:", stop_app_containers(TEACHER_APP))


if __name__ == "__main__":
    main()
