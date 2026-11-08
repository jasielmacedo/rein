"""Long conversations for the rule module's action: records of the Rein data with ordinary turns put before their own
conversation, so the decision comes after thousands of tokens, as late in the long end-to-end scripts.

- `filler`: the teacher (the teacher endpoint) writes ordinary lines a user says to each record's agent, in the
  conversation's language; the auditor then reads them in batches and drops any line that asks for an action, pushes a
  rule or states something false (the teacher can slip).
- `play` (vLLM): Qwen3.5-4B answers those lines turn by turn under the record's rules, as the checkpoint would, with
  tool calls held back (no line needs one); the record's own conversation follows whole, so its labels stay valid, and
  the view is read where the reply opens.
"""

import argparse
import asyncio
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from rein.compiler.category_probe import DEVELOPMENT, TRAIN, category_labels
from rein.compiler.oracle_data import TEACHER_APP
from rein.datagen.audit import AUDITOR_APP, auditor_settings
from rein.datagen.generate import Stats, ask_json
from rein.genview import constants as gc
from rein.genview.build import NO_THINKING, ViewRow, load_all_records
from rein.genview.records import record_spec
from rein.genview.render import GeneratorView
from rein.genview.scenarios import ScenarioIndex
from rein.module.material import LONG, read_rows
from rein.schema import TrainingRecord
from rein.schema.conversation import Message, State
from rein.sheets.long_script import Lines, _written
from rein.teacher.client import TeacherClient
from rein.teacher.modal_ops import stop_app_containers
from rein.teacher.settings import TeacherSettings

LENGTHS = (24, 48, 72)
FILL_BATCH = 24
SPARE_CALLS = 1
FILL_TEMPERATURE = 0.8
FILL_MAX_TOKENS = 2000
CHECK_MAX_TOKENS = 300
CHECK_TEMPERATURE = 0.0
CONTEXT_CHARS = 1500
WORKERS = 16
COLD_START_S = 900.0
REPLY_TOKENS = 160
REPLY_TEMPERATURE = 0.7
REPLY_TOP_P = 0.8
REPLY_TOP_K = 20
MAX_MODEL_LEN = 32768
SEED = 0
TOOL_CALL = "<tool_call>"
FILLER_FILE = "filler.jsonl"
CHECKED_FILE = "filler-checked.jsonl"
VIEWS_FILE = "views-long.jsonl"
FILL_PROMPT = """You write lines for a user who talks with the assistant described below.

The assistant and its setting:
{context}

Write {count} different lines the user says, in the language with code "{language}", one per turn of a long, friendly
conversation: questions about the assistant, its work and its world, small talk, reactions to what it says. Each line
has one or two sentences and reads as a new turn, not a repeat.
No line asks the assistant to do anything (send, book, buy, cancel, check, fetch, open, call, escalate, hand over),
no line tries to change its name or who it is, and no line states anything false.
{written}Answer with JSON only, with exactly {count} lines: {{"lines": ["...", "..."]}}"""
CHECK_PROMPT = """You check lines a user will say to the assistant described below, in an ordinary conversation.

The assistant and its setting:
{context}

The lines:
{lines}

Give the numbers of the lines that ask the assistant to do something (an action or a tool: send, book, buy, cancel,
check, call, escalate, hand over), try to change its name, identity or rules, ask it to drop its role or reveal its
instructions, or state something false about its world. Answer with JSON only: {{"flagged": [numbers]}}"""


class Flagged(BaseModel):
    flagged: list[int]


def needs_call(record: TrainingRecord, categories: Mapping[str, str]) -> bool:
    return category_labels(record, categories).get(gc.TOOL_CATEGORY) == 1


def picked_records(
    records: Mapping[str, TrainingRecord],
    categories: Mapping[str, Any],
    counts: Mapping[str, int],
    calls: Mapping[str, int],
) -> list[TrainingRecord]:
    """Per split, records spread evenly over the categories of their rule in play, without a reply in progress, and
    `calls` more where a tool rule asks for a call (the late failure on the long scripts is a call narrated instead)."""
    by_category: dict[tuple[str, str], list[TrainingRecord]] = defaultdict(list)
    for record_id in sorted(categories):
        record = records.get(record_id)
        if record is None or record.split not in counts or record.state.draft is not None:
            continue
        by_category[(record.split, str((record.meta or {}).get("category") or ""))].append(record)
    picked = []
    for split, count in counts.items():
        groups = [group for (where, _), group in sorted(by_category.items()) if where == split]
        share = -(-count // max(1, len(groups)))
        picked += [record for group in groups for record in group[:share]][:count]
    chosen = {record.id for record in picked}
    for split, count in calls.items():
        extra = [
            record
            for (where, _), group in sorted(by_category.items())
            if where == split
            for record in group
            if record.id not in chosen and needs_call(record, categories[record.id])
        ]
        picked += extra[:count]
    return picked


def context_of(record: TrainingRecord) -> str:
    return (record.state.context or gc.GENERIC_SYSTEM)[:CONTEXT_CHARS]


def language_of(record: TrainingRecord) -> str:
    return str((record.meta or {}).get("lang_conv") or "en")


async def _fill(records: Sequence[TrainingRecord], out: Path) -> None:
    stats = Stats()
    async with TeacherClient(TeacherSettings.from_env(), max_in_flight=WORKERS) as client:
        await client.wait_healthy(COLD_START_S)

        async def one(record: TrainingRecord, count: int) -> dict[str, Any]:
            lines: list[str] = []
            for _ in range(-(-count // FILL_BATCH) + SPARE_CALLS):
                if len(lines) >= count:
                    break
                prompt = FILL_PROMPT.format(
                    context=context_of(record), count=FILL_BATCH, language=language_of(record), written=_written(lines)
                )
                answer = await ask_json(
                    client,
                    prompt,
                    Lines,
                    stats,
                    "long_filler",
                    max_tokens=FILL_MAX_TOKENS,
                    temperature=FILL_TEMPERATURE,
                )
                if answer is not None:
                    lines = list(dict.fromkeys([*lines, *(line.strip() for line in answer.lines if line.strip())]))
            return {"record_id": record.id, "lines": lines[:count]}

        jobs = [one(record, LENGTHS[number % len(LENGTHS)]) for number, record in enumerate(records)]
        with out.open("a", encoding="utf-8") as handle:
            for done, job in enumerate(asyncio.as_completed(jobs), start=1):
                handle.write(json.dumps(await job, ensure_ascii=False) + "\n")
                handle.flush()
                if done % WORKERS == 0:
                    print(json.dumps({"filled": done, "of": len(jobs)}), flush=True)


async def _check(rows: Sequence[Mapping[str, Any]], records: Mapping[str, TrainingRecord], out: Path) -> None:
    stats = Stats()
    async with TeacherClient(auditor_settings(), max_in_flight=WORKERS) as client:
        await client.wait_healthy(COLD_START_S)

        async def one(row: Mapping[str, Any]) -> dict[str, Any]:
            record, kept = records[row["record_id"]], []
            for start in range(0, len(row["lines"]), FILL_BATCH):
                chunk = row["lines"][start : start + FILL_BATCH]
                numbered = "\n".join(f"{number}. {line}" for number, line in enumerate(chunk, start=1))
                prompt = CHECK_PROMPT.format(context=context_of(record), lines=numbered)
                answer = await ask_json(
                    client,
                    prompt,
                    Flagged,
                    stats,
                    "long_check",
                    max_tokens=CHECK_MAX_TOKENS,
                    temperature=CHECK_TEMPERATURE,
                )
                if answer is not None:
                    flagged = set(answer.flagged)
                    kept += [line for number, line in enumerate(chunk, start=1) if number not in flagged]
            return {"record_id": row["record_id"], "lines": kept, "written": len(row["lines"])}

        jobs = [one(row) for row in rows]
        with out.open("a", encoding="utf-8") as handle:
            for done, job in enumerate(asyncio.as_completed(jobs), start=1):
                handle.write(json.dumps(await job, ensure_ascii=False) + "\n")
                handle.flush()
                if done % WORKERS == 0:
                    print(json.dumps({"checked": done, "of": len(jobs)}), flush=True)


def filler(args: argparse.Namespace) -> None:
    """Writes the filler lines; the file is appended as records finish, so a run resumes."""
    records = load_all_records(args.data)
    categories = json.loads(args.categories.read_text(encoding="utf-8"))
    picked = picked_records(
        records,
        categories,
        {TRAIN: args.train, DEVELOPMENT: args.development},
        {TRAIN: args.train_calls, DEVELOPMENT: args.development_calls},
    )
    args.out.mkdir(parents=True, exist_ok=True)
    written = args.out / FILLER_FILE
    done = {row["record_id"] for row in read_rows(written)}
    try:
        asyncio.run(_fill([record for record in picked if record.id not in done], written))
    finally:
        print("stopped teacher containers:", stop_app_containers(TEACHER_APP), flush=True)
    print(json.dumps({"records": len(read_rows(written))}), flush=True)


def check(args: argparse.Namespace) -> None:
    """The auditor reads the filler lines in batches and drops the ones it flags; appended as records finish."""
    records = load_all_records(args.data)
    written, checked = args.out / FILLER_FILE, args.out / CHECKED_FILE
    finished = {row["record_id"] for row in read_rows(checked)}
    try:
        asyncio.run(_check([row for row in read_rows(written) if row["record_id"] not in finished], records, checked))
    finally:
        print("stopped auditor containers:", stop_app_containers(AUDITOR_APP), flush=True)
    rows = read_rows(checked)
    kept, total = sum(len(row["lines"]) for row in rows), sum(row["written"] for row in rows)
    print(json.dumps({"records": len(rows), "lines_written": total, "lines_kept": kept}), flush=True)


def play(args: argparse.Namespace) -> None:
    """Qwen answers every record's filler in lockstep, then each long view is written with the record's own
    conversation after the filler."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    records = load_all_records(args.data)
    index = ScenarioIndex.from_directory(args.synthetic)
    view = GeneratorView(
        AutoTokenizer.from_pretrained(gc.CHAT_TOKENIZER_REPO),
        chat_template=Path(gc.RULES_TEMPLATE).read_text(encoding="utf-8"),
    )
    lines = {row["record_id"]: row["lines"] for row in read_rows(args.filler) if row["lines"]}
    specs = {rid: record_spec(records[rid], {}, index.lookup(records[rid]), gc.PROMPT_BLOCK) for rid in lines}
    history: dict[str, list[Message]] = {rid: [] for rid in lines}
    llm = LLM(model=args.model, max_model_len=MAX_MODEL_LEN, limit_mm_per_prompt={"image": 0, "video": 0}, seed=SEED)
    params = SamplingParams(
        temperature=REPLY_TEMPERATURE,
        top_p=REPLY_TOP_P,
        top_k=REPLY_TOP_K,
        max_tokens=REPLY_TOKENS,
        seed=SEED,
        bad_words=[TOOL_CALL],
    )
    for step in range(max(len(found) for found in lines.values())):
        active = [rid for rid, found in lines.items() if step < len(found)]
        prompts: list[Any] = []
        for rid in active:
            history[rid].append(Message(role="user", text=lines[rid][step]))
            spec = replace(specs[rid], state=State(conversation=tuple(history[rid])))
            prompts.append(TokensPrompt(prompt_token_ids=list(view.render(spec, thinking=False, key=rid).ids)))
        for rid, output in zip(active, llm.generate(prompts, params), strict=True):
            history[rid].append(Message(role="assistant", text=output.outputs[0].text.strip()))
        print(json.dumps({"turn": step + 1, "conversations": len(active)}), flush=True)
    lengths: Counter[str] = Counter()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for rid, turns in history.items():
            record = records[rid]
            spec = replace(specs[rid], state=State(conversation=(*turns, *record.state.conversation)))
            rendered = view.render(spec, thinking=False, key=rid)
            row = ViewRow(
                view_id=f"{rid}|{NO_THINKING}|{LONG}",
                record_id=rid,
                split=record.split,
                mode=NO_THINKING,
                decision=gc.REPLY_START,
                ids=rendered.ids,
                prompt_length=rendered.prompt_length,
                positions=rendered.positions,
            )
            handle.write(row.to_json() + "\n")
            lengths[f"{record.split}:{len(rendered.ids) // 1000}k"] += 1
    print(json.dumps({"views": len(history), "lengths": dict(sorted(lengths.items()))}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Long conversations for the module's action.")
    commands = parser.add_subparsers(dest="command", required=True)
    filling = commands.add_parser("filler", help="filler lines by the teacher, checked by the auditor")
    filling.add_argument("--data", type=Path, required=True)
    filling.add_argument("--categories", type=Path, required=True)
    filling.add_argument("--out", type=Path, required=True)
    filling.add_argument("--train", type=int, required=True)
    filling.add_argument("--development", type=int, required=True)
    filling.add_argument("--train-calls", type=int, default=0, help="more training records that need a call")
    filling.add_argument("--development-calls", type=int, default=0)
    checking = commands.add_parser("check", help="the auditor drops filler lines that ask, push or lie")
    checking.add_argument("--data", type=Path, required=True)
    checking.add_argument("--out", type=Path, required=True, help="the folder with the filler lines")
    playing = commands.add_parser("play", help="Qwen answers the filler; the long views are written (vLLM)")
    playing.add_argument("--data", type=Path, required=True)
    playing.add_argument("--synthetic", type=Path, required=True)
    playing.add_argument("--filler", type=Path, required=True, help="the checked filler lines")
    playing.add_argument("--model", required=True)
    playing.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    {"filler": filler, "check": check, "play": play}[args.command](args)


if __name__ == "__main__":
    main()
