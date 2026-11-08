"""Decision points from public conversational tool-use data: NVIDIA's Nemotron-RL-Agentic-Conversational-Tool-Use-
Pivot-v1 (CC-BY-4.0, written with open models). Each one holds a customer-service policy in the system prompt, the
tools, the conversation so far with its calls and their results, and the step expected next: a call or a reply in text.

`prepare` turns them into views for Qwen3.5 (the checkpoint's template, its native tool format, no thinking), split by
conversation so no conversation crosses splits. `judge` compares Qwen's own replies drawn there with the expected step
by the tool's name, then asks the auditor to confirm each training pair (the reference can be wrong too): the right
reply must be a right step and the wrong one a wrong step, or the pair is left out.
"""

import argparse
import asyncio
import hashlib
import json
import random
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from rein.compiler.category_probe import DEVELOPMENT, TRAIN
from rein.compiler.tool_confusion import LISTED, TEXT, classify
from rein.datagen.audit import AUDITOR_APP, auditor_settings
from rein.datagen.generate import Stats, ask_json
from rein.genview import constants as gc
from rein.genview.build import NO_THINKING, ViewRow
from rein.module.material import AUDITOR_FILE, VERDICTS_FILE, read_rows
from rein.module.pairs import BROKE, JUDGE_CHUNK, KEPT, UNCLEAR
from rein.teacher.client import TeacherClient
from rein.teacher.modal_ops import stop_app_containers

SOURCE_REPO = "nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1"
SOURCE_FILE = "train.jsonl"
CATEGORY = "external_tool_use"
CALL, MESSAGE, REASONING, OUTPUT = "function_call", "message", "reasoning", "function_call_output"
EXPECTED_FILE = "expected.jsonl"
GRADES_CACHE = "grades-auditor.jsonl"
GRADES_FILE = "grades.json"
DEVELOPMENT_SHARE = 0.1
LONGEST_SHARE = 0.5
SEED = 0
POLICY_CHARS = 2000
HISTORY_ITEMS = 8
ITEM_CHARS = 400
JUDGE_WORKERS = 32
JUDGE_MAX_TOKENS = 200
JUDGE_TEMPERATURE = 0.0
COLD_START_S = 900.0
STEP_PROMPT = """You check one reply of a customer-service assistant that can call tools.

The assistant's policy (its system prompt, shortened):
{policy}

The tools it has:
{tools}

The latest turns of the conversation:
{history}

A reference says the right next step is:
{expected}

The assistant's reply:
{reply}

Is the reply a right next step under the policy (a call to the right tool with sensible arguments, or the right
message in text), or a wrong one (a call that is not needed, the wrong tool, wrong arguments, an action described in
text instead of called, or a reply that breaks the policy)? The reference can be wrong: judge the reply itself, and
say whether the reference is right.
Answer with JSON only: {{"right": true | false, "reference_right": true | false}}"""


class StepVerdict(BaseModel):
    right: bool
    reference_right: bool


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if isinstance(part, Mapping))


def chat_messages(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The conversation in the chat-template form: reasoning dropped, calls in a row merged into one assistant turn."""
    messages: list[dict[str, Any]] = []
    for item in items:
        kind = item.get("type") or MESSAGE
        if kind == REASONING:
            continue
        if kind == CALL:
            call = {"type": "function", "function": {"name": item["name"], "arguments": _arguments(item)}}
            if messages and messages[-1]["role"] == "assistant" and messages[-1].get("tool_calls"):
                messages[-1]["tool_calls"].append(call)
            else:
                messages.append({"role": "assistant", "content": "", "tool_calls": [call]})
        elif kind == OUTPUT:
            messages.append({"role": "tool", "content": str(item.get("output", "")).strip()})
        else:
            messages.append({"role": item["role"], "content": _text(item.get("content", ""))})
    return messages


def _arguments(item: Mapping[str, Any]) -> dict[str, Any]:
    try:
        parsed = json.loads(item.get("arguments") or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def chat_tools(tools: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {}),
            },
        }
        for tool in tools
    ]


def split_of(trajectory: str) -> str:
    digest = int(hashlib.sha256(trajectory.encode("utf-8")).hexdigest()[:8], 16)
    return DEVELOPMENT if digest / 0xFFFFFFFF < DEVELOPMENT_SHARE else TRAIN


def _visible_chars(items: Sequence[Mapping[str, Any]]) -> int:
    return sum(len(json.dumps(item)) for item in items if item.get("type") != REASONING)


def candidates(source: Path) -> Iterator[tuple[int, str, str, int]]:
    """(line, split, expected kind, visible size) of every decision point, without keeping the records."""
    with source.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle):
            row = json.loads(line)
            items = row["responses_create_params"]["input"]
            yield number, split_of(str(row["trajectory_id"])), row["expected_action"]["type"], _visible_chars(items)


def chosen_lines(source: Path, quotas: Mapping[tuple[str, str], int]) -> set[int]:
    """Per split and expected kind, half the quota from the longest conversations and half drawn at random."""
    pools: dict[tuple[str, str], list[tuple[int, int]]] = {key: [] for key in quotas}
    for number, split, kind, size in candidates(source):
        if (split, kind) in pools:
            pools[(split, kind)].append((size, number))
    rng = random.Random(SEED)
    picked: set[int] = set()
    for key, pool in pools.items():
        quota = min(quotas[key], len(pool))
        longest = sorted(pool, reverse=True)[: int(quota * LONGEST_SHARE)]
        rest = [entry for entry in pool if entry not in set(longest)]
        drawn = rng.sample(rest, min(quota - len(longest), len(rest)))
        picked.update(number for _, number in (*longest, *drawn))
    return picked


def brief(items: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """What the auditor reads: the policy, the tools and the latest turns, shortened."""
    visible = [item for item in items if item.get("type") != REASONING]
    policy = next((_text(item.get("content", "")) for item in visible if item.get("role") == "system"), "")
    lines = []
    for item in visible[-HISTORY_ITEMS:]:
        kind = item.get("type") or MESSAGE
        if kind == CALL:
            lines.append(f"assistant called {item['name']}({item.get('arguments', '')})"[:ITEM_CHARS])
        elif kind == OUTPUT:
            lines.append(f"tool result: {str(item.get('output', '')).strip()}"[:ITEM_CHARS])
        elif item.get("role") != "system":
            lines.append(f"{item['role']}: {_text(item.get('content', ''))}"[:ITEM_CHARS])
    names = "\n".join(f"- {tool['name']}: {tool.get('description', '')}"[:ITEM_CHARS] for tool in tools)
    return {"policy": policy[:POLICY_CHARS], "tools_text": names, "history": "\n".join(lines)}


def expected_text(action: Mapping[str, Any]) -> str:
    if action["type"] == CALL:
        return f"call {action['name']}({action.get('arguments', '')})"[:ITEM_CHARS]
    return f"reply in text: {_text(action.get('content', ''))}"[:ITEM_CHARS]


def encoded(tokenizer: Any, messages: list[dict[str, Any]], tools: list[dict[str, Any]], template: str) -> list[int]:
    """The prompt's token ids up to the reply opening, non-thinking."""
    output = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        chat_template=template,
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
    )
    ids = output["input_ids"] if isinstance(output, Mapping) else output
    return [int(token) for token in ids]


def prepare(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(gc.CHAT_TOKENIZER_REPO)
    template = Path(gc.RULES_TEMPLATE).read_text(encoding="utf-8")
    quotas = {
        (TRAIN, CALL): args.train_calls,
        (TRAIN, MESSAGE): args.train_messages,
        (DEVELOPMENT, CALL): args.development_calls,
        (DEVELOPMENT, MESSAGE): args.development_messages,
    }
    picked = chosen_lines(args.source, quotas)
    args.out.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    with (
        args.source.open(encoding="utf-8") as source,
        (args.out / args.views_name).open("w", encoding="utf-8") as views,
        (args.out / EXPECTED_FILE).open("w", encoding="utf-8") as expected,
    ):
        for number, line in enumerate(source):
            if number not in picked:
                continue
            row = json.loads(line)
            params = row["responses_create_params"]
            items, tools = params["input"], params.get("tools", [])
            ids = encoded(tokenizer, chat_messages(items), chat_tools(tools), template)
            split, action = split_of(str(row["trajectory_id"])), row["expected_action"]
            record_id = f"nemotron:{row['trajectory_id']}:{number}"
            view = ViewRow(
                view_id=f"{record_id}|{NO_THINKING}",
                record_id=record_id,
                split=split,
                mode=NO_THINKING,
                decision=gc.REPLY_START,
                ids=tuple(ids),
                prompt_length=len(ids),
                positions={gc.REPLY_START: (len(ids) - 1,)},
            )
            views.write(view.to_json() + "\n")
            entry = {
                "view_id": view.view_id,
                "record_id": record_id,
                "split": split,
                "type": action["type"],
                "name": action.get("name"),
                "tools": [tool["name"] for tool in tools],
                "expected": expected_text(action),
                **brief(items, tools),
            }
            expected.write(json.dumps(entry, ensure_ascii=False) + "\n")
            counts[f"{split}:{action['type']}"] += 1
    print(json.dumps(dict(counts)), flush=True)


def name_verdict(text: str, entry: Mapping[str, Any]) -> str:
    """Right or wrong by the expected step's kind and the tool's name alone."""
    kind, name = classify(text, set(entry["tools"]), set())
    if entry["type"] == CALL:
        return KEPT if kind == LISTED and name == entry["name"] else BROKE
    return KEPT if kind == TEXT else BROKE


async def _confirm(items: Sequence[tuple[Mapping[str, Any], str]]) -> list[StepVerdict | None]:
    stats = Stats()
    async with TeacherClient(auditor_settings(), max_in_flight=JUDGE_WORKERS) as client:
        await client.wait_healthy(COLD_START_S)

        async def one(entry: Mapping[str, Any], reply: str) -> StepVerdict | None:
            prompt = STEP_PROMPT.format(
                policy=entry["policy"],
                tools=entry["tools_text"],
                history=entry["history"],
                expected=entry["expected"],
                reply=reply,
            )
            return await ask_json(
                client,
                prompt,
                StepVerdict,
                stats,
                "external_step",
                max_tokens=JUDGE_MAX_TOKENS,
                temperature=JUDGE_TEMPERATURE,
            )

        return list(await asyncio.gather(*(one(entry, reply) for entry, reply in items)))


def confirmed(asked: Mapping[tuple[str, int], tuple[Mapping[str, Any], str]], cache: Path) -> dict:
    """The auditor's verdict per (view, reply), in chunks appended to `cache`, so an interrupted run resumes."""
    done = {(row["view_id"], row["number"]): row["verdict"] for row in read_rows(cache)}
    pending = [key for key in asked if key not in done]
    try:
        for start in range(0, len(pending), JUDGE_CHUNK):
            chunk = pending[start : start + JUDGE_CHUNK]
            verdicts = asyncio.run(_confirm([asked[key] for key in chunk]))
            with cache.open("a", encoding="utf-8") as handle:
                for (view_id, number), verdict in zip(chunk, verdicts, strict=True):
                    saved = verdict.model_dump() if verdict else None
                    handle.write(json.dumps({"view_id": view_id, "number": number, "verdict": saved}) + "\n")
                    done[(view_id, number)] = saved
            print(json.dumps({"confirmed": start + len(chunk), "of": len(pending)}), flush=True)
    finally:
        print("stopped containers:", stop_app_containers(AUDITOR_APP), flush=True)
    return done


def judge(args: argparse.Namespace) -> None:
    """Verdicts in the form `rein.module.pairs build` reads: by name for every reply, and for the training views that
    have both outcomes, the auditor's confirmation of one reply of each (left out as unclear when it disagrees)."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(gc.CHAT_TOKENIZER_REPO)
    expected = {row["view_id"]: row for row in read_rows(args.expected)}
    rows = read_rows(args.samples)
    texts = {
        row["view_id"]: [str(tokenizer.decode(reply, skip_special_tokens=False)) for reply in row["replies"]]
        for row in rows
    }
    by_name = {
        view_id: [name_verdict(text, expected[view_id]) for text in replies] for view_id, replies in texts.items()
    }
    asked: dict[tuple[str, int], tuple[Mapping[str, Any], str]] = {}
    for row in rows:
        names = by_name[row["view_id"]]
        if row["split"] != TRAIN or KEPT not in names or BROKE not in names:
            continue
        for number in (names.index(KEPT), names.index(BROKE)):
            asked[(row["view_id"], number)] = (expected[row["view_id"]], texts[row["view_id"]][number])
    args.out.mkdir(parents=True, exist_ok=True)
    answers = confirmed(asked, args.out / AUDITOR_FILE) if asked else {}
    reference: Counter[str] = Counter()
    with (args.out / VERDICTS_FILE).open("w", encoding="utf-8") as handle:
        for row in rows:
            entry, names = expected[row["view_id"]], by_name[row["view_id"]]
            verdicts = list(names)
            if any((row["view_id"], number) in asked for number in range(len(names))):
                verdicts = [UNCLEAR] * len(names)
                for number, name in enumerate(names):
                    if (row["view_id"], number) not in asked:
                        continue
                    answer = answers.get((row["view_id"], number))
                    reference[str(answer["reference_right"]) if answer else "none"] += 1
                    if answer is not None and answer["right"] == (name == KEPT):
                        verdicts[number] = name
            needed = entry["name"] if entry["type"] == CALL else None
            situation = {
                "record_id": row["record_id"],
                "category": CATEGORY,
                "rule_fires": False,
                "needed_tool": needed,
                "quiet": needed is None,
            }
            line = {"view_id": row["view_id"], "split": row["split"], "situation": situation, "verdicts": verdicts}
            handle.write(json.dumps({**line, "tools": names}) + "\n")
    print(
        json.dumps({"views": len(rows), "confirmed_by_auditor": len(asked), "reference": dict(reference)}), flush=True
    )


def grade(args: argparse.Namespace) -> None:
    """Every development reply graded twice: by the expected step's kind and tool name, and by the auditor, which also
    reads the arguments and the policy; the shares right per expected kind."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(gc.CHAT_TOKENIZER_REPO)
    expected = {row["view_id"]: row for row in read_rows(args.expected)}
    asked = {
        (row["view_id"], number): (expected[row["view_id"]], str(tokenizer.decode(reply, skip_special_tokens=False)))
        for row in read_rows(args.samples)
        if row["split"] == DEVELOPMENT
        for number, reply in enumerate(row["replies"])
    }
    args.out.mkdir(parents=True, exist_ok=True)
    answers = confirmed(asked, args.out / GRADES_CACHE)
    tallies: dict[str, Counter[str]] = {}
    for key, (entry, reply) in asked.items():
        tally = tallies.setdefault(entry["type"], Counter())
        answer = answers.get(key)
        tally["replies"] += 1
        tally["right_by_name"] += int(name_verdict(reply, entry) == KEPT)
        tally["judged"] += int(answer is not None)
        tally["right_by_auditor"] += int(bool(answer and answer["right"]))
    report = {
        kind: {
            "replies": tally["replies"],
            "right_by_name": tally["right_by_name"] / max(1, tally["replies"]),
            "right_by_auditor": tally["right_by_auditor"] / max(1, tally["judged"]),
        }
        for kind, tally in sorted(tallies.items())
    }
    (args.out / GRADES_FILE).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Public conversational tool-use decision points for the module.")
    commands = parser.add_subparsers(dest="command", required=True)
    preparing = commands.add_parser("prepare", help="views and expected steps from the source file")
    preparing.add_argument("--source", type=Path, required=True)
    preparing.add_argument("--out", type=Path, required=True)
    preparing.add_argument("--views-name", default="views.jsonl")
    preparing.add_argument("--train-calls", type=int, required=True)
    preparing.add_argument("--train-messages", type=int, required=True)
    preparing.add_argument("--development-calls", type=int, required=True)
    preparing.add_argument("--development-messages", type=int, required=True)
    judging = commands.add_parser("judge", help="verdicts by name, training pairs confirmed by the auditor")
    judging.add_argument("--samples", type=Path, required=True)
    judging.add_argument("--expected", type=Path, required=True)
    judging.add_argument("--out", type=Path, required=True)
    grading = commands.add_parser("grade", help="every development reply graded by name and by the auditor")
    grading.add_argument("--samples", type=Path, required=True)
    grading.add_argument("--expected", type=Path, required=True)
    grading.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    {"prepare": prepare, "judge": judge, "grade": grade}[args.command](args)


if __name__ == "__main__":
    main()
