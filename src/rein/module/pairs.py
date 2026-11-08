"""Material for the rule module's action (option A): Qwen3.5-4B's own replies at each reply opening, judged, and paired
from the same prompt, a reply that did the right thing against one that did not.

A reply does the wrong thing when it breaks the record's rule in play (the auditor's verdict), misses or misnames the
call a tool rule asks for, or opens a call where no rule of the record fires (read from the reply itself). Where some
other rule fires, a call may be the right action (an escalation tool when a human is needed) or the wrong one (a
discount past a limit), and the reply cannot say which, so those calls are left out. Views where every reply did the
right thing keep their first reply as an anchor: the module must leave those replies as Qwen wrote them.
"""

import argparse
import asyncio
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rein.compiler.category_probe import DEVELOPMENT, TRAIN, category_labels, fires
from rein.compiler.category_steer import judged_item
from rein.compiler.tool_confusion import LISTED, TEXT, classify, tool_answer
from rein.datagen.audit import AUDITOR_APP
from rein.genview import constants as gc
from rein.genview.build import NO_THINKING, ViewRow, load_all_records
from rein.genview.scenarios import ScenarioIndex
from rein.module.material import (
    ANCHORS_FILE,
    AUDITOR_FILE,
    PAIRS_FILE,
    SUMMARY_FILE,
    VERDICTS_FILE,
    read_rows,
)
from rein.neurons.extract import load_views
from rein.schema import TrainingRecord
from rein.sheets.steer import Verdict, judge_items
from rein.teacher.modal_ops import stop_app_containers

SAMPLES = 4
MAX_REPLY_TOKENS = 192
SPLITS = (TRAIN, DEVELOPMENT)
KEPT, BROKE, UNCLEAR = "kept", "broke", "unclear"
UNNEEDED_CALL, NEEDED_CALL, RULE = "unneeded_call", "needed_call", "rule"
QUIET, NEEDS_CALL, TOOL_SAYS_NONE, RULE_FIRES, OTHER_FIRES = (
    "quiet",
    "needs_call",
    "tool_says_none",
    "rule_fires",
    "other_rule_fires",
)
JUDGE_CHUNK = 2000

Key = tuple[str, int]


@dataclass(frozen=True)
class Situation:
    """What a view asks of a reply: the rule in play when it fires (judged by the auditor), and the tool call."""

    record_id: str
    target: str
    category: str
    rule_fires: bool
    tools: tuple[str, ...]
    needed_tool: str | None
    quiet: bool


def situation(record: TrainingRecord, categories: Mapping[str, str], index: ScenarioIndex) -> Situation:
    meta = record.meta or {}
    target = str(meta.get("target_rule") or "")
    category = str(meta.get("category") or "")
    info = index.lookup(record)
    labels = category_labels(record, categories)
    tool_fires = labels.get(gc.TOOL_CATEGORY) == 1
    return Situation(
        record_id=record.id,
        target=target,
        category=category,
        rule_fires=category != gc.TOOL_CATEGORY and target in record.questions and fires(record, target) is True,
        tools=tuple(sorted(info.tools)) if info else (),
        needed_tool=tool_answer(record, categories) if tool_fires else None,
        quiet=not any(labels.values()),
    )


def tool_verdict(text: str, place: Situation, rule_types: set[str]) -> str | None:
    """Right or wrong on the tool call (unclear for a call where another rule fires), or None when the agent lists no
    tool and none is needed."""
    if not place.tools and place.needed_tool is None:
        return None
    kind, name = classify(text, set(place.tools), rule_types)
    if place.needed_tool is not None:
        return KEPT if kind == LISTED and name == place.needed_tool else BROKE
    if kind == TEXT:
        return KEPT
    return BROKE if place.quiet else UNCLEAR


def combined(tool: str | None, rule: Verdict | None, rule_fires: bool) -> str:
    """One verdict per reply: wrong on either side breaks it; a rule the auditor could not judge leaves it out."""
    rule_verdict = (rule.verdict if rule else UNCLEAR) if rule_fires else None
    if BROKE in (tool, rule_verdict):
        return BROKE
    if UNCLEAR in (tool, rule_verdict) or (rule is not None and not rule.natural):
        return UNCLEAR
    return KEPT


def picked_views(
    views: Sequence[ViewRow],
    records: Mapping[str, TrainingRecord],
    categories: Mapping[str, Any],
    splits: Sequence[str] = SPLITS,
) -> list[ViewRow]:
    return [
        view
        for view in views
        if view.mode == NO_THINKING
        and view.split in splits
        and view.record_id in records
        and view.record_id in categories
    ]


def sample(args: argparse.Namespace) -> None:
    from rein.genview.replies import sample_replies

    views = load_views(args.views)
    if args.data is None:
        views = [view for view in views if view.mode == NO_THINKING and view.split in args.splits]
    else:
        records = load_all_records(args.data)
        categories = json.loads(args.categories.read_text(encoding="utf-8"))
        views = picked_views(views, records, categories, args.splits)
    replies = sample_replies(args.model, views, args.max_tokens, args.samples)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for view, drawn in zip(views, replies, strict=True):
            row = {"view_id": view.view_id, "record_id": view.record_id, "split": view.split, "prompt": list(view.ids)}
            handle.write(json.dumps({**row, "replies": drawn}) + "\n")
    print(json.dumps({"views": len(views), "samples": args.samples, "out": str(args.out)}), flush=True)


def auditor_verdicts(asked: Mapping[Key, tuple[str, str, str, str]], cache: Path) -> dict[Key, Verdict | None]:
    """The auditor's verdict per (view, reply), in chunks appended to `cache`, so an interrupted run resumes."""
    done: dict[Key, Verdict | None] = {
        (row["view_id"], row["number"]): Verdict(**row["verdict"]) if row["verdict"] else None
        for row in read_rows(cache)
    }
    pending = [key for key in asked if key not in done]
    try:
        for start in range(0, len(pending), JUDGE_CHUNK):
            chunk = pending[start : start + JUDGE_CHUNK]
            verdicts = asyncio.run(judge_items([asked[key] for key in chunk]))
            with cache.open("a", encoding="utf-8") as handle:
                for (view_id, number), verdict in zip(chunk, verdicts, strict=True):
                    saved = verdict.model_dump() if verdict else None
                    handle.write(json.dumps({"view_id": view_id, "number": number, "verdict": saved}) + "\n")
                    done[(view_id, number)] = verdict
            print(json.dumps({"judged": start + len(chunk), "of": len(pending)}), flush=True)
    finally:
        print("stopped containers:", stop_app_containers(AUDITOR_APP), flush=True)
    return done


def scenario_index(synthetic: Path | None, gold_project: Path | None) -> ScenarioIndex:
    """The agents' tools and rule categories: the hand-written gold project's sheets, or the synthetic scenarios."""
    if gold_project is not None:
        from rein.gold.prepare import load_project
        from rein.gold.project import gold_index

        return gold_index(load_project(gold_project))
    if synthetic is None:
        raise ValueError("give the synthetic scenarios or a gold project")
    return ScenarioIndex.from_directory(synthetic)


def judge_samples(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(gc.CHAT_TOKENIZER_REPO)
    records = load_all_records(args.data)
    mapping: dict[str, dict[str, str]] = json.loads(args.categories.read_text(encoding="utf-8"))
    index = scenario_index(args.synthetic, args.gold_project)
    rows = read_rows(args.samples)
    places = {row["view_id"]: situation(records[row["record_id"]], mapping[row["record_id"]], index) for row in rows}
    texts = {
        row["view_id"]: [str(tokenizer.decode(reply, skip_special_tokens=False)) for reply in row["replies"]]
        for row in rows
    }
    tools = {
        row["view_id"]: [
            tool_verdict(text, places[row["view_id"]], set(mapping[row["record_id"]].values()))
            for text in texts[row["view_id"]]
        ]
        for row in rows
    }
    asked = {
        (view_id, number): judged_item(records[place.record_id], place.target, text)
        for view_id, place in places.items()
        if place.rule_fires
        for number, text in enumerate(texts[view_id])
        if tools[view_id][number] in (KEPT, None)
    }
    args.out.mkdir(parents=True, exist_ok=True)
    answers = auditor_verdicts(asked, args.out / AUDITOR_FILE)
    with (args.out / VERDICTS_FILE).open("w", encoding="utf-8") as handle:
        for row in rows:
            view_id = row["view_id"]
            place = places[view_id]
            verdicts = [
                combined(tool, answers.get((view_id, number)), place.rule_fires)
                for number, tool in enumerate(tools[view_id])
            ]
            entry = {"view_id": view_id, "split": row["split"], "situation": asdict(place), "verdicts": verdicts}
            handle.write(json.dumps({**entry, "tools": tools[view_id]}) + "\n")
    print(json.dumps({"views": len(rows), "judged_by_auditor": len(asked)}), flush=True)


def situation_kind(place: Mapping[str, Any]) -> str:
    """Which part of the behavior a view tests, for the per-situation shares."""
    if place["needed_tool"] is not None:
        return NEEDS_CALL
    if place["category"] == gc.TOOL_CATEGORY:
        return TOOL_SAYS_NONE
    if place["rule_fires"]:
        return RULE_FIRES
    return QUIET if place["quiet"] else OTHER_FIRES


def pair_kind(tool: str | None, needed_tool: str | None) -> str:
    """What the wrong reply of a pair got wrong: a call the turn did not need, the call it needed, or the rule."""
    if tool != BROKE:
        return RULE
    return UNNEEDED_CALL if needed_tool is None else NEEDED_CALL


def build(args: argparse.Namespace) -> None:
    """One pair per training view that has both a right and a wrong reply, and an anchor per training view where every
    reply was right; with the share of right replies per split and category, Qwen's own baseline."""
    samples = {row["view_id"]: row for row in read_rows(args.samples)}
    pairs, anchors = [], []
    tallies: dict[str, Counter[str]] = defaultdict(Counter)
    situations: dict[str, Counter[str]] = defaultdict(Counter)
    for judged in read_rows(args.verdicts):
        row, verdicts = samples[judged["view_id"]], judged["verdicts"]
        category = judged["situation"]["category"]
        tallies[f"{judged['split']}:{category}"].update(verdicts)
        situations[f"{judged['split']}:{situation_kind(judged['situation'])}"].update(verdicts)
        if judged["split"] != TRAIN:
            continue
        base = {"view_id": row["view_id"], "record_id": row["record_id"], "category": category, "prompt": row["prompt"]}
        if KEPT in verdicts and BROKE in verdicts:
            chosen, rejected = verdicts.index(KEPT), verdicts.index(BROKE)
            kind = pair_kind(judged["tools"][rejected], judged["situation"]["needed_tool"])
            pairs.append({**base, "kind": kind, "chosen": row["replies"][chosen], "rejected": row["replies"][rejected]})
        elif all(verdict == KEPT for verdict in verdicts):
            anchors.append({**base, "reply": row["replies"][0]})
    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in ((PAIRS_FILE, pairs), (ANCHORS_FILE, anchors)):
        with (args.out / name).open("w", encoding="utf-8") as handle:
            handle.writelines(json.dumps(row) + "\n" for row in rows)
    summary = {
        "pairs": len(pairs),
        "anchors": len(anchors),
        "pairs_by_category": dict(Counter(pair["category"] for pair in pairs)),
        "pairs_by_kind": dict(Counter(pair["kind"] for pair in pairs)),
        "right_share": {key: tally[KEPT] / max(1, sum(tally.values())) for key, tally in sorted(tallies.items())},
        "replies": {key: dict(tally) for key, tally in sorted(tallies.items())},
        "by_situation": {key: dict(tally) for key, tally in sorted(situations.items())},
    }
    (args.out / SUMMARY_FILE).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("pairs", "anchors", "pairs_by_kind")}), flush=True)


def replied_views(args: argparse.Namespace) -> None:
    """Each sampled view with its first reply appended, the form the group 2 check reads (`rein.compiler.sameness`)."""
    with args.out.open("w", encoding="utf-8") as handle:
        for row in read_rows(args.samples):
            ids = (*row["prompt"], *row["replies"][0])
            view = ViewRow(
                view_id=row["view_id"],
                record_id=row["record_id"],
                split=row["split"],
                mode=NO_THINKING,
                decision=gc.REPLY_START,
                ids=ids,
                prompt_length=len(row["prompt"]),
            )
            handle.write(view.to_json() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen's own replies, judged and paired, for the module's action.")
    commands = parser.add_subparsers(dest="command", required=True)
    sampling = commands.add_parser("sample", help="draw replies at every non-thinking reply opening (vLLM)")
    sampling.add_argument("--views", type=Path, required=True)
    sampling.add_argument("--model", required=True)
    sampling.add_argument("--data", type=Path, default=None, help="leave out to keep views of any source")
    sampling.add_argument("--categories", type=Path, default=None)
    sampling.add_argument("--out", type=Path, required=True)
    sampling.add_argument("--samples", type=int, default=SAMPLES)
    sampling.add_argument("--max-tokens", type=int, default=MAX_REPLY_TOKENS)
    sampling.add_argument("--splits", nargs="+", default=list(SPLITS))
    judging = commands.add_parser("judge", help="judge every reply (the auditor for the rule in play)")
    judging.add_argument("--samples", type=Path, required=True)
    judging.add_argument("--data", type=Path, required=True)
    judging.add_argument("--categories", type=Path, required=True)
    judging.add_argument("--synthetic", type=Path, default=None)
    judging.add_argument("--gold-project", type=Path, default=None, help="the gold editor's project, for its tools")
    judging.add_argument("--out", type=Path, required=True, help="folder for the auditor cache and the verdicts")
    building = commands.add_parser("build", help="pairs, anchors and Qwen's baseline")
    building.add_argument("--samples", type=Path, required=True)
    building.add_argument("--verdicts", type=Path, required=True)
    building.add_argument("--out", type=Path, required=True)
    viewing = commands.add_parser("views", help="the sampled views with their first reply, for group 2")
    viewing.add_argument("--samples", type=Path, required=True)
    viewing.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    {"sample": sample, "judge": judge_samples, "build": build, "views": replied_views}[args.command](args)


if __name__ == "__main__":
    main()
