"""What each reply opening must be, so the module learns to act on its decision where the reply starts: open with the
call the turn needs (`<tool_call>` and the function name), not open with a call, or no target (the decision only).

A Rein view takes its target from the record's labels where Rein-r1 agrees on the tool category: the needed tool when
both say the tool rule fires, no call when the agent has tools and both say nothing asks for one. A Nemotron view takes
the dataset's expected next step. A conversation cut at its first user turn (`rein.module.firstturn`) or retold in
another style (`rein.module.restyle`) has no label of its own and takes Rein-r1's decision alone, at the gates
Rein-r1's runtime acts on. Rows hold view ids; the prompts stay in the view files.
"""

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rein.compiler.category_probe import TRAIN
from rein.compiler.tool_confusion import TOOL_CALL
from rein.export.hub.rein_control import DEFAULT_FIRE, TOOL_SUPPRESS_BELOW
from rein.genview import constants as gc
from rein.genview.build import NO_THINKING, ViewRow, load_all_records
from rein.genview.scenarios import ScenarioIndex
from rein.module.material import SUMMARY_FILE, read_rows
from rein.module.pairs import NEEDS_CALL, QUIET, TOOL_SAYS_NONE, situation, situation_kind
from rein.neurons.extract import load_views
from rein.schema import TrainingRecord

OPENINGS_FILE = "openings.jsonl"
AGREEMENT = 0.5
CALL_PREFIX = TOOL_CALL + "\n<function={name}>\n"
EXPECTED_CALL = "function_call"
REIN, EXTERNAL = "rein", "external"

Target = str | bool | None


def opening_target(kind: str, needed_tool: str | None, offers_tools: bool, tool_fire: float | None) -> Target:
    """The tool the reply must open with, False when it must not open with a call, None for no target."""
    if tool_fire is None:
        return None
    if kind == NEEDS_CALL and needed_tool is not None and tool_fire >= AGREEMENT:
        return needed_tool
    if kind in (QUIET, TOOL_SAYS_NONE) and offers_tools and tool_fire < AGREEMENT:
        return False
    return None


def taught_target(fire: Mapping[str, float], tool: Sequence[Any] | None, offers_tools: bool) -> Target:
    """An opening's target from Rein-r1 alone: the likeliest tool where the tool rule fires as surely as Rein-r1's
    runtime needs to open a call, no call where it fires less than the runtime's floor for offering tools and no other
    rule clearly fires, and no target in between or when the agent has no tools."""
    if not offers_tools:
        return None
    tool_fire = fire.get(gc.TOOL_CATEGORY, 0.0)
    if tool_fire >= DEFAULT_FIRE and tool:
        return str(tool[0])
    others = max((chance for name, chance in fire.items() if name != gc.TOOL_CATEGORY), default=0.0)
    if tool_fire < TOOL_SUPPRESS_BELOW and others < AGREEMENT:
        return False
    return None


def opening_row(view_id: str, record_id: str, target: Target, tokenizer: Any) -> dict[str, Any]:
    """`call` holds the call's first tokens for a needed call, and the token to avoid when no call is wanted."""
    call: list[int] = []
    if isinstance(target, str):
        call = tokenizer.encode(CALL_PREFIX.format(name=target), add_special_tokens=False)
    elif target is False:
        call = tokenizer.encode(TOOL_CALL, add_special_tokens=False)[:1]
    calls = None if target is None else bool(target)
    return {"view_id": view_id, "record_id": record_id, "calls": calls, "call": call}


def rein_rows(
    views: Sequence[ViewRow],
    records: Mapping[str, TrainingRecord],
    mapping: Mapping[str, Mapping[str, str]],
    index: ScenarioIndex,
    tool_fire: Mapping[str, float],
    tokenizer: Any,
) -> list[dict[str, Any]]:
    rows = []
    for view in views:
        if view.split != TRAIN or view.mode != NO_THINKING or view.record_id not in mapping:
            continue
        place = situation(records[view.record_id], mapping[view.record_id], index)
        offers = bool(place.tools) or place.needed_tool is not None
        target = opening_target(situation_kind(asdict(place)), place.needed_tool, offers, tool_fire.get(view.view_id))
        rows.append({**opening_row(view.view_id, view.record_id, target, tokenizer), "source": REIN})
    return rows


def external_rows(expected: Sequence[Mapping[str, Any]], tokenizer: Any) -> list[dict[str, Any]]:
    rows = []
    for step in expected:
        if step["split"] != TRAIN:
            continue
        target: Target = step["name"] if step["type"] == EXPECTED_CALL else False
        rows.append({**opening_row(step["view_id"], step["record_id"], target, tokenizer), "source": EXTERNAL})
    return rows


def taught_rows(
    views: Sequence[ViewRow],
    records: Mapping[str, TrainingRecord],
    index: ScenarioIndex,
    taught: Mapping[str, Mapping[str, Any]],
    tokenizer: Any,
) -> list[dict[str, Any]]:
    """Openings of changed conversations (cut at the first turn, or restyled) that Rein-r1 labelled, with its
    decision alone as their target; the source is the view id's last part."""
    rows = []
    for view in views:
        record, row = records.get(view.record_id), taught.get(view.view_id)
        if record is None or row is None:
            continue
        tools = set(info.tools) if (info := index.lookup(record)) else set()
        target = taught_target(row["fire"], row.get("tool"), bool(tools))
        listed = target if not isinstance(target, str) or target in tools else None
        source = view.view_id.rsplit("|", 1)[-1]
        rows.append({**opening_row(view.view_id, view.record_id, listed, tokenizer), "source": source})
    return rows


def build(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(gc.CHAT_TOKENIZER_REPO)
    records = load_all_records(args.data)
    mapping = json.loads(args.categories.read_text(encoding="utf-8"))
    index = ScenarioIndex.from_directory(args.synthetic)
    taught = {row["view_id"]: row for row in read_rows(args.teacher)}
    tool_fire = {view_id: float(row["fire"].get(gc.TOOL_CATEGORY, 0.0)) for view_id, row in taught.items()}
    views = [view for path in args.views for view in load_views(path)]
    rows = rein_rows(views, records, mapping, index, tool_fire, tokenizer) + external_rows(
        read_rows(args.expected), tokenizer
    )
    for views_path, data_path in zip(args.taught_views, args.taught_data, strict=True):
        rows += taught_rows(load_views(views_path), load_all_records(data_path), index, taught, tokenizer)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / OPENINGS_FILE).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    summary = {"rows": len(rows), "by_target": dict(Counter(f"{row['source']}:{row['calls']}" for row in rows))}
    (args.out / SUMMARY_FILE).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="The target of every training reply opening for the module.")
    parser.add_argument("--views", type=Path, nargs="+", required=True, help="the Rein views, short and long")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--categories", type=Path, required=True)
    parser.add_argument("--synthetic", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True, help="Rein-r1's labels per view")
    parser.add_argument("--expected", type=Path, required=True, help="the external views' expected next steps")
    parser.add_argument("--taught-views", type=Path, nargs="*", default=[], help="views of changed conversations")
    parser.add_argument("--taught-data", type=Path, nargs="*", default=[], help="their records' folders, in order")
    parser.add_argument("--out", type=Path, required=True)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
