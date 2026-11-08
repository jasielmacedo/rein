"""How each way of passing the rules changes Qwen3.5's own tool calls (DR-026).

The same development records are rendered several ways (the bases only, the whole rules as prose, the rules block,
the rules block that says rules are not functions) and Qwen3.5-4B replies once to each with its non-thinking sampling.
Each reply is plain text, a call to one of the agent's tools, a call named after a rule type, or a call to some other
name; cases where the agent's tool rule fires are counted apart (there a call is needed, and the right tool is known).
"""

import argparse
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rein.compiler.category_probe import category_labels
from rein.genview import constants as gc
from rein.genview.build import load_all_records
from rein.genview.scenarios import ScenarioIndex
from rein.neurons.extract import load_views
from rein.schema import TrainingRecord

VARIANT_SEPARATOR = "|"
TOOL_CALL = "<tool_call>"
FUNCTION_NAME = re.compile(r"<function=([^>\s]+)>")
TEXT, LISTED, RULE_TYPE, OTHER = "text", "listed", "rule_type", "other"
KINDS = (TEXT, LISTED, RULE_TYPE, OTHER)
REPORT_FILE = "tool_confusion.json"


def classify(reply: str, tools: set[str], rule_types: set[str]) -> tuple[str, str]:
    """What the reply opens with, and the function it names (empty for text)."""
    if not reply.lstrip().startswith(TOOL_CALL):
        return TEXT, ""
    names = FUNCTION_NAME.findall(reply)
    name = names[0] if names else ""
    if name in tools:
        return LISTED, name
    normalized = name.lower().removeprefix("rule_")
    return (RULE_TYPE if normalized in rule_types else OTHER), name


def tool_answer(record: TrainingRecord, categories: Mapping[str, str]) -> str | None:
    """The tool the record's tool rule asks for, None when no tool rule fires there."""
    for question_id, category in categories.items():
        question = record.questions.get(question_id)
        if category != gc.TOOL_CATEGORY or question is None or (index := question.label_index()) is None:
            continue
        key = question.option_keys()[index]
        if key not in gc.NULL_OPTIONS:
            return key
    return None


def summarize(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Per variant: on cases that need a call, how often one opens and names the right tool; on the others, how the
    replies split into text and the three kinds of call."""
    report: dict[str, Any] = {}
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_variant[case["variant"]].append(case)
    for variant, mine in by_variant.items():
        needed = [case for case in mine if case["answer"]]
        quiet = [case for case in mine if not case["answer"]]
        report[variant] = {
            "needed": {
                "n": len(needed),
                "opened": sum(case["kind"] != TEXT for case in needed) / max(1, len(needed)),
                "right_tool": sum(case["name"] == case["answer"] for case in needed) / max(1, len(needed)),
            },
            "not_needed": {
                "n": len(quiet),
                **{kind: sum(c["kind"] == kind for c in quiet) / max(1, len(quiet)) for kind in KINDS},
            },
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify Qwen's replies under each way of passing the rules.")
    parser.add_argument("--replied", type=Path, required=True, help="views with Qwen's reply, ids as VARIANT|view")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--synthetic", type=Path, required=True)
    parser.add_argument("--categories", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(gc.CHAT_TOKENIZER_REPO)
    records = load_all_records(args.data)
    index = ScenarioIndex.from_directory(args.synthetic)
    mapping: dict[str, dict[str, str]] = json.loads(args.categories.read_text(encoding="utf-8"))
    cases = []
    for view in load_views(args.replied):
        record = records[view.record_id]
        info = index.lookup(record)
        categories = mapping.get(record.id, {})
        reply = str(tokenizer.decode(list(view.ids[view.prompt_length :]), skip_special_tokens=False))
        kind, name = classify(reply, set(info.tools) if info else set(), set(categories.values()))
        fired = category_labels(record, categories).get(gc.TOOL_CATEGORY) == 1
        cases.append(
            {
                "variant": view.view_id.split(VARIANT_SEPARATOR, 1)[0],
                "record_id": record.id,
                "kind": kind,
                "name": name,
                "answer": tool_answer(record, categories) if fired else None,
            }
        )
    report = summarize(cases)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
