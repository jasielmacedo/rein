"""How Qwen3.5 calls tools on its own at the reply start of tool-rule cases: the baseline the neurons push on.

For every development case whose rule asks which tool the request needs, it records whether the model opens a tool
call by itself (greedy), which function it names, and which function it names once the opening is forced.
"""

import argparse
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rein.genview.build import ViewRow
from rein.neurons.extract import load_views
from rein.schema import Question, TrainingRecord
from rein.train.data import load_records

NULL_TOOLS = frozenset({"none", "general"})
TOOL_CALL = "<tool_call>"
FORCED_OPENING = "<tool_call>\n<function="
FUNCTION_NAME = re.compile(r"<function=([A-Za-z0-9_.:\-]+)>")
FORCED_NAME = re.compile(r"^([A-Za-z0-9_.:\-]+)>")
NATURAL_TOKENS = 64
FORCED_TOKENS = 24
MAX_MODEL_LEN = 16384
NO_IMAGES = {"image": 0, "video": 0}
DEV_FILE = "development.jsonl"


def tool_question(record: TrainingRecord, prompt_text: str) -> tuple[str, Question] | None:
    """The record's choice rule whose options, apart from "none", are all tools offered in the prompt."""
    for question_id, question in record.questions.items():
        if question.type != "choice" or question.label_index() is None:
            continue
        tools = [key for key in question.option_keys() if key not in NULL_TOOLS]
        if tools and all(f'"name": "{tool}"' in prompt_text for tool in tools):
            return question_id, question
    return None


def _rate(hits: int, total: int) -> float:
    return hits / total if total else float("nan")


def summarize(cases: Sequence[dict[str, Any]]) -> dict[str, float]:
    needed = [case for case in cases if case["label"] not in NULL_TOOLS]
    not_needed = [case for case in cases if case["label"] in NULL_TOOLS]
    called = [case for case in needed if case["natural_called"]]
    return {
        "cases": float(len(cases)),
        "tool_needed": float(len(needed)),
        "called_when_needed": _rate(len(called), len(needed)),
        "right_tool_when_called": _rate(sum(case["natural_name"] == case["label"] for case in called), len(called)),
        "right_tool_when_needed": _rate(sum(case["natural_name"] == case["label"] for case in needed), len(needed)),
        "called_when_not_needed": _rate(sum(case["natural_called"] for case in not_needed), len(not_needed)),
        "right_tool_after_forced_opening": _rate(
            sum(case["forced_name"] == case["label"] for case in needed), len(needed)
        ),
    }


def run(model: str, views: Sequence[ViewRow], records: dict[str, TrainingRecord]) -> list[dict[str, Any]]:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(model=model, max_model_len=MAX_MODEL_LEN, limit_mm_per_prompt=NO_IMAGES, seed=0)
    tokenizer = llm.get_tokenizer()
    picked: list[tuple[ViewRow, str]] = []
    for view in views:
        record = records.get(view.record_id)
        found = tool_question(record, tokenizer.decode(list(view.ids))) if record else None
        if found is not None:
            question = found[1]
            label_index = question.label_index()
            assert label_index is not None
            picked.append((view, question.option_keys()[label_index]))
    opening = tokenizer(FORCED_OPENING, add_special_tokens=False)["input_ids"]
    natural = llm.generate(
        [TokensPrompt(prompt_token_ids=list(view.ids)) for view, _ in picked],
        SamplingParams(temperature=0.0, max_tokens=NATURAL_TOKENS),
    )
    forced = llm.generate(
        [TokensPrompt(prompt_token_ids=[*view.ids, *opening]) for view, _ in picked],
        SamplingParams(temperature=0.0, max_tokens=FORCED_TOKENS),
    )
    cases = []
    for (view, label), free, opened in zip(picked, natural, forced, strict=True):
        text = free.outputs[0].text
        name = FUNCTION_NAME.search(text)
        forced_name = FORCED_NAME.match(opened.outputs[0].text.strip())
        cases.append(
            {
                "view_id": view.view_id,
                "label": label,
                "natural_called": TOOL_CALL in text,
                "natural_name": name.group(1) if name else None,
                "forced_name": forced_name.group(1) if forced_name else None,
            }
        )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure how Qwen3.5 calls tools on its own in tool-rule cases.")
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    records = {record.id: record for record in load_records(args.data / DEV_FILE)}
    views = [view for view in load_views(args.views) if view.split == "development" and view.mode == "nothink"]
    cases = run(args.model, views, records)
    report = {"summary": summarize(cases), "cases": cases}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"]), flush=True)


if __name__ == "__main__":
    main()
