"""How the compiled tool neurons and Qwen's own tool calls combine, case by case, on the development set.

Joins the native measurement (does Qwen open a call by itself, and with which function) with the compiler's yes/no
tool questions and scores a few ways of deciding a call: Qwen alone; the neurons alone, with the opening forced and the
name left to Qwen or taken from the per-tool neurons; and the neurons vetoing Qwen's calls, with or without forcing the
calls they are sure of.
"""

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from rein.compiler.native_tools import NULL_TOOLS
from rein.compiler.tool_binary import ANY_SUFFIX, EACH_SEPARATOR, load_tool_rules

THRESHOLDS = (0.5, 0.7, 0.8, 0.85, 0.9)
VETO_THRESHOLDS = (0.2, 0.5)
LAYER = 20
POSITION = "nothink:reply_start"
YES = 1
UNNAMED = "?"


@dataclass(frozen=True)
class ToolCase:
    """One case of a tool rule: what it needs, what Qwen does by itself, and what the neurons say."""

    label: str
    natural_name: str | None
    forced_name: str | None
    any_tool: float
    each_tool: Mapping[str, float]

    @property
    def needed(self) -> bool:
        return self.label not in NULL_TOOLS

    def named_by_neurons(self) -> str:
        return max(self.each_tool, key=self.each_tool.__getitem__) if self.each_tool else UNNAMED


Decide = Callable[[ToolCase], str | None]


def score(cases: Sequence[ToolCase], decide: Decide) -> dict[str, float]:
    """Calls made, how many were needed (precision) or found (recall), and the same counting the right function."""
    decided = [(case, decide(case)) for case in cases]
    made = [(case, name) for case, name in decided if name is not None]
    needed = sum(case.needed for case in cases)
    right = sum(name == case.label for case, name in made)
    return {
        "calls": float(len(made)),
        "call_precision": sum(case.needed for case, _ in made) / max(1, len(made)),
        "call_recall": sum(case.needed for case, _ in made) / max(1, needed),
        "right_tool_precision": right / max(1, len(made)),
        "right_tool_recall": right / max(1, needed),
    }


def qwen_alone(case: ToolCase) -> str | None:
    return case.natural_name


def neurons_open(threshold: float, case: ToolCase) -> str | None:
    return (case.forced_name or UNNAMED) if case.any_tool >= threshold else None


def neurons_name(threshold: float, case: ToolCase) -> str | None:
    return case.named_by_neurons() if case.any_tool >= threshold else None


def veto(threshold: float, case: ToolCase) -> str | None:
    return case.natural_name if case.any_tool >= threshold else None


def veto_and_force(veto_threshold: float, force_threshold: float, case: ToolCase) -> str | None:
    if case.any_tool >= force_threshold:
        return case.natural_name or case.named_by_neurons()
    return veto(veto_threshold, case)


def policies() -> dict[str, Decide]:
    chosen: dict[str, Decide] = {"qwen_alone": qwen_alone}
    for threshold in THRESHOLDS:
        chosen[f"neurons_open@{threshold}"] = partial(neurons_open, threshold)
        chosen[f"neurons_name@{threshold}"] = partial(neurons_name, threshold)
    for low in VETO_THRESHOLDS:
        chosen[f"veto@{low}"] = partial(veto, low)
        for high in THRESHOLDS:
            if high > low:
                chosen[f"veto@{low}+force@{high}"] = partial(veto_and_force, low, high)
    return chosen


def join(
    native: Sequence[Mapping[str, object]], predictions: Sequence[Mapping[str, object]], tool_rules: Mapping[str, str]
) -> list[ToolCase]:
    """Cases measured natively whose tool rule has yes/no predictions from the compiler."""
    yes: dict[str, dict[str, float]] = {}
    for row in predictions:
        if row["layer"] != LAYER or row["position"] != POSITION:
            continue
        probs = row["probs"]
        assert isinstance(probs, list)
        yes.setdefault(str(row["record_id"]), {})[str(row["question_id"])] = float(probs[YES])
    cases = []
    for entry in native:
        record_id = str(entry["view_id"]).split("|", 1)[0]
        rule = tool_rules.get(record_id)
        answers = yes.get(record_id, {})
        if rule is None or f"{rule}{ANY_SUFFIX}" not in answers:
            continue
        prefix = f"{rule}{EACH_SEPARATOR}"
        called = bool(entry["natural_called"])
        cases.append(
            ToolCase(
                label=str(entry["label"]),
                natural_name=(str(entry["natural_name"] or UNNAMED)) if called else None,
                forced_name=str(entry["forced_name"]) if entry["forced_name"] else None,
                any_tool=answers[f"{rule}{ANY_SUFFIX}"],
                each_tool={key.removeprefix(prefix): value for key, value in answers.items() if key.startswith(prefix)},
            )
        )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Score ways of combining the tool neurons with Qwen's own calls.")
    parser.add_argument("--native", type=Path, required=True, help="native_tools.json")
    parser.add_argument("--predictions", type=Path, required=True, help="the compiler's predictions-development.jsonl")
    parser.add_argument("--tool-rules", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    native = json.loads(args.native.read_text(encoding="utf-8"))["cases"]
    predictions = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line]
    cases = join(native, predictions, load_tool_rules(args.tool_rules))
    report = {
        "cases": len(cases),
        "needed": sum(case.needed for case in cases),
        "policies": {name: score(cases, decide) for name, decide in policies().items()},
    }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"cases": report["cases"], "needed": report["needed"], "written": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
