"""The generator's views of a sheet's examples and of its script: the sheet's persona, scene, bases and tools in the
system prompt, as in the compiled checkpoint's real use."""

import argparse
import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from rein.genview import constants as gc
from rein.genview.build import NO_THINKING, build_views
from rein.genview.render import GeneratorView
from rein.gold.project import GoldProject, build_records, gold_index, sheet_bases
from rein.schema import TrainingRecord
from rein.sheets.model import Sheet, ToolAction, as_gold_sheet, load_sheet
from rein.train.data import load_records

SCRIPT_FILE = "script.jsonl"
BASES_FILE = "bases.jsonl"
TOOL_RULES_FILE = "tool_rules.json"


def sheet_project(sheet: Sheet) -> GoldProject:
    """The sheet with its script as one gold conversation."""
    script = [message.model_dump(by_alias=True) for message in sheet.script]
    return GoldProject.model_validate(
        {
            "format": "rein-gold-project",
            "version": 1,
            "sheets": [as_gold_sheet(sheet)],
            "conversations": [
                {"id": f"{sheet.id}-script", "sheet": sheet.id, "language": sheet.language, "messages": script}
            ],
        }
    )


def script_records(sheet: Sheet) -> list[TrainingRecord]:
    built = build_records(sheet_project(sheet))
    if built.problems:
        raise ValueError(f"the script of {sheet.id} has problems: {built.problems}")
    return built.records


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a sheet's examples and script the way Qwen3.5 sees them.")
    parser.add_argument("--sheet", type=Path, required=True)
    parser.add_argument("--records", type=Path, nargs="*", default=[], help="judged example records")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sheet = load_sheet(args.sheet)
    project = sheet_project(sheet)
    script = script_records(sheet)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    (args.out.parent / SCRIPT_FILE).write_text(
        "".join(record.model_dump_json() + "\n" for record in script), encoding="utf-8"
    )
    records = [record for path in args.records for record in load_records(path)] + script
    view = GeneratorView(AutoTokenizer.from_pretrained(gc.CHAT_TOKENIZER_REPO))
    counts: Counter[str] = Counter()
    with args.out.open("w", encoding="utf-8") as handle:
        for item in build_views(
            records, sheet_bases(project), gold_index(project), view, gc.PROMPT_BASES, [NO_THINKING]
        ):
            if isinstance(item, str):
                counts["without_base"] += 1
                continue
            handle.write(item.to_json() + "\n")
            counts[f"{item.split}:{item.mode}"] += 1
    bases = sheet_bases(project)
    rows = (json.dumps({"key": key, "base": base}) for key, base in bases.items())
    (args.out.parent / BASES_FILE).write_text("".join(f"{row}\n" for row in rows), encoding="utf-8")
    tool = next((rule.id for rule in sheet.rules if isinstance(rule.action, ToolAction)), None)
    mapping = {record.id: tool for record in records if tool is not None and tool in record.questions}
    (args.out.parent / TOOL_RULES_FILE).write_text(json.dumps(mapping), encoding="utf-8")
    print(json.dumps({"records": len(records), "views": dict(counts), "bases": len(bases)}), flush=True)


if __name__ == "__main__":
    main()
