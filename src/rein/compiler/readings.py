import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from rein.compiler.reader import RuleReader, RuleReading
from rein.compiler.tool_binary import derived_bases, load_tool_rules, with_tool_questions
from rein.genview.build import load_all_records
from rein.genview.keys import load_bases, rule_key
from rein.model.backbone import load_text_backbone, placed, truncate_backbone
from rein.model.encode import RequestEncoder
from rein.schema import Question, TrainingRecord

READINGS_FILE = "readings.safetensors"
KEYS_FILE = "readings.json"
DEFAULT_BATCH_RULES = 64
STORE_DTYPE = torch.float16


def rules_with_bases(records: Iterable[TrainingRecord], bases: Mapping[str, str]) -> dict[str, tuple[Question, str]]:
    rules: dict[str, tuple[Question, str]] = {}
    for record in records:
        for question in record.questions.values():
            key = rule_key(question)
            if key in bases:
                rules.setdefault(key, (question.model_copy(update={"label": None, "target": None}), bases[key]))
    return rules


def _padded(reading: RuleReading, width: int) -> RuleReading:
    extra = width - reading.options.shape[1]
    options = torch.nn.functional.pad(reading.options, (0, 0, 0, extra))
    mask = torch.nn.functional.pad(reading.mask, (0, extra), value=False)
    return RuleReading(reading.query, options, mask, reading.option_keys)


def read_all(reader: RuleReader, rules: Sequence[tuple[Question, str]], batch_rules: int) -> RuleReading:
    parts = [reader.read(rules[start : start + batch_rules]) for start in range(0, len(rules), batch_rules)]
    width = max(part.options.shape[1] for part in parts)
    padded = [_padded(part, width) for part in parts]
    return RuleReading(
        query=torch.cat([part.query for part in padded]),
        options=torch.cat([part.options for part in padded]),
        mask=torch.cat([part.mask for part in padded]),
        option_keys=tuple(keys for part in padded for keys in part.option_keys),
    )


def save_readings(directory: Path, keys: Sequence[str], reading: RuleReading, layer: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    tensors = {
        "query": reading.query.to(STORE_DTYPE),
        "options": reading.options.to(STORE_DTYPE),
        "mask": reading.mask,
    }
    save_file(tensors, str(directory / READINGS_FILE))
    meta = {"keys": list(keys), "option_keys": [list(item) for item in reading.option_keys], "layer": layer}
    (directory / KEYS_FILE).write_text(json.dumps(meta), encoding="utf-8")


def load_readings(directory: Path) -> tuple[dict[str, int], RuleReading]:
    tensors = load_file(str(directory / READINGS_FILE))
    meta = json.loads((directory / KEYS_FILE).read_text(encoding="utf-8"))
    reading = RuleReading(
        query=tensors["query"].float(),
        options=tensors["options"].float(),
        mask=tensors["mask"].bool(),
        option_keys=tuple(tuple(item) for item in meta["option_keys"]),
    )
    return {key: index for index, key in enumerate(meta["keys"])}, reading


def main() -> None:
    parser = argparse.ArgumentParser(description="Read every rule once with the frozen Qwen through Rein's encoder.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--bases", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--layer", type=int, required=True, help="0-based layer whose output reads the rule")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-rules", type=int, default=DEFAULT_BATCH_RULES)
    parser.add_argument("--tool-binary", type=Path, default=None, help="add yes/no versions of each tool rule")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    backbone = placed(truncate_backbone(load_text_backbone(args.model, dtype=dtype), args.layer + 1), device)
    by_id = with_tool_questions(load_all_records(args.data), load_tool_rules(args.tool_binary))
    bases = load_bases(args.bases)
    bases = {**derived_bases(by_id, bases), **bases}
    rules = rules_with_bases(by_id.values(), bases)
    reader = RuleReader(backbone, RequestEncoder(tokenizer))
    reading = read_all(reader, list(rules.values()), args.batch_rules)
    save_readings(args.out, list(rules), reading, args.layer)
    print(json.dumps({"rules": len(rules), "layer": args.layer, "width": int(reading.options.shape[1])}), flush=True)


if __name__ == "__main__":
    main()
