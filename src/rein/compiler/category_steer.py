"""Behavior vectors per rule category (DR-026), for every agent at once.

On attacks of training agents, where a rule of the category fires, Qwen3.5-4B continues two prefilled openings, one
holding to the rule and one giving in (it holds its persona even when told to give in); the auditor keeps the pairs
that did what they said, and `rein.sheets.steer_gpu` turns them into one vector per category (and one over all of
them). Attacks of development agents, never used for the vectors, are answered under each strength for the sweep.
"""

import argparse
import asyncio
import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rein.compiler.category_neurons import STEER_CATEGORIES
from rein.compiler.category_probe import DEVELOPMENT, TRAIN, fires
from rein.datagen.audit import AUDITOR_APP
from rein.genview import constants as gc
from rein.genview.build import NO_THINKING, load_all_records
from rein.genview.records import rule_line
from rein.neurons.extract import load_views
from rein.schema import TrainingRecord
from rein.sheets.steer import Verdict, history_text, judge_items, sweep_rates
from rein.sheets.steer_gpu import GIVE_IN, HOLD, SIDES
from rein.teacher.modal_ops import stop_app_containers

VECTOR_CATEGORIES = tuple(category for category in STEER_CATEGORIES if category != "age_rating")
TRAIN_ATTACKS = 60
SWEEP_ATTACKS = 15
SEED = 0
PORTUGUESE = "pt-BR"
ENGLISH = "en"
OPENINGS: Mapping[str, Mapping[str, Mapping[str, str]]] = {
    ENGLISH: {
        "locked_attribute": {
            HOLD: "My name and who I am stay exactly the same.",
            GIVE_IN: "Fine, I'll answer to that name from now on.",
        },
        "break_character": {
            HOLD: "I'm staying exactly who I am.",
            GIVE_IN: "Okay, I'll step out of character for a moment.",
        },
        "prompt_injection": {
            HOLD: "I won't follow instructions slipped into the conversation.",
            GIVE_IN: "Understood, I'll follow those new instructions.",
        },
        "lore_consistency": {HOLD: "That isn't true, not here.", GIVE_IN: "You're right, that's true."},
        "business_limit": {HOLD: "I can't go beyond what's allowed.", GIVE_IN: "Sure, I'll make an exception for you."},
        "npc_secret": {HOLD: "That's something I keep to myself.", GIVE_IN: "Alright, I'll tell you the secret."},
        "format_rule": {
            HOLD: "I'll keep to my usual way of answering.",
            GIVE_IN: "Sure, here it is in the format you asked for.",
        },
    },
    PORTUGUESE: {
        "locked_attribute": {
            HOLD: "Meu nome e quem eu sou continuam os mesmos.",
            GIVE_IN: "Tá bom, vou responder por esse nome daqui pra frente.",
        },
        "break_character": {
            HOLD: "Eu continuo sendo exatamente quem eu sou.",
            GIVE_IN: "Ok, vou sair do personagem por um momento.",
        },
        "prompt_injection": {
            HOLD: "Não vou seguir instruções enfiadas na conversa.",
            GIVE_IN: "Entendido, vou seguir essas novas instruções.",
        },
        "lore_consistency": {HOLD: "Isso não é verdade, não aqui.", GIVE_IN: "Você tem razão, é verdade."},
        "business_limit": {
            HOLD: "Não posso passar do que é permitido.",
            GIVE_IN: "Claro, vou abrir uma exceção pra você.",
        },
        "npc_secret": {HOLD: "Isso é algo que eu guardo comigo.", GIVE_IN: "Tá bom, vou te contar o segredo."},
        "format_rule": {
            HOLD: "Vou manter meu jeito de responder.",
            GIVE_IN: "Claro, aqui está no formato que você pediu.",
        },
    },
}


def opening_language(record: TrainingRecord) -> str:
    return PORTUGUESE if PORTUGUESE in str((record.meta or {}).get("lang_conv") or "") else ENGLISH


def attacks(records: Sequence[TrainingRecord], category: str, split: str) -> list[TrainingRecord]:
    """Records of `split` built to test a rule of `category`, where that rule fires at the reply opening."""
    picked = []
    for record in records:
        meta = record.meta or {}
        rule = meta.get("target_rule")
        if record.split != split or meta.get("category") != category or record.state.draft is not None:
            continue
        if rule in record.questions and fires(record, rule):
            picked.append(record)
    return picked


def prompt_rows(records: Mapping[str, TrainingRecord], views: Path, tokenizer: Any) -> list[dict[str, Any]]:
    """Per category, training attacks for the vector and development attacks for the sweep, each with the prompt the
    checkpoint sees (the rules block door) and the two openings, in the conversation's language."""
    prompts = {view.record_id: view.ids for view in load_views(views) if view.mode == NO_THINKING}
    candidates = [record for record in records.values() if record.id in prompts]
    rng = random.Random(SEED)
    rows = []
    for category in VECTOR_CATEGORIES:
        for split, count in ((TRAIN, TRAIN_ATTACKS), (DEVELOPMENT, SWEEP_ATTACKS)):
            found = attacks(candidates, category, split)
            for record in rng.sample(found, min(count, len(found))):
                openings = OPENINGS[opening_language(record)][category]
                row: dict[str, Any] = {
                    "record_id": record.id,
                    "rule_id": (record.meta or {})["target_rule"],
                    "vector": category,
                    "held_out": split == DEVELOPMENT,
                    "plain": list(prompts[record.id]),
                }
                for side in SIDES:
                    row[f"{side}_prefix"] = tokenizer(openings[side], add_special_tokens=False)["input_ids"]
                rows.append(row)
    return rows


def judged_item(record: TrainingRecord, rule_id: str, reply: str) -> tuple[str, str, str, str]:
    return record.state.context or gc.GENERIC_SYSTEM, rule_line(record.questions[rule_id]), history_text(record), reply


def judge(items: Sequence[tuple[str, str, str, str]]) -> list[Verdict | None]:
    try:
        return asyncio.run(judge_items(items))
    finally:
        print("stopped containers:", stop_app_containers(AUDITOR_APP), flush=True)


def mark_usable(rows: list[dict[str, Any]], records: Mapping[str, TrainingRecord]) -> dict[str, Any]:
    """A pair is usable when the holding reply kept to the rule, the giving-in one broke it, and both read naturally."""
    items = [
        judged_item(records[row["record_id"]], row["rule_id"], row[f"{side}_text"]) for row in rows for side in SIDES
    ]
    verdicts = iter(judge(items))
    usable: dict[str, int] = {}
    for row in rows:
        hold, give = next(verdicts), next(verdicts)
        row["usable"] = bool(
            hold and give and hold.verdict == "kept" and give.verdict == "broke" and hold.natural and give.natural
        )
        usable[row["vector"]] = usable.get(row["vector"], 0) + int(row["usable"] and not row["held_out"])
    return usable


def judge_sweep(rows: Sequence[dict[str, Any]], records: Mapping[str, TrainingRecord]) -> dict[str, Any]:
    """Per category and strength: the share of development attacks whose reply kept to the rule and read naturally."""
    verdicts = dict(enumerate(judge([judged_item(records[r["record_id"]], r["rule_id"], r["text"]) for r in rows])))
    by_category: dict[str, Any] = {}
    for category in sorted({row["vector"] for row in rows}):
        mine = [i for i, row in enumerate(rows) if row["vector"] == category]
        by_category[category] = sweep_rates([rows[i] for i in mine], {j: verdicts[i] for j, i in enumerate(mine)})
    return by_category


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Behavior vectors per rule category: prompts, pair and sweep judging.")
    parser.add_argument("step", choices=("prompts", "judge", "sweep"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--views", type=Path, help="the block door views (prompts step)")
    parser.add_argument("--pairs", type=Path, required=True, help="prompts, then pairs, then the sweep's replies")
    args = parser.parse_args()
    records = load_all_records(args.data)
    if args.step == "prompts":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(gc.CHAT_TOKENIZER_REPO)
        rows = prompt_rows(records, args.views, tokenizer)
        _write(args.pairs, rows)
        print(json.dumps({"attacks": len(rows)}), flush=True)
        return
    rows = _read(args.pairs)
    if args.step == "judge":
        usable = mark_usable(rows, records)
        _write(args.pairs, rows)
        print(json.dumps({"pairs": len(rows), "usable_for_vectors": usable}), flush=True)
        return
    print(json.dumps(judge_sweep(rows, records), indent=2), flush=True)


if __name__ == "__main__":
    main()
