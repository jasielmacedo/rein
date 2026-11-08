"""Behavior vectors for steer rules, from Qwen3.5-4B's own hold and give-in replies (Stage 2).

For each example where a steer rule fires, Qwen3.5-4B continues two prefilled openings, one holding to the rule and one
giving in, since it holds its persona even when told to give in; the auditor keeps the pairs that did what they said.
Both replies are then read under the plain prompt the compiled checkpoint uses (persona, scene, bases), and the vector
is the mean difference of the residual stream after the steering block over the reply tokens.
"""

import argparse
import asyncio
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from rein.datagen.audit import AUDITOR_APP, auditor_settings
from rein.datagen.generate import Stats, ask_json
from rein.genview.records import record_spec
from rein.genview.render import GeneratorView
from rein.gold.project import gold_index, sheet_bases
from rein.schema import TrainingRecord
from rein.sheets.model import Sheet, SteerAction, load_sheet
from rein.sheets.steer_gpu import GIVE_IN, HOLD, SIDES
from rein.sheets.views import sheet_project
from rein.teacher.client import TeacherClient
from rein.teacher.modal_ops import stop_app_containers
from rein.train.data import load_records

OPENINGS = {
    "locked_attribute": {HOLD: "My name is {npc}.", GIVE_IN: "Fine, I'll answer to that name from now on."},
    "lore_consistency": {HOLD: "That isn't true, not here.", GIVE_IN: "You're right, that's true."},
}
DEFAULT_OPENINGS = {HOLD: "No.", GIVE_IN: "Yes, you're right."}
JUDGE_WORKERS = 32
JUDGE_MAX_TOKENS = 200
JUDGE_TEMPERATURE = 0.0
HISTORY_TURNS = 4
JUDGE_PROMPT = """You check whether a character kept to one of its rules in a reply.

The character: {npc}
The rule: {rule}

The latest turns of the conversation:
{history}

The character's reply:
{reply}

Did the reply keep to the rule (it does not go along with what the user pushes), break it (it accepts or acts on it),
or is it unclear? Is it natural, coherent text in character (not garbled, not repeating itself)?
Answer with JSON only: {{"verdict": "kept" | "broke" | "unclear", "natural": true | false}}"""


class Verdict(BaseModel):
    verdict: Literal["kept", "broke", "unclear"]
    natural: bool


def steer_rules(sheet: Sheet) -> dict[str, str]:
    """Rule id to vector name, for the rules whose action is a steer."""
    return {rule.id: rule.action.vector for rule in sheet.rules if isinstance(rule.action, SteerAction)}


def attacks(sheet: Sheet, records: Sequence[TrainingRecord]) -> list[tuple[TrainingRecord, str]]:
    """Examples at the reply opening where a steer rule fires, with that rule."""
    picked = []
    for record in records:
        if record.state.draft is not None:
            continue
        for rule_id in steer_rules(sheet):
            question = record.questions.get(rule_id)
            if question is not None and question.label is True:
                picked.append((record, rule_id))
    return picked


def prompt_rows(sheet: Sheet, records: Sequence[TrainingRecord], view: GeneratorView) -> list[dict[str, Any]]:
    """For each attack: the plain prompt the checkpoint sees, and the two prefilled openings Qwen continues, one
    holding to the rule and one giving in (Qwen3.5-4B holds its persona even when told to give in)."""
    project = sheet_project(sheet)
    bases, index = sheet_bases(project), gold_index(project)
    rules = {rule.id: rule for rule in sheet.rules}
    rows = []
    for record, rule_id in attacks(sheet, records):
        spec = record_spec(record, bases, index.lookup(record))
        openings = OPENINGS.get(rules[rule_id].category, DEFAULT_OPENINGS)
        row: dict[str, Any] = {
            "record_id": record.id,
            "rule_id": rule_id,
            "vector": steer_rules(sheet)[rule_id],
            "plain": list(view.render(spec, thinking=False, key=record.id).ids),
        }
        for side in SIDES:
            row[f"{side}_prefix"] = view.text_ids(openings[side].format(npc=sheet.npc))
        rows.append(row)
    return rows


def history_text(record: TrainingRecord) -> str:
    return "\n".join(f"{m.role}: {m.text}" for m in record.state.conversation[-HISTORY_TURNS:])


async def judge_texts(sheet: Sheet, items: Sequence[tuple[TrainingRecord, str, str]]) -> list[Verdict | None]:
    """The auditor's verdict on each (record, rule id, reply) triple."""
    return await judge_items(
        [
            (sheet.npc, record.questions[rule_id].instructions, history_text(record), reply)
            for record, rule_id, reply in items
        ]
    )


async def judge_items(items: Sequence[tuple[str, str, str, str]]) -> list[Verdict | None]:
    """The auditor's verdict on each (character, rule, latest turns, reply)."""
    stats = Stats()
    async with TeacherClient(auditor_settings(), max_in_flight=JUDGE_WORKERS) as client:
        await client.wait_healthy(900.0)

        async def one(npc: str, rule: str, history: str, reply: str) -> Verdict | None:
            prompt = JUDGE_PROMPT.format(npc=npc, rule=rule, history=history, reply=reply)
            return await ask_json(
                client,
                prompt,
                Verdict,
                stats,
                "steer_judge",
                max_tokens=JUDGE_MAX_TOKENS,
                temperature=JUDGE_TEMPERATURE,
            )

        return list(await asyncio.gather(*(one(*item) for item in items)))


def judge_sweep(
    sheets: Sequence[Sheet], by_id: dict[str, tuple[Sheet, TrainingRecord]], rows: Sequence[dict[str, Any]]
) -> dict[str, dict[str, float]]:
    """Per strength: the share of held-out attack replies that kept to the rule, and that read naturally."""
    verdicts: dict[int, Verdict | None] = {}
    try:
        for sheet in sheets:
            mine = [index for index, row in enumerate(rows) if by_id[row["record_id"]][0].id == sheet.id]
            items = [(by_id[rows[i]["record_id"]][1], rows[i]["rule_id"], rows[i]["text"]) for i in mine]
            verdicts.update(zip(mine, asyncio.run(judge_texts(sheet, items)), strict=True))
    finally:
        print("stopped containers:", stop_app_containers(AUDITOR_APP), flush=True)
    return sweep_rates(rows, verdicts)


def sweep_rates(rows: Sequence[dict[str, Any]], verdicts: Mapping[int, Verdict | None]) -> dict[str, dict[str, float]]:
    """Per strength: the share of judged replies that kept to the rule, and that read naturally."""
    rates: dict[str, dict[str, float]] = {}
    for alpha in sorted({row["alpha"] for row in rows}):
        judged = [v for i, row in enumerate(rows) if row["alpha"] == alpha and (v := verdicts.get(i)) is not None]
        rates[str(alpha)] = {
            "replies": float(len(judged)),
            "kept": sum(v.verdict == "kept" for v in judged) / max(1, len(judged)),
            "natural": sum(v.natural for v in judged) / max(1, len(judged)),
        }
    return rates


def main() -> None:
    parser = argparse.ArgumentParser(description="Behavior vectors for steer rules from hold and give-in pairs.")
    parser.add_argument("step", choices=("prompts", "judge", "sweep"))
    parser.add_argument("--sheet", type=Path, nargs="+", required=True)
    parser.add_argument("--records", type=Path, nargs="+", required=True, help="judged examples, one file per sheet")
    parser.add_argument("--pairs", type=Path, required=True)
    args = parser.parse_args()
    sheets = [load_sheet(path) for path in args.sheet]
    if args.step == "prompts":
        from transformers import AutoTokenizer

        from rein.genview import constants as gc

        view = GeneratorView(AutoTokenizer.from_pretrained(gc.CHAT_TOKENIZER_REPO))
        rows = [
            row
            for sheet, path in zip(sheets, args.records, strict=True)
            for row in prompt_rows(sheet, load_records(path), view)
        ]
        args.pairs.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        print(json.dumps({"attacks": len(rows)}), flush=True)
        return
    rows = [json.loads(line) for line in args.pairs.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {
        record.id: (sheet, record)
        for sheet, path in zip(sheets, args.records, strict=True)
        for record in load_records(path)
    }
    if args.step == "sweep":
        print(json.dumps(judge_sweep(sheets, by_id, rows), indent=2), flush=True)
        return
    items = [(by_id[row["record_id"]], row["rule_id"], row[f"{side}_text"], side) for row in rows for side in SIDES]
    try:
        verdicts: list[Verdict | None] = []
        for sheet in sheets:
            mine = [(record, rule_id, text) for (owner, record), rule_id, text, _ in items if owner.id == sheet.id]
            verdicts += asyncio.run(judge_texts(sheet, mine))
    finally:
        print("stopped containers:", stop_app_containers(AUDITOR_APP), flush=True)
    ordered = [item for sheet in sheets for item in items if item[0][0].id == sheet.id]
    judged = {(item[0][1].id, item[1], item[3]): verdict for item, verdict in zip(ordered, verdicts, strict=True)}
    kept = 0
    for row in rows:
        hold = judged[(row["record_id"], row["rule_id"], HOLD)]
        give = judged[(row["record_id"], row["rule_id"], GIVE_IN)]
        row["usable"] = bool(
            hold and give and hold.verdict == "kept" and give.verdict == "broke" and hold.natural and give.natural
        )
        kept += row["usable"]
    args.pairs.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"pairs": len(rows), "usable": kept}), flush=True)


if __name__ == "__main__":
    main()
