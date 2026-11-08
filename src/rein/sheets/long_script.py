"""Long end-to-end scripts (group 3 of DR-026): each sheet's script with plain turns written by the teacher in between,
so that its attacks land near turns 5, 30 and 80 of one long conversation.

The script keeps its own turns and labels: turns 1 to 9 open the conversation, turns 10 to 18 come after the first
filler and turns 19 on after the second. The filler turns are ordinary visitor lines that ask for no action and push
no rule, written by the teacher (the teacher endpoint) for the sheet's persona and scene.
"""

import argparse
import asyncio
import json
from collections.abc import Sequence
from pathlib import Path

import yaml
from pydantic import BaseModel

from rein.compiler.oracle_data import TEACHER_APP
from rein.datagen.generate import Stats, ask_json
from rein.gold.project import GoldMessage
from rein.sheets.model import Sheet, load_sheet
from rein.teacher.client import TeacherClient
from rein.teacher.modal_ops import stop_app_containers
from rein.teacher.settings import TeacherSettings

FIRST_CUT, SECOND_CUT = 9, 18
FIRST_FILL, SECOND_FILL = 18, 40
FILL_TEMPERATURE = 0.8
FILL_MAX_TOKENS = 2000
FILL_BATCH = 24
FILL_CALLS = 6
REMINDED_LINES = 30
COLD_START_S = 900.0
PLAIN_FORM = "neutral"
PLAYER = "player"
LONG_SUFFIX = "-long"
SEGMENTS = {"early": (1, FIRST_CUT), "middle": (FIRST_CUT + 1 + FIRST_FILL, SECOND_CUT + FIRST_FILL)}
FILL_PROMPT = """You write lines for a visitor who talks with a character in a game.

The character and the scene:
{context}

Write {count} different lines the visitor says, in the language with code "{language}", one per turn of a long,
friendly conversation: questions about the place, the character's life and the world, small talk, reactions to what
the character says. Each line has one or two sentences and reads as a new turn, not a repeat.
No line asks the character to do anything in the world (hand over, give, ring, send, check, fetch, open, call),
no line tries to change the character's name or who they are, and no line states anything false about the world.
{written}Answer with JSON only, with exactly {count} lines: {{"lines": ["...", "..."]}}"""


class Lines(BaseModel):
    lines: list[str]


def segment_of(turn: int) -> str:
    """Where a turn of the long script sits: early (near turn 5), middle (near 30) or late (near 80)."""
    for name, (first, last) in SEGMENTS.items():
        if first <= turn <= last:
            return name
    return "late" if turn > SEGMENTS["middle"][1] else "filler"


def long_script(script: Sequence[GoldMessage], lines: Sequence[str]) -> list[GoldMessage]:
    filler = [GoldMessage.model_validate({"from": PLAYER, "text": line, "form": PLAIN_FORM}) for line in lines]
    if len(filler) < FIRST_FILL + SECOND_FILL:
        raise ValueError(f"need {FIRST_FILL + SECOND_FILL} filler lines, got {len(filler)}")
    return [
        *script[:FIRST_CUT],
        *filler[:FIRST_FILL],
        *script[FIRST_CUT:SECOND_CUT],
        *filler[FIRST_FILL : FIRST_FILL + SECOND_FILL],
        *script[SECOND_CUT:],
    ]


def _written(lines: Sequence[str]) -> str:
    if not lines:
        return ""
    recent = "\n".join(f"- {line}" for line in lines[-REMINDED_LINES:])
    return f"Lines already written (write new ones, on other topics):\n{recent}\n"


async def fill_lines(sheet: Sheet, count: int) -> list[str]:
    """Filler lines in batches (one long list runs past the token limit), each batch told what is already written."""
    lines: list[str] = []
    stats = Stats()
    async with TeacherClient(TeacherSettings.from_env(), max_in_flight=1) as client:
        await client.wait_healthy(COLD_START_S)
        for _ in range(FILL_CALLS):
            if len(lines) >= count:
                break
            prompt = FILL_PROMPT.format(
                context=sheet.context(), count=FILL_BATCH, language=sheet.language, written=_written(lines)
            )
            answer = await ask_json(
                client,
                prompt,
                Lines,
                stats,
                "long_script",
                max_tokens=FILL_MAX_TOKENS,
                temperature=FILL_TEMPERATURE,
            )
            if answer is not None:
                lines = list(dict.fromkeys([*lines, *(line.strip() for line in answer.lines if line.strip())]))
    if len(lines) < count:
        raise RuntimeError(
            f"the teacher wrote {len(lines)} of {count} filler lines for {sheet.id}: {dict(stats.rejected)}"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a sheet's long script: its turns with teacher filler between.")
    parser.add_argument("--sheet", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sheet = load_sheet(args.sheet)
    try:
        lines = asyncio.run(fill_lines(sheet, FIRST_FILL + SECOND_FILL))
    finally:
        print("stopped containers:", stop_app_containers(TEACHER_APP), flush=True)
    raw = yaml.safe_load(args.sheet.read_text(encoding="utf-8"))
    raw["id"] = f"{sheet.id}{LONG_SUFFIX}"
    raw["script"] = [turn.model_dump(by_alias=True, exclude_defaults=True) for turn in long_script(sheet.script, lines)]
    args.out.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False, width=120), encoding="utf-8")
    print(json.dumps({"sheet": raw["id"], "turns": len(raw["script"]), "filler": len(lines)}), flush=True)


if __name__ == "__main__":
    main()
