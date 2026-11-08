from collections.abc import Sequence

from rein.control import constants as c
from rein.control.sheet import RuleSheet


def compose_system(character: str | None, bases: Sequence[str]) -> str:
    """The generator's system prompt: who it is, then one line per rule base, never a trigger or an action."""
    lines = [f"{c.BASE_BULLET}{base}" for base in dict.fromkeys(base.strip() for base in bases if base.strip())]
    parts = [part for part in ((character or "").strip(), "\n".join(lines)) if part]
    return c.SECTION_BREAK.join(parts)


def sheet_bases(sheet: RuleSheet) -> list[str]:
    return [rule.base for rule in sheet.rules if rule.base]
