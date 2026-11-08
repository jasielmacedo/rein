from pathlib import Path

from rein.schema import Question
from rein.teacher.store import JsonlStore, content_key

BASE_KEY_TAG = "base"


def rule_key(question: Question) -> str:
    """Bases depend on the rule's text and options only, never on its label."""
    return content_key(BASE_KEY_TAG, question.type, question.instructions, question.option_texts())


def load_bases(path: Path) -> dict[str, str]:
    return {row["key"]: row["base"] for row in JsonlStore(path).rows() if isinstance(row.get("base"), str)}
