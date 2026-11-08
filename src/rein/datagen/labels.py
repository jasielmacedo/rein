from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from rein.datagen.models import RuleDraft
from rein.schema import Question


@dataclass(frozen=True)
class Resolved:
    question: Question
    unanimous: bool
    votes: tuple[str, ...]


def needs_tiebreak(votes: Sequence[str]) -> bool:
    return len(set(votes)) > 1


def resolve(rule: RuleDraft, votes: Sequence[str]) -> Resolved:
    """Hard label when every vote agrees, otherwise a soft target from the vote frequencies."""
    base = rule.question()
    keys = base.option_keys()
    counts = Counter(votes)
    if len(counts) == 1:
        (winner,) = counts
        if base.type == "noul":
            label: bool | int | str = winner == "yes"
        elif base.type == "score":
            label = keys.index(winner)
        else:
            label = winner
        return Resolved(base.model_copy(update={"label": label}), True, tuple(votes))
    target = {key: counts[key] / len(votes) for key in keys if counts[key]}
    return Resolved(base.model_copy(update={"target": target}), False, tuple(votes))
