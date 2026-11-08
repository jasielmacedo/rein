from collections.abc import Iterable

from rein.control import constants as c


def base_problems(base: str, forbidden_terms: Iterable[str] = (), action_texts: Iterable[str] = ()) -> list[str]:
    """Why a base would put the rule's logic in the prompt; empty when it only says that the rule exists."""
    text = base.strip()
    problems: list[str] = []
    if not text:
        problems.append("the base is empty")
    if len(text) > c.MAX_BASE_CHARS:
        problems.append(f"the base is longer than {c.MAX_BASE_CHARS} characters")
    if "\n" in text:
        problems.append("the base spans more than one line")
    folded = text.casefold()
    problems.extend(
        f"the base names the forbidden term {term!r}" for term in forbidden_terms if term.casefold() in folded
    )
    if any(action.strip() and action.strip().casefold() in folded for action in action_texts):
        problems.append("the base repeats an action text")
    return problems
