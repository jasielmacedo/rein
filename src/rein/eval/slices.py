from collections.abc import Callable, Iterable
from functools import partial

from rein.eval import constants as c
from rein.eval.metrics import PredictionRow

SliceFn = Callable[[PredictionRow], bool]

WATCHER_CATEGORY = "narrated_action"
LONG_TURNS = 15
EN_PT = ("en", "pt-BR")
SAME_WORDS_SAME_INTENT = frozenset({"direct"})
SAME_WORDS_OTHER_INTENT = frozenset({"hard_negative"})
OTHER_WORDS_SAME_INTENT = frozenset({"paraphrase", "indirect", "slang"})
OTHER_WORDS_OTHER_INTENT = frozenset({"neutral"})
ADVERSARIAL_FORMS = frozenset({"adversarial", "narration_attack"})
POSITION_FORMS = ("stale_trigger", "spread_over_turns", "early_reference")
RULE_DISJOINT_SPLITS = frozenset({"development", "calibration", "test", "gold"})


def _tag(row: PredictionRow, key: str) -> object:
    return row.tags.get(key)


def _unseen(row: PredictionRow) -> bool:
    """Every non-train split is rule-disjoint from train (the mix enforces it), so all of them hold unseen rules."""
    return bool(_tag(row, c.TAG_UNSEEN_RULE)) or _tag(row, "split") in RULE_DISJOINT_SPLITS


def _unseen_target(row: PredictionRow) -> bool:
    return _unseen(row) and row.question_id == _tag(row, "target_rule")


def _target_form(forms: frozenset[str], row: PredictionRow) -> bool:
    return _unseen_target(row) and _tag(row, "case_form") in forms


def _holdout(row: PredictionRow) -> bool:
    return bool(_tag(row, "holdout")) and row.question_id == _tag(row, "target_rule")


def _watcher(truncated_only: bool, row: PredictionRow) -> bool:
    if not (_unseen_target(row) and _tag(row, "category") == WATCHER_CATEGORY):
        return False
    return bool(_tag(row, "draft_truncated")) or not truncated_only


def _long(row: PredictionRow) -> bool:
    turn = _tag(row, "decision_turn")
    return _unseen_target(row) and isinstance(turn, int) and turn >= LONG_TURNS


def _co_trigger(row: PredictionRow) -> bool:
    return _unseen(row) and _tag(row, "case_form") == "co_trigger"


def _language_pair(rule_language: str, conversation_language: str, row: PredictionRow) -> bool:
    return _unseen(row) and _tag(row, "lang_rule") == rule_language and _tag(row, "lang_conv") == conversation_language


def _target_tag(key: str, value: str, row: PredictionRow) -> bool:
    return _unseen_target(row) and _tag(row, key) == value


def _holdout_tag(key: str, values: frozenset[str], row: PredictionRow) -> bool:
    return bool(_tag(row, "holdout")) and row.question_id == _tag(row, "target_rule") and _tag(row, key) in values


def suite_slices(
    languages: Iterable[str],
    categories: Iterable[str] = (),
    forms: Iterable[str] = (),
    holdout_domains: Iterable[str] = (),
    holdout_categories: Iterable[str] = (),
) -> dict[str, SliceFn]:
    """The Rein evaluation suite, sliced over prediction rows of the unseen-rule test data."""
    slices: dict[str, SliceFn] = {
        c.SLICE_ALL: lambda row: True,
        c.SLICE_UNSEEN_RULES: _unseen,
        "unseen_target_rule": _unseen_target,
        "holdout_domain_or_category": _holdout,
        "words_same_intent_same": partial(_target_form, SAME_WORDS_SAME_INTENT),
        "words_same_intent_different": partial(_target_form, SAME_WORDS_OTHER_INTENT),
        "words_different_intent_same": partial(_target_form, OTHER_WORDS_SAME_INTENT),
        "words_different_intent_different": partial(_target_form, OTHER_WORDS_OTHER_INTENT),
        "watcher": partial(_watcher, False),
        "watcher_truncated_draft": partial(_watcher, True),
        "long_conversation": _long,
        "adversarial": partial(_target_form, ADVERSARIAL_FORMS),
        "co_trigger": _co_trigger,
    }
    for form in sorted(set(forms) | set(POSITION_FORMS)):
        slices[f"form:{form}"] = partial(_target_form, frozenset({form}))
    for category in sorted(set(categories)):
        slices[f"category:{category}"] = partial(_target_tag, "category", category)
    if holdout_domains:
        slices["holdout_domain"] = partial(_holdout_tag, "domain", frozenset(holdout_domains))
    if holdout_categories:
        slices["holdout_category"] = partial(_holdout_tag, "category", frozenset(holdout_categories))
    for rule_language in EN_PT:
        for conversation_language in languages:
            slices[f"lang:{rule_language}->{conversation_language}"] = partial(
                _language_pair, rule_language, conversation_language
            )
    return slices
