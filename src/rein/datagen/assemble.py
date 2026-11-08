import hashlib
from collections.abc import Mapping

from rein.datagen.labels import Resolved
from rein.datagen.models import ConversationDraft, ScenarioDraft
from rein.datagen.plan import CaseSpec, ScenarioSpec
from rein.datagen.prompts import PROMPT_VERSION
from rein.schema import Split, State, TrainingRecord

SPLIT_BUCKETS = 100
SPLIT_UPPER_BOUNDS: tuple[tuple[Split, int], ...] = (
    ("train", 80),
    ("development", 87),
    ("calibration", 93),
    ("test", SPLIT_BUCKETS),
)
UNSEEN_SPLITS = frozenset({"test", "gold"})
HOLDOUT_SPLIT: Split = "test"
SYNTHETIC_SOURCE = "synthetic"


def split_for(scenario_id: str) -> Split:
    """Whole scenarios (and so all their rules) land in one split, which keeps test rules unseen."""
    bucket = int(hashlib.sha256(scenario_id.encode("utf-8")).hexdigest(), 16) % SPLIT_BUCKETS
    for split, upper in SPLIT_UPPER_BOUNDS:
        if bucket < upper:
            return split
    raise AssertionError("split bounds must cover every bucket")


def build_record(
    scenario_spec: ScenarioSpec,
    scenario: ScenarioDraft,
    case_spec: CaseSpec,
    conversation: ConversationDraft,
    resolved: Mapping[str, Resolved],
    force_split: Split | None = None,
) -> TrainingRecord:
    split: Split = force_split or (HOLDOUT_SPLIT if scenario_spec.holdout else split_for(scenario_spec.scenario_id))
    target = next(rule for rule in scenario.rules if rule.id == case_spec.target_rule)
    meta = {
        "prompt_version": PROMPT_VERSION,
        "domain": scenario_spec.domain,
        "target_rule": target.id,
        "category": target.category,
        "case_form": case_spec.case_form,
        "second_rule": case_spec.second_rule,
        "rule_style": scenario_spec.rule_style,
        "holdout": scenario_spec.holdout,
        "draft_truncated": case_spec.truncate_draft and conversation.draft is not None,
        "construction_option": case_spec.target_option,
        "lang_rule": scenario_spec.rule_language,
        "lang_conv": case_spec.conversation_language,
        "decision_turn": len(conversation.conversation),
        "draft_kind": case_spec.draft_kind,
        "unanimous": {rule_id: item.unanimous for rule_id, item in resolved.items()},
        "unseen_rule": split in UNSEEN_SPLITS,
    }
    return TrainingRecord(
        id=case_spec.case_id,
        source=SYNTHETIC_SOURCE,
        scenario_id=scenario_spec.scenario_id,
        split=split,
        state=State(context=scenario.context, conversation=conversation.conversation, draft=conversation.draft),
        questions={rule_id: item.question for rule_id, item in resolved.items()},
        meta=meta,
    )
