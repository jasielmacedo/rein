import random
from collections.abc import Mapping
from dataclasses import dataclass

from rein.datagen.models import RuleDraft, ScenarioDraft
from rein.datagen.taxonomy import Taxonomy

MIN_CATEGORIES = 4
MAX_CATEGORIES = 6
SPREAD_MIN_TURNS = 6
EARLY_REFERENCE_MIN_TURNS = 8
EXTRA_DRAFT_RATE = 0.1
TRUNCATE_DRAFT_RATE = 0.35
NEGATIVE_OPTION_KEYS = frozenset({"none", "general", "other", "nenhum", "nenhuma", "nenhuma_das_anteriores"})
WATCHER_NEGATIVE_DRAFTS = ("tool_call", "narration_after_call", "refusal", "dialogue")
NARRATED_ACTION = "narrated_action"


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    domain: str
    categories: tuple[str, ...]
    rule_language: str
    rule_style: str
    holdout: bool


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    scenario_id: str
    target_rule: str
    case_form: str
    target_option: str
    conversation_language: str
    turns: int
    draft_kind: str | None
    truncate_draft: bool
    second_rule: str | None = None
    second_option: str | None = None


def weighted(rng: random.Random, weights: Mapping[str, float]) -> str:
    keys = list(weights)
    return rng.choices(keys, weights=[weights[key] for key in keys], k=1)[0]


def plan_scenarios(
    taxonomy: Taxonomy, count: int, seed: int, prefix: str, required_categories: tuple[str, ...] = ()
) -> list[ScenarioSpec]:
    """Regular scenarios never touch the holdout domains or categories; holdout scenarios always do."""
    rng = random.Random(seed)
    holdout = taxonomy.holdout
    regular_categories = [name for name in taxonomy.categories if name not in holdout.categories]
    regular_domains = [name for name in taxonomy.domains if name not in holdout.domains]
    specs = []
    for index in range(count):
        is_holdout = bool(holdout.domains or holdout.categories) and rng.random() < holdout.scenario_fraction
        size = rng.randint(MIN_CATEGORIES, MAX_CATEGORIES)
        if is_holdout:
            forced = list(holdout.categories)
            pool = [name for name in taxonomy.categories if name not in forced]
            categories = forced + rng.sample(pool, size - len(forced))
            domain = rng.choice(holdout.domains or taxonomy.domains)
        else:
            pool = [name for name in regular_categories if name not in required_categories]
            categories = list(required_categories) + rng.sample(pool, size - len(required_categories))
            domain = rng.choice(regular_domains)
        rng.shuffle(categories)
        specs.append(
            ScenarioSpec(
                scenario_id=f"{prefix}-s{index:05d}",
                domain=domain,
                categories=tuple(categories),
                rule_language=weighted(rng, taxonomy.languages.rules),
                rule_style=rng.choice(taxonomy.rule_styles),
                holdout=is_holdout,
            )
        )
    return specs


def target_option(rule: RuleDraft, fires: bool, rng: random.Random) -> str:
    keys = rule.question().option_keys()
    if rule.type == "noul":
        return "yes" if fires else "no"
    if rule.type == "score":
        half = len(keys) // 2
        return rng.choice(keys[half + 1 :] if fires else keys[: max(1, half)])
    negatives = [key for key in keys if key.lower() in NEGATIVE_OPTION_KEYS]
    positives = [key for key in keys if key not in negatives] or list(keys)
    return rng.choice(positives if fires or not negatives else negatives)


def sample_turns(taxonomy: Taxonomy, form: str, rng: random.Random) -> int:
    band = taxonomy.conversation_lengths[weighted(rng, {k: v.weight for k, v in taxonomy.conversation_lengths.items()})]
    floor = {"spread_over_turns": SPREAD_MIN_TURNS, "early_reference": EARLY_REFERENCE_MIN_TURNS}.get(form, band.min)
    return rng.randint(max(band.min, floor), max(band.max, floor))


def plan_cases(
    taxonomy: Taxonomy,
    scenario_id: str,
    scenario: ScenarioDraft,
    per_scenario: int,
    seed: int,
    target_categories: tuple[str, ...] = (),
) -> list[CaseSpec]:
    rng = random.Random(f"{seed}:{scenario_id}")
    forms = list(taxonomy.case_forms)
    targets = [rule for rule in scenario.rules if rule.category in target_categories] or list(scenario.rules)
    cases = []
    for index in range(per_scenario):
        rule = targets[index % len(targets)]
        form = rng.choice(forms)
        spec_form = taxonomy.case_forms[form]
        if spec_form.two_rules and len(scenario.rules) < 2:
            form, spec_form = "direct", taxonomy.case_forms["direct"]
        second_rule = None
        second_option = None
        if spec_form.two_rules:
            partner = rng.choice([other for other in scenario.rules if other.id != rule.id])
            second_rule, second_option = partner.id, target_option(partner, True, rng)
        category = taxonomy.categories.get(rule.category)
        draft_kind: str | None = None
        if category is not None and category.needs_draft:
            draft_kind = NARRATED_ACTION if spec_form.fires else rng.choice(WATCHER_NEGATIVE_DRAFTS)
        elif rng.random() < EXTRA_DRAFT_RATE:
            draft_kind = "dialogue"
        cases.append(
            CaseSpec(
                case_id=f"{scenario_id}-c{index:03d}",
                scenario_id=scenario_id,
                target_rule=rule.id,
                case_form=form,
                target_option=target_option(rule, spec_form.fires, rng),
                conversation_language=weighted(rng, taxonomy.languages.conversation),
                turns=sample_turns(taxonomy, form, rng),
                draft_kind=draft_kind,
                truncate_draft=draft_kind is not None and rng.random() < TRUNCATE_DRAFT_RATE,
                second_rule=second_rule,
                second_option=second_option,
            )
        )
    return cases
