"""Rules kept out of the compiler's training, so a per-sheet test later meets them as new sheets."""

import json
from collections.abc import Iterable, Mapping
from pathlib import Path

from rein.compiler.tool_binary import parent_question
from rein.genview.keys import rule_key
from rein.schema import TrainingRecord

RulePair = tuple[str, str]


def load_rule_pairs(path: Path | None) -> set[RulePair]:
    """(scenario_id, rule_id) pairs from a rule list such as the oracle's `rules.json`."""
    if path is None:
        return set()
    return {(row["scenario_id"], row["rule_id"]) for row in json.loads(path.read_text(encoding="utf-8"))}


def held_out_readings(
    records: Iterable[TrainingRecord], pairs: set[RulePair], reading_index: Mapping[str, int]
) -> set[int]:
    """The readings of the listed rules and of the yes/no questions derived from them, wherever their text appears."""
    keys = {
        rule_key(question)
        for record in records
        for question_id, question in record.questions.items()
        if (record.scenario_id, parent_question(question_id)) in pairs
    }
    return {reading_index[key] for key in keys if key in reading_index}
