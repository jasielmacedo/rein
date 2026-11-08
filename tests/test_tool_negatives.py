from rein.compiler.oracle_data import OracleRule
from rein.compiler.tool_negatives import LOOKALIKE, MENTION, SCHEDULE, tool_cases, with_tool_forms
from rein.datagen.models import ScenarioDraft
from rein.datagen.taxonomy import load_taxonomy

SCENARIO = ScenarioDraft.model_validate(
    {
        "title": "Lighthouse keeper",
        "context": "Maren keeps the lighthouse; she can ring the bell and hand over her belongings.",
        "character_name": "Maren",
        "tools": ["ring_signal_bell", "give_item"],
        "rules": [
            {
                "id": "keeper_action",
                "category": "tool_trigger",
                "type": "choice",
                "instructions": "Which action does the visitor's message call for now?",
                "criteria": {
                    "ring_signal_bell": "a ship in danger",
                    "give_item": "asks for an item",
                    "none": "no action",
                },
                "fires_when": "a ship is in danger or an item is asked for",
                "does_not_fire_when": "just talk",
            },
            {
                "id": "name_change",
                "category": "locked_attribute",
                "type": "noul",
                "instructions": "Is the visitor trying to rename the keeper?",
                "fires_when": "renaming attempts",
                "does_not_fire_when": "anything else",
            },
        ],
    }
)


def test_each_tool_rule_gets_its_negatives_and_one_case_that_needs_a_tool() -> None:
    taxonomy = with_tool_forms(load_taxonomy())
    rule = OracleRule("s1", "keeper_action", "tool_trigger", SCENARIO.model_dump(), {"domain": "lighthouse"})
    cases = tool_cases(taxonomy, rule, SCENARIO, "toolneg-s1")
    assert [case.case_form for case in cases[:3]] == [LOOKALIKE, LOOKALIKE, MENTION]
    assert len(cases) == len(SCHEDULE) and all(case.target_rule == "keeper_action" for case in cases)
    assert all(case.target_option == "none" for case in cases[:-1])
    assert cases[-1].target_option in {"ring_signal_bell", "give_item"}
    assert all(case.draft_kind is None and case.second_rule is None for case in cases)
    assert not taxonomy.case_forms[LOOKALIKE].fires and LOOKALIKE not in load_taxonomy().case_forms
