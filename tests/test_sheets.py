from pathlib import Path

import pytest
from pydantic import ValidationError

from rein.gold.project import GoldProject, build_records
from rein.sheets.model import Sheet, ToolAction, as_gold_sheet, as_scenario, load_sheet

SHEETS = sorted(Path("configs/sheets").glob("*.yaml"))


@pytest.mark.parametrize("path", SHEETS, ids=lambda path: path.stem)
def test_each_end_to_end_sheet_works_as_a_scenario_and_as_a_gold_sheet(path: Path) -> None:
    sheet = load_sheet(path)
    scenario = as_scenario(sheet)
    assert [rule.id for rule in scenario.rules] == [rule.id for rule in sheet.rules]
    assert any(isinstance(rule.action, ToolAction) for rule in sheet.rules)
    conversation = {"id": "script", "sheet": sheet.id, "messages": [m.model_dump(by_alias=True) for m in sheet.script]}
    project = GoldProject.model_validate(
        {"format": "rein-gold-project", "version": 1, "sheets": [as_gold_sheet(sheet)], "conversations": [conversation]}
    )
    built = build_records(project)
    assert not built.problems and len(built.records) == len(sheet.script)
    assert {tool["function"]["name"] for tool in sheet.tool_schemas()} == set(scenario.tools)


def test_a_tool_action_needs_options_that_are_the_sheets_tools() -> None:
    raw = load_sheet(SHEETS[0]).model_dump(by_alias=True)
    raw["tools"] = []
    with pytest.raises(ValidationError):
        Sheet.model_validate(raw)
