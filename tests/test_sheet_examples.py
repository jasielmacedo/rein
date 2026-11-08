import asyncio
from pathlib import Path

import pytest

from rein.datagen.plan import CaseSpec
from rein.datagen.taxonomy import load_taxonomy
from rein.sheets.examples import ReplayModel, case_prompts
from rein.sheets.model import load_sheet
from rein.sheets.qwen_writer import written_case
from rein.teacher.client import TeacherError

PER_RULE = 3


def test_each_rule_gets_its_own_cases_and_qwen_answers_replay_by_prompt() -> None:
    sheet = load_sheet(Path("configs/sheets/maren.yaml"))
    rows = case_prompts(load_taxonomy(), sheet, PER_RULE)
    assert len(rows) == PER_RULE * len(sheet.rules)
    assert {row["spec"]["target_rule"] for row in rows} == {rule.id for rule in sheet.rules}
    writer = ReplayModel({rows[0]["prompt"]: '{"conversation": []}'})
    completion = asyncio.run(writer.chat([{"role": "user", "content": rows[0]["prompt"]}]))
    assert completion.content == '{"conversation": []}'
    with pytest.raises(TeacherError):
        asyncio.run(writer.chat([{"role": "user", "content": rows[1]["prompt"]}]))
    assert not written_case("no json here", CaseSpec(**rows[0]["spec"]))
