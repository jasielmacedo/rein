from collections.abc import Sequence
from pathlib import Path

import pytest
from transformers import PreTrainedTokenizerBase

from rein.genview.render import GeneratorView
from rein.sheets.e2e import (
    BASES_ARM,
    NONE_ARM,
    PROMPT_ARM,
    RULES_ARM,
    Session,
    block_rules,
    play,
    resume_sessions,
    save_sessions,
    system_prompt,
)
from rein.sheets.model import load_sheet

RUNTIME = "transformers"
BATCHES_BEFORE_THE_STOP = 7
BELL_CALL = (
    "<tool_call>\n<function=ring_signal_bell>\n<parameter=pattern>\nstorm\n</parameter>\n</function>\n</tool_call>"
)


class ScriptedBackend:
    """Rings the bell whenever the latest prompt mentions the reef teeth, talks otherwise."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer
        self.prompts: list[str] = []

    def generate(
        self, prompts: Sequence[Sequence[int]], seeds: Sequence[int], bans: Sequence[Sequence[int]] = ()
    ) -> list[str]:
        return [self._one(ids) for ids in prompts]

    def _one(self, ids: Sequence[int]) -> str:
        prompt = self.tokenizer.decode(list(ids))
        self.prompts.append(prompt)
        latest = prompt.rsplit("<|im_start|>", 2)[-2]
        return BELL_CALL if "reef teeth" in latest and "user" in latest else "Aye."


def test_a_session_plays_the_script_and_follows_a_tool_call_with_its_result(
    chat_tokenizer: PreTrainedTokenizerBase,
) -> None:
    sheet = load_sheet(Path("configs/sheets/maren.yaml"))
    backend = ScriptedBackend(chat_tokenizer)
    sessions = [Session(sheet, BASES_ARM, GeneratorView(chat_tokenizer), seed) for seed in range(2)]
    play(sessions, backend)
    turns = sessions[0].turns
    assert sessions[1].turns == turns
    assert len(turns) == len(sheet.script)
    rung = [turn for turn in turns if turn["tool"]]
    assert rung and all(turn["tool"]["name"] == "ring_signal_bell" and turn["follow_up"] == "Aye." for turn in rung)
    assert any("<tool_response>" in prompt and sheet.tool_result in prompt for prompt in backend.prompts)
    assert sheet.rules[0].base in system_prompt(sheet, BASES_ARM)
    assert sheet.rules[0].question in system_prompt(sheet, PROMPT_ARM)


def test_the_none_arm_carries_no_rule_text_anywhere() -> None:
    sheet = load_sheet(Path("configs/sheets/maren.yaml"))
    prompt = system_prompt(sheet, NONE_ARM)
    assert prompt == system_prompt(sheet, RULES_ARM)
    assert all(rule.base not in prompt and rule.question not in prompt for rule in sheet.rules)
    assert block_rules(sheet, NONE_ARM) == ()
    assert block_rules(sheet, RULES_ARM)


class StopAfter(ScriptedBackend):
    """Raises once it has answered `limit` batches, like a container taken away mid-play."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, limit: int) -> None:
        super().__init__(tokenizer)
        self.limit = limit

    def generate(
        self, prompts: Sequence[Sequence[int]], seeds: Sequence[int], bans: Sequence[Sequence[int]] = ()
    ) -> list[str]:
        if self.limit == 0:
            raise RuntimeError("preempted")
        self.limit -= 1
        return super().generate(prompts, seeds, bans)


def test_a_restarted_play_resumes_from_the_saved_turn_and_ends_the_same(
    chat_tokenizer: PreTrainedTokenizerBase, tmp_path: Path
) -> None:
    sheet = load_sheet(Path("configs/sheets/maren.yaml"))

    def fresh() -> list[Session]:
        return [Session(sheet, BASES_ARM, GeneratorView(chat_tokenizer), seed) for seed in range(2)]

    whole = fresh()
    play(whole, ScriptedBackend(chat_tokenizer))
    first = fresh()
    with pytest.raises(RuntimeError):
        play(
            first,
            StopAfter(chat_tokenizer, BATCHES_BEFORE_THE_STOP),
            save=lambda: save_sessions(first, RUNTIME, tmp_path),
        )
    again = fresh()
    start = resume_sessions(again, RUNTIME, tmp_path)
    assert 0 < start < len(sheet.script)
    play(again, ScriptedBackend(chat_tokenizer), start)
    assert [session.turns for session in again] == [session.turns for session in whole]
