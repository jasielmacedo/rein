import json
from pathlib import Path

import pytest

from rein.neurons.runtimes import Generation, Prompt, Run, compare, forced_rate, load_prompts, load_run

SIGNAL = "<|fim_pad|>"
SIGNAL_ID = 248063


def test_compare_counts_identical_replies_and_shared_prefixes() -> None:
    reference = [Generation("a", (1, 2, 3), "x"), Generation("b", (4, 5, 6), "y")]
    other = [Generation("a", (1, 2, 3), "x"), Generation("b", (4, 9, 6), "z")]
    report = compare(reference, other)
    assert report == {"prompts": 2.0, "identical": 0.5, "first_token_agreement": 1.0, "mean_shared_prefix": 2.0}


def test_forced_rate_reads_the_first_id_and_the_visible_text() -> None:
    runs = [
        Generation("a", (SIGNAL_ID, 7), f"{SIGNAL}hello"),
        Generation("b", (SIGNAL_ID,), ""),
        Generation("c", (3, 4), "hello"),
    ]
    rate = forced_rate(runs, SIGNAL_ID, SIGNAL)
    assert rate["first_id"] == pytest.approx(2 / 3)
    assert rate["visible_text"] == pytest.approx(1 / 3)


def test_runs_and_prompts_round_trip_through_files(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps({"prompt_id": "p|think", "ids": [1, 2, 3]}) + "\n", encoding="utf-8")
    assert load_prompts(prompts) == [Prompt("p|think", (1, 2, 3))]
    saved = tmp_path / "run.json"
    payload = {"seconds": 2.0, "generations": [{"prompt_id": "p|think", "ids": [5, 6, 7, 8], "text": "ok"}]}
    saved.write_text(json.dumps(payload), encoding="utf-8")
    run = load_run(saved)
    assert run == Run([Generation("p|think", (5, 6, 7, 8), "ok")], 2.0)
    assert run.tokens_per_s == pytest.approx(2.0)
