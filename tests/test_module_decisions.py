import pytest

from rein.module.decisions import LONG, SHORT, chances, fitted_temperature, play_scores, scores_by_set, separation

TRUTHS = {"a|nothink": (0, True), "b|nothink": (1, False), "c|nothink|long": (0, False)}
MODULE = {"a|nothink": (0.9, 0.1), "b|nothink": (0.2, 0.7), "c|nothink|long": (0.4, 0.0)}
TEACHER = {"a|nothink": (0.8, 0.1), "b|nothink": (0.1, 0.2), "c|nothink|long": (0.6, 0.0)}


def test_both_models_are_scored_on_the_rule_in_play_and_compared() -> None:
    short = {view_id: truth for view_id, truth in TRUTHS.items() if not view_id.endswith(LONG)}
    scores = play_scores(MODULE, TEACHER, short)
    assert scores["n"] == 2
    assert scores["module"] == pytest.approx(0.5)
    assert scores["rein_r1"] == pytest.approx(1.0)
    assert scores["agreement"] == pytest.approx(0.5)
    assert scores["mean_gap"] == pytest.approx(((0.1 + 0.0) / 2 + (0.1 + 0.5) / 2) / 2)


def test_views_the_teacher_did_not_read_are_left_out() -> None:
    assert play_scores(MODULE, {}, TRUTHS) == {"n": 0}


def test_long_conversations_are_scored_apart() -> None:
    scores = scores_by_set(MODULE, TEACHER, TRUTHS)
    assert scores[SHORT]["n"] == 2
    assert scores[LONG]["n"] == 1
    assert scores[LONG]["module"] == pytest.approx(1.0)
    assert scores[LONG]["rein_r1"] == pytest.approx(0.0)


def test_the_temperature_softens_an_overconfident_read() -> None:
    odds = {"a": (4.0,), "b": (-4.0,), "c": (4.0,), "d": (-4.0,)}
    labels = {"a": {0: 1}, "b": {0: 0}, "c": {0: 0}, "d": {0: 1}}
    assert fitted_temperature(odds, labels) > 1.0


def test_the_temperature_sharpens_an_underconfident_read() -> None:
    odds = {"a": (0.5,), "b": (-0.5,)}
    labels = {"a": {0: 1}, "b": {0: 0}}
    assert fitted_temperature(odds, labels) < 1.0
    assert chances(odds, fitted_temperature(odds, labels))["a"][0] > chances(odds, 1.0)["a"][0]


def test_separation_is_the_chance_a_firing_view_ranks_above_a_quiet_one() -> None:
    fire = {"a": (0.9,), "b": (0.4,), "c": (0.2,)}
    truths = {"a": (0, True), "b": (0, False), "c": (0, True)}
    assert separation(fire, truths, ["a", "b", "c"]) == pytest.approx(0.5)
    assert separation(fire, {"a": (0, True)}, ["a"]) is None
