import httpx
import respx

from rein.eval.slices import suite_slices
from rein.eval.systemone import SYSTEMONE_PATH, probabilities, score_remote, to_request
from rein.train.toy import toy_records

BASE_URL = "http://baseline.test"


def test_request_renders_state_and_keeps_question_shape() -> None:
    record = toy_records("test", 1)[0]
    request = to_request(record)
    assert "Conversation:" in request["state"]
    assert set(request["questions"]) == set(record.questions)
    assert "label" not in request["questions"]["photo"]


def test_probabilities_follow_option_keys() -> None:
    assert probabilities({"type": "noul", "noul": 0.8}, ("no", "yes")) == (0.19999999999999996, 0.8)
    assert probabilities({"type": "choice", "probabilities": {"b": 3, "a": 1}}, ("a", "b")) == (0.25, 0.75)


@respx.mock
async def test_score_remote_maps_answers_to_rows() -> None:
    respx.post(f"{BASE_URL}{SYSTEMONE_PATH}").mock(
        return_value=httpx.Response(
            200,
            json={
                "answers": {
                    "photo": {"type": "noul", "noul": 0.9},
                    "tool": {"type": "choice", "probabilities": {"send_photo": 0.7, "web_search": 0.2, "none": 0.1}},
                }
            },
        )
    )
    rows = await score_remote(BASE_URL, toy_records("test", 2))
    assert len(rows) == 4
    assert {row.question_id for row in rows} == {"photo", "tool"}


def test_suite_has_the_paper_slices() -> None:
    slices = suite_slices(["en", "pt-BR", "hi"])
    for name in ("unseen_rules", "watcher", "long_conversation", "words_same_intent_different", "lang:pt-BR->hi"):
        assert name in slices
