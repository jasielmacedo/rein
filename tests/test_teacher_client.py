import json

import httpx
import pytest
import respx

from rein.teacher import client as client_module
from rein.teacher import constants as c
from rein.teacher.client import TeacherClient, TeacherError, TeacherUnavailable, parse_json_object
from rein.teacher.settings import TeacherSettings

BASE_URL = "https://teacher.test"
CHAT_URL = f"{BASE_URL}{c.CHAT_PATH}"
HEALTH_URL = f"{BASE_URL}{c.HEALTH_PATH}"
MESSAGES = [{"role": "user", "content": "hi"}]


def completion_body(content: str = '{"ok": true}') -> dict[str, object]:
    return {
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7},
    }


@pytest.fixture
def settings() -> TeacherSettings:
    return TeacherSettings(base_url=BASE_URL, proxy_key="key-value", proxy_secret="secret-value")


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(client_module, "backoff_delay", lambda attempt: 0.0)


@respx.mock
async def test_chat_sends_headers_and_disables_thinking(settings: TeacherSettings) -> None:
    route = respx.post(CHAT_URL).mock(return_value=httpx.Response(200, json=completion_body()))
    async with TeacherClient(settings) as teacher:
        result = await teacher.chat(MESSAGES, max_tokens=16)
    request = route.calls.last.request
    payload = json.loads(request.content)
    assert request.headers[c.PROXY_KEY_HEADER] == "key-value"
    assert request.headers[c.PROXY_SECRET_HEADER] == "secret-value"
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["response_format"] == {"type": "json_object"}
    assert result.content == '{"ok": true}'
    assert (result.prompt_tokens, result.completion_tokens) == (5, 7)


@respx.mock
async def test_chat_retries_retryable_status(settings: TeacherSettings) -> None:
    route = respx.post(CHAT_URL).mock(
        side_effect=[httpx.Response(503), httpx.Response(429), httpx.Response(200, json=completion_body())]
    )
    async with TeacherClient(settings) as teacher:
        result = await teacher.chat(MESSAGES, max_tokens=16)
    assert route.call_count == 3
    assert result.finish_reason == "stop"


@respx.mock
async def test_chat_fails_fast_on_client_error(settings: TeacherSettings) -> None:
    route = respx.post(CHAT_URL).mock(return_value=httpx.Response(400, text="bad request"))
    async with TeacherClient(settings) as teacher:
        with pytest.raises(TeacherError, match="400"):
            await teacher.chat(MESSAGES, max_tokens=16)
    assert route.call_count == 1


@respx.mock
async def test_chat_gives_up_after_max_attempts(settings: TeacherSettings) -> None:
    route = respx.post(CHAT_URL).mock(return_value=httpx.Response(503))
    async with TeacherClient(settings) as teacher:
        with pytest.raises(TeacherError, match="giving up"):
            await teacher.chat(MESSAGES, max_tokens=16)
    assert route.call_count == c.MAX_ATTEMPTS


@respx.mock
async def test_chat_rejects_empty_content(settings: TeacherSettings) -> None:
    respx.post(CHAT_URL).mock(return_value=httpx.Response(200, json=completion_body("  ")))
    async with TeacherClient(settings) as teacher:
        with pytest.raises(TeacherError, match="empty"):
            await teacher.chat(MESSAGES, max_tokens=16)


@respx.mock
async def test_wait_healthy_polls_until_ok(settings: TeacherSettings) -> None:
    route = respx.get(HEALTH_URL).mock(side_effect=[httpx.Response(503), httpx.Response(200)])
    async with TeacherClient(settings) as teacher:
        elapsed = await teacher.wait_healthy(timeout_s=5.0, poll_s=0.0)
    assert route.call_count == 2
    assert elapsed >= 0.0


@respx.mock
async def test_wait_healthy_stops_on_auth_failure(settings: TeacherSettings) -> None:
    respx.get(HEALTH_URL).mock(return_value=httpx.Response(401))
    async with TeacherClient(settings) as teacher:
        with pytest.raises(TeacherUnavailable, match="credentials"):
            await teacher.wait_healthy(timeout_s=5.0, poll_s=0.0)


def test_parse_json_object_accepts_fences() -> None:
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}


@pytest.mark.parametrize("content", ["not json", "[1, 2]"])
def test_parse_json_object_rejects_non_objects(content: str) -> None:
    with pytest.raises(TeacherError):
        parse_json_object(content)


def test_settings_reject_plain_http() -> None:
    with pytest.raises(ValueError, match="https"):
        TeacherSettings(base_url="http://teacher.test")


def test_settings_require_key_and_secret_together() -> None:
    with pytest.raises(ValueError, match="both"):
        TeacherSettings(base_url=BASE_URL, proxy_key="only-key")


def test_settings_repr_hides_secrets(settings: TeacherSettings) -> None:
    assert "key-value" not in repr(settings)
    assert "secret-value" not in repr(settings)
