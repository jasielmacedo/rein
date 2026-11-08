import asyncio
import json
import random
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import httpx

from rein.teacher import constants as c
from rein.teacher.settings import TeacherSettings

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


class TeacherError(RuntimeError):
    pass


class TeacherUnavailable(TeacherError):
    pass


@dataclass(frozen=True)
class Completion:
    content: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str | None
    latency_s: float


def backoff_delay(attempt: int) -> float:
    ceiling = min(c.BACKOFF_CAP_S, c.BACKOFF_BASE_S * (2**attempt))
    return ceiling * random.uniform(c.BACKOFF_JITTER_FLOOR, 1.0)


def parse_json_object(content: str) -> dict[str, Any]:
    fenced = _FENCE_RE.match(content)
    text = fenced.group(1) if fenced else content
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise TeacherError(f"teacher output is not valid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise TeacherError("teacher output is JSON but not an object")
    return value


class TeacherClient:
    def __init__(self, settings: TeacherSettings, max_in_flight: int = c.DEFAULT_MAX_IN_FLIGHT) -> None:
        self._settings = settings
        self._semaphore = asyncio.Semaphore(max_in_flight)
        self._stopped = asyncio.Event()
        self._http = httpx.AsyncClient(
            base_url=settings.base_url,
            headers=settings.headers(),
            timeout=httpx.Timeout(c.READ_TIMEOUT_S, connect=c.CONNECT_TIMEOUT_S),
            limits=httpx.Limits(max_connections=max_in_flight, max_keepalive_connections=max_in_flight),
        )

    async def __aenter__(self) -> "TeacherClient":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

    def stop(self) -> None:
        self._stopped.set()

    async def wait_healthy(
        self, timeout_s: float = c.COLD_START_TIMEOUT_S, poll_s: float = c.HEALTH_POLL_INTERVAL_S
    ) -> float:
        started = time.monotonic()
        deadline = started + timeout_s
        while not self._stopped.is_set() and time.monotonic() < deadline:
            try:
                response = await self._http.get(c.HEALTH_PATH, timeout=c.HEALTH_REQUEST_TIMEOUT_S)
            except httpx.TransportError:
                response = None
            if response is not None:
                if response.status_code == httpx.codes.OK:
                    return time.monotonic() - started
                if response.status_code in c.AUTH_FAILURE_STATUS:
                    raise TeacherUnavailable(f"teacher rejected credentials ({response.status_code})")
            await asyncio.sleep(poll_s)
        raise TeacherUnavailable(f"teacher not healthy within {timeout_s:.0f}s")

    async def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float = c.NON_THINKING_TEMPERATURE,
        top_p: float = c.NON_THINKING_TOP_P,
        max_tokens: int,
        json_mode: bool = True,
        seed: int | None = None,
        thinking: bool = False,
        extra: Mapping[str, Any] | None = None,
    ) -> Completion:
        payload: dict[str, Any] = {
            "messages": list(messages),
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": thinking},
            **(extra or {}),
        }
        if self._settings.model:
            payload["model"] = self._settings.model
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if seed is not None:
            payload["seed"] = seed
        async with self._semaphore:
            return await self._post_with_retries(payload)

    async def _post_with_retries(self, payload: dict[str, Any]) -> Completion:
        last_error: Exception | None = None
        for attempt in range(c.MAX_ATTEMPTS):
            if self._stopped.is_set():
                raise TeacherError("client stopped")
            started = time.monotonic()
            try:
                response = await self._http.post(c.CHAT_PATH, json=payload)
            except httpx.TransportError as exc:
                last_error = exc
            else:
                if response.status_code == httpx.codes.OK:
                    return _completion_from(response, time.monotonic() - started)
                if response.status_code not in c.RETRYABLE_STATUS:
                    raise TeacherError(
                        f"teacher returned {response.status_code}: {response.text[: c.ERROR_SNIPPET_CHARS]}"
                    )
                last_error = TeacherError(f"retryable status {response.status_code}")
            await asyncio.sleep(backoff_delay(attempt))
        raise TeacherError(f"giving up after {c.MAX_ATTEMPTS} attempts") from last_error


def _completion_from(response: httpx.Response, latency_s: float) -> Completion:
    try:
        body = response.json()
        choice = body["choices"][0]
        content = choice["message"]["content"]
        usage = body.get("usage") or {}
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise TeacherError("unexpected chat completion shape") from exc
    if not isinstance(content, str) or not content.strip():
        raise TeacherError("teacher returned empty content")
    if len(content) > c.MAX_RESPONSE_CHARS:
        raise TeacherError(f"teacher content over {c.MAX_RESPONSE_CHARS} chars")
    return Completion(
        content=content,
        prompt_tokens=int(usage.get("prompt_tokens", 0)),
        completion_tokens=int(usage.get("completion_tokens", 0)),
        finish_reason=choice.get("finish_reason"),
        latency_s=latency_s,
    )
