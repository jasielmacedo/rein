import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from rein.control import constants as c
from rein.control.plan import Directive

Check = Callable[[str], Directive | None]
Emit = Callable[[str], None]


@dataclass(frozen=True)
class WatchConfig:
    check_every_chars: int = c.WATCH_CHECK_EVERY_CHARS
    max_checks: int = c.WATCH_MAX_CHECKS
    max_interventions: int = c.WATCH_MAX_INTERVENTIONS
    hold_windows: int = c.WATCH_HOLD_WINDOWS


@dataclass
class WatchResult:
    """`text` is what survives: the whole reply, or the reply cut back to the last text the user already saw."""

    text: str
    fired: Directive | None = None
    checks: int = 0
    check_seconds: list[float] = field(default_factory=list)


def watch_stream(chunks: Iterable[str], check: Check, config: WatchConfig, emit: Emit) -> WatchResult:
    """Shows a window only once `hold_windows` later windows pass too, so a leak split across windows is cut whole."""
    result = WatchResult(text="")
    text = ""
    emitted = 0
    passed: list[int] = []

    def release(upto: int) -> None:
        nonlocal emitted
        if upto > emitted:
            emit(text[emitted:upto])
            emitted = upto

    def run_check() -> Directive | None:
        started = time.monotonic()
        verdict = check(text)
        result.check_seconds.append(time.monotonic() - started)
        result.checks += 1
        return verdict

    def checked() -> int:
        return passed[-1] if passed else 0

    def fire(verdict: Directive) -> WatchResult:
        result.text, result.fired = text[:emitted], verdict
        return result

    for chunk in chunks:
        text += chunk
        if result.checks >= config.max_checks:
            release(len(text))
            continue
        if len(text) - checked() < config.check_every_chars:
            continue
        verdict = run_check()
        if verdict is not None:
            return fire(verdict)
        passed.append(len(text))
        if len(passed) > config.hold_windows:
            release(passed[-1 - config.hold_windows])
    if len(text) > checked() and result.checks < config.max_checks:
        verdict = run_check()
        if verdict is not None:
            return fire(verdict)
    release(len(text))
    result.text = text
    return result
