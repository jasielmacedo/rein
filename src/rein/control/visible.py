from rein.control.watcher import Emit

TOOL_CALL_MARKER = "<tool_call>"


class VisibleText:
    """Forwards reply text to the user and stops at the first tool-call marker, which belongs to the application."""

    def __init__(self, emit: Emit, marker: str = TOOL_CALL_MARKER) -> None:
        self._emit = emit
        self._marker = marker
        self._raw = ""
        self._sent = 0
        self._closed = False

    def __call__(self, text: str) -> None:
        if self._closed or not text:
            return
        self._raw += text
        cut = self._raw.find(self._marker, self._sent)
        if cut >= 0:
            self._send(cut)
            self._closed = True
            return
        self._send(max(self._sent, len(self._raw) - len(self._marker) + 1))

    def flush(self) -> None:
        if not self._closed:
            self._send(len(self._raw))

    def _send(self, end: int) -> None:
        if end > self._sent:
            self._emit(self._raw[self._sent : end])
            self._sent = end

    @property
    def text(self) -> str:
        return self._raw[: self._sent]
