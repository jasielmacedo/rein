"""llama.cpp inside Modal containers: converting a checkpoint to GGUF, quantizing it, and serving it."""

import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from images import LLAMA_CPP_DIR

PORT = 8080
CONTEXT = 8192
READY_TIMEOUT_S = 900
READY_POLL_S = 3
ALL_LAYERS = "99"


def wait_ready(process: subprocess.Popen[bytes], port: int) -> None:
    deadline = time.monotonic() + READY_TIMEOUT_S
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited with {process.returncode}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=READY_POLL_S)
            return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            time.sleep(READY_POLL_S)
    raise TimeoutError("llama-server did not become ready")


@contextmanager
def serving(gguf: Path, port: int = PORT, context: int = CONTEXT, parallel: int = 1) -> Iterator[str]:
    """A llama-server on the GPU for the duration of the block; yields its URL. With `parallel` slots the context is
    split between them, so it grows with them."""
    server = subprocess.Popen(
        [
            f"{LLAMA_CPP_DIR}/build/bin/llama-server",
            "-m",
            str(gguf),
            "--port",
            str(port),
            "-ngl",
            ALL_LAYERS,
            "-c",
            str(context * parallel),
            "--parallel",
            str(parallel),
        ]
    )
    try:
        wait_ready(server, port)
        yield f"http://127.0.0.1:{port}"
    finally:
        server.terminate()
        server.wait()
