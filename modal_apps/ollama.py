"""Ollama inside a Modal container: one GGUF imported as a model that passes prompts through untouched, served for the
duration of a block. The end-to-end sessions send the very prompt text the other runtimes get (`raw` generation)."""

import os
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from rein.neurons.runtimes import OLLAMA_MODEL

PORT = 11434
READY_TIMEOUT_S = 300
READY_POLL_S = 2
MODELFILE = Path("/tmp/Modelfile")
PASS_THROUGH = '"""{{ .Prompt }}"""'


def _wait_ready(server: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + READY_TIMEOUT_S
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"ollama serve exited with {server.returncode}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/version", timeout=READY_POLL_S)
            return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            time.sleep(READY_POLL_S)
    raise TimeoutError("ollama did not become ready")


@contextmanager
def ollama_serving(gguf: Path, context: int, parallel: int) -> Iterator[str]:
    """`ollama serve` with `parallel` slots of `context` tokens, the GGUF imported as OLLAMA_MODEL; yields the URL."""
    env = {
        **os.environ,
        "OLLAMA_HOST": f"127.0.0.1:{PORT}",
        "OLLAMA_NUM_PARALLEL": str(parallel),
        "OLLAMA_CONTEXT_LENGTH": str(context),
        "OLLAMA_KEEP_ALIVE": "-1",
    }
    server = subprocess.Popen(["ollama", "serve"], env=env)
    try:
        _wait_ready(server)
        subprocess.run(["ollama", "--version"], env=env, check=True)
        MODELFILE.write_text(f"FROM {gguf}\nTEMPLATE {PASS_THROUGH}\nPARAMETER num_ctx {context}\n", encoding="utf-8")
        subprocess.run(["ollama", "create", OLLAMA_MODEL, "-f", str(MODELFILE)], env=env, check=True)
        yield f"http://127.0.0.1:{PORT}"
    finally:
        server.terminate()
        server.wait()
