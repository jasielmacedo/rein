import json
import subprocess
import time

import modal

APP_NAME = "rein-auditor"
MODEL_ID = "Qwen/Qwen3.8-27B-FP8"
VLLM_IMAGE = "vllm/vllm-openai:v0.27.1"
GPU = "H100"
MAX_CONTEXT = 32768
VLLM_PORT = 8000
MINUTES = 60
STARTUP_TIMEOUT_S = 20 * MINUTES
REQUEST_TIMEOUT_S = 30 * MINUTES
SCALEDOWN_S = 10 * MINUTES
MAX_CONCURRENT_INPUTS = 64
GPU_MEMORY_UTILIZATION = 0.92
HEALTH_POLL_S = 2
CACHE_MOUNT = "/root/.cache/huggingface"
HF_SECRET_NAME = "huggingface"

app = modal.App(APP_NAME)
cache_volume = modal.Volume.from_name("rein-auditor-cache", create_if_missing=True)

image = (
    modal.Image.from_registry(VLLM_IMAGE)
    .entrypoint([])
    .run_commands("ln -sf $(which python3) /usr/bin/python")
    .env(
        {
            "HF_HUB_CACHE": CACHE_MOUNT,
            "HF_XET_HIGH_PERFORMANCE": "1",
            "VLLM_CACHE_ROOT": f"{CACHE_MOUNT}/vllm-root",
            "TORCH_CPP_LOG_LEVEL": "FATAL",
        }
    )
)

with image.imports():
    import requests
    from huggingface_hub import snapshot_download


def _wait_ready(process: subprocess.Popen) -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vllm exited with code {process.returncode}")
        try:
            requests.get(f"http://127.0.0.1:{VLLM_PORT}/health", timeout=5).raise_for_status()
            return
        except (requests.exceptions.ConnectionError, requests.exceptions.HTTPError, requests.exceptions.Timeout):
            time.sleep(HEALTH_POLL_S)
    raise TimeoutError(f"vllm not ready within {STARTUP_TIMEOUT_S}s")


@app.cls(
    image=image,
    gpu=GPU,
    scaledown_window=SCALEDOWN_S,
    timeout=REQUEST_TIMEOUT_S,
    volumes={CACHE_MOUNT: cache_volume},
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
)
@modal.concurrent(max_inputs=MAX_CONCURRENT_INPUTS)
class Auditor:
    """Qwen3.8-27B behind an OpenAI-compatible endpoint, used only to audit and correct labels."""

    @modal.enter()
    def startup(self) -> None:
        model_path = snapshot_download(MODEL_ID)
        cache_volume.commit()
        command = [
            "vllm",
            "serve",
            model_path,
            "--served-model-name",
            MODEL_ID,
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT),
            "--max-model-len",
            str(MAX_CONTEXT),
            "--gpu-memory-utilization",
            str(GPU_MEMORY_UTILIZATION),
            "--reasoning-parser",
            "qwen3",
            "--language-model-only",
            "--enable-prefix-caching",
            "--max-num-seqs",
            str(MAX_CONCURRENT_INPUTS),
        ]
        print(json.dumps(command))
        self.process = subprocess.Popen(command)
        _wait_ready(self.process)

    @modal.exit()
    def stop(self) -> None:
        self.process.terminate()

    @modal.web_server(port=VLLM_PORT, requires_proxy_auth=True, startup_timeout=STARTUP_TIMEOUT_S)
    def serve(self) -> None:
        pass
