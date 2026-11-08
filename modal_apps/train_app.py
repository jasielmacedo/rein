import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import modal

APP_NAME = "rein-train"
GPU = "H100"
HOURS = 60 * 60
TRAIN_TIMEOUT_S = 8 * HOURS
TRAIN_RETRIES = 2
VOLUME_COMMIT_EVERY_S = 120
FINISHED_MARKER = "metrics.json"
SMOKE_TIMEOUT_S = 1 * HOURS
KERNEL_CHECK_TIMEOUT_S = 10 * 60
LATENCY_RECORDS = 200
EVAL_SPLIT_FILES = ("development.jsonl", "test.jsonl")
SWEEP_MEMORY_MB = 96 * 1024
KERNEL_CHECK = "from rein.model.kernels import require_fast_kernels; print(require_fast_kernels())"
DATA_MOUNT = "/data"
HF_MOUNT = "/hf-cache"
REPO_MOUNT = "/root/rein"
HF_SECRET_NAME = "huggingface"
PYTHON_VERSION = "3.12"
TORCH_VERSION = "2.13.0"
FLA_VERSION = "0.6.0"
CAUSAL_CONV1D_VERSION = "1.7.0"
LINGUA_PACKAGE = "lingua-language-detector>=2.0"
CUDA_DEVEL_IMAGE = "nvidia/cuda:13.0.0-devel-ubuntu22.04"
HOPPER_ARCH = "9.0"
CONV_BUILD_ENV = {
    "TORCH_CUDA_ARCH_LIST": HOPPER_ARCH,
    "CAUSAL_CONV1D_FORCE_BUILD": "TRUE",
    "MAX_JOBS": "8",
    "CC": "gcc",
    "CXX": "g++",
}
GATE_TESTS = (
    "tests/test_gpu_sync.py",
    "tests/test_acceptance.py",
    "tests/test_shared_state.py",
    "tests/test_batch_padding.py",
)
TRITON_CACHE = f"{DATA_MOUNT}/triton-cache"

EVAL_TIMEOUT_S = 4 * HOURS
SERVER_READY_TIMEOUT_S = 30 * 60
SERVER_POLL_S = 10
KEV_REPO = "https://github.com/jaredpalmer/kev.git"
KEV_DIR = "/opt/kev"
DEFAULT_BASELINE_CONCURRENCY = 8
BASELINES = {
    "kev-4b": {
        "port": 8009,
        "command": [
            f"{KEV_DIR}/.venv/bin/python",
            "-m",
            "kev.serve",
            "--run",
            "jaredpalmer/kev-4b",
            "--port",
            "8009",
            "--host",
            "127.0.0.1",
        ],
    },
    "kev-0.8b": {
        "port": 8010,
        "command": [
            f"{KEV_DIR}/.venv/bin/python",
            "-m",
            "kev.serve",
            "--run",
            "jaredpalmer/kev-0.8b",
            "--port",
            "8010",
            "--host",
            "127.0.0.1",
        ],
    },
    "tinyjev-4b": {"port": 8077, "concurrency": 1, "command": ["tinyjev", "serve", "TinyJev-4B", "--port", "8077"]},
    "tinyjev-0.6b": {
        "port": 8078,
        "concurrency": 1,
        "command": ["tinyjev", "serve", "TinyJev-0.6B", "--port", "8078"],
    },
}

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name("rein-data", create_if_missing=True)
hf_volume = modal.Volume.from_name("rein-hf-cache", create_if_missing=True)

image = (
    modal.Image.from_registry(CUDA_DEVEL_IMAGE, add_python=PYTHON_VERSION)
    .apt_install("build-essential")
    .pip_install(
        f"torch=={TORCH_VERSION}",
        "transformers>=5.17,<6",
        "peft>=0.17",
        "accelerate>=1.10",
        "safetensors>=0.5",
        "pydantic>=2.9",
        "httpx>=0.28",
        "pyyaml>=6.0",
        "python-dotenv>=1.0",
        "huggingface_hub>=0.34",
        f"flash-linear-attention=={FLA_VERSION}",
        "pytest>=8.3",
        "pytest-asyncio>=0.24",
        "respx>=0.21",
        "ninja>=1.11",
        "packaging>=24",
    )
    .env(CONV_BUILD_ENV)
    .run_commands(f"pip install --no-build-isolation causal-conv1d=={CAUSAL_CONV1D_VERSION}")
    .pip_install(LINGUA_PACKAGE)
    .env(
        {
            "HF_HOME": HF_MOUNT,
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONPATH": f"{REPO_MOUNT}/src",
            "PYTHONUNBUFFERED": "1",
            "TRITON_CACHE_DIR": TRITON_CACHE,
        }
    )
    .add_local_dir("src", f"{REPO_MOUNT}/src")
    .add_local_dir("tests", f"{REPO_MOUNT}/tests")
    .add_local_dir("configs", f"{REPO_MOUNT}/configs")
    .add_local_file("pyproject.toml", f"{REPO_MOUNT}/pyproject.toml")
)

VOLUMES = {DATA_MOUNT: data_volume, HF_MOUNT: hf_volume}
SECRETS = [modal.Secret.from_name(HF_SECRET_NAME)]


def _run(*args: str) -> None:
    subprocess.run(list(args), cwd=REPO_MOUNT, check=True)


def _run_committing(*args: str) -> None:
    """Runs a long job and commits the data volume periodically, so resume points survive a lost container."""
    process = subprocess.Popen(list(args), cwd=REPO_MOUNT)
    running = True
    while running:
        try:
            process.wait(timeout=VOLUME_COMMIT_EVERY_S)
            running = False
        except subprocess.TimeoutExpired:
            data_volume.commit()
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, list(args))


def _gate() -> None:
    _run("python", "-c", KERNEL_CHECK)
    _run("python", "-m", "pytest", "-q", *GATE_TESTS)


@app.function(image=image, gpu=GPU, timeout=KERNEL_CHECK_TIMEOUT_S)
def kernels() -> None:
    _run("python", "-c", "from rein.model.kernels import kernel_report; [print(s) for s in kernel_report()]")


@app.function(image=image, gpu=GPU, volumes=VOLUMES, secrets=SECRETS, timeout=SMOKE_TIMEOUT_S)
def smoke(base: str = "Qwen/Qwen3.5-0.8B-Base") -> None:
    _gate()
    _run("python", "-m", "rein.train.smoke", "--base", base, "--out", f"{DATA_MOUNT}/runs/smoke")
    data_volume.commit()
    hf_volume.commit()


@app.function(image=image, gpu=GPU, volumes=VOLUMES, secrets=SECRETS, timeout=TRAIN_TIMEOUT_S, retries=TRAIN_RETRIES)
def train(config: str, eval_data: list[str] | None = None, debug: bool = False) -> None:
    import os

    import yaml

    if debug:
        os.environ["TRITON_PRINT_AUTOTUNING"] = "1"
    _gate()
    settings = yaml.safe_load(Path(REPO_MOUNT, config).read_text(encoding="utf-8"))
    run_dir = f"{settings['output_dir']}/{settings['run_name']}"
    try:
        if Path(run_dir, FINISHED_MARKER).exists():
            print(f"{run_dir} already finished training, running the evaluation only")
        else:
            _run_committing("python", "-m", "rein.train.run", "--config", config)
            data_volume.commit()
        for split_file in eval_data or []:
            split = Path(split_file).stem
            _run(
                "python",
                "-m",
                "rein.eval.run_eval",
                "--run",
                run_dir,
                "--data",
                split_file,
                "--out",
                f"{run_dir}/eval-{split}",
                "--latency-records",
                str(LATENCY_RECORDS),
            )
    finally:
        data_volume.commit()
        hf_volume.commit()


@app.function(image=image, gpu=GPU, volumes=VOLUMES, secrets=SECRETS, timeout=SMOKE_TIMEOUT_S, memory=SWEEP_MEMORY_MB)
def layer_sweep(base: str, train_file: str, dev_file: str, out: str) -> None:
    _run("python", "-c", KERNEL_CHECK)
    try:
        _run(
            "python",
            "-m",
            "rein.eval.layer_sweep",
            "--base",
            base,
            "--train",
            train_file,
            "--dev",
            dev_file,
            "--out",
            out,
        )
    finally:
        data_volume.commit()
        hf_volume.commit()


@app.local_entrypoint()
def sweep(base: str, train_file: str, dev_file: str, out: str) -> None:
    layer_sweep.remote(base, f"{DATA_MOUNT}/{train_file}", f"{DATA_MOUNT}/{dev_file}", f"{DATA_MOUNT}/{out}")


@app.local_entrypoint()
def main(action: str = "smoke", config: str = "", data_dir: str = "", base: str = "Qwen/Qwen3.5-0.8B-Base") -> None:
    if data_dir:
        local = Path(data_dir)
        with data_volume.batch_upload(force=True) as batch:
            batch.put_directory(str(local), f"/{local.name}")
        print(f"uploaded {local} to the rein-data volume")
    if action == "kernels":
        kernels.remote()
    elif action == "smoke":
        smoke.remote(base)
    elif action == "debug":
        train.remote(config, [], True)
    elif action == "train":
        if not config:
            raise SystemExit("--config is required for train")
        import yaml

        settings = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
        data_dir = Path(settings["train_file"]).parent.as_posix()
        train.remote(config, [f"{data_dir}/{name}" for name in EVAL_SPLIT_FILES])
    elif action != "upload":
        raise SystemExit(f"unknown action {action!r}")


baseline_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl")
    .pip_install(
        "uv",
        "tinyjev[torch]",
        "httpx>=0.28",
        "pydantic>=2.9",
        "pyyaml>=6.0",
        "python-dotenv>=1.0",
        "huggingface_hub>=0.34",
    )
    .run_commands(f"git clone --depth 1 {KEV_REPO} {KEV_DIR}", f"cd {KEV_DIR} && uv sync --extra serve")
    .env({"HF_HOME": HF_MOUNT, "PYTHONPATH": f"{REPO_MOUNT}/src"})
    .add_local_dir("src", f"{REPO_MOUNT}/src")
)


def _wait_ready(port: int, process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + SERVER_READY_TIMEOUT_S
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"baseline server exited with {process.returncode}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=SERVER_POLL_S)
            return
        except urllib.error.HTTPError:
            return
        except OSError:
            time.sleep(SERVER_POLL_S)
    raise TimeoutError("baseline server did not become ready")


@app.function(image=image, gpu=GPU, volumes=VOLUMES, secrets=SECRETS, timeout=SMOKE_TIMEOUT_S)
def export_model(run: str, repo: str, verify_data: str, card: str) -> None:
    run_dir = f"{DATA_MOUNT}/runs/{run}"
    try:
        _run(
            "python",
            "-m",
            "rein.export.merge",
            "--run",
            run_dir,
            "--out",
            f"{DATA_MOUNT}/exports/{run}",
            "--verify-data",
            f"{DATA_MOUNT}/{verify_data}",
            "--repo",
            repo,
            "--card",
            card,
        )
    finally:
        data_volume.commit()
        hf_volume.commit()


@app.local_entrypoint()
def export(run: str, repo: str, card: str, verify_data: str = "mix-v1/test.jsonl") -> None:
    export_model.remote(run, repo, verify_data, card)


@app.function(image=image, gpu=GPU, volumes=VOLUMES, secrets=SECRETS, timeout=EVAL_TIMEOUT_S)
def evaluate_rein(run: str, data: list[str], out: str) -> None:
    subprocess.run(
        ["python", "-m", "rein.eval.run_eval", "--run", run, "--data", *data, "--out", out], cwd=REPO_MOUNT, check=True
    )
    data_volume.commit()


@app.function(image=baseline_image, gpu=GPU, volumes=VOLUMES, secrets=SECRETS, timeout=EVAL_TIMEOUT_S)
def evaluate_baseline(name: str, data: list[str], out: str, rows: list[str]) -> None:
    """A System One baseline scored on the same items; `rows` (name=path of saved Rein rows) joins them in one report
    with paired deltas against the row set named `rein`."""
    spec = BASELINES[name]
    server = subprocess.Popen(spec["command"])
    try:
        _wait_ready(int(spec["port"]), server)
        subprocess.run(
            [
                "python",
                "-m",
                "rein.eval.run_eval",
                "--data",
                *data,
                "--out",
                out,
                "--baseline-url",
                f"{name}=http://127.0.0.1:{spec['port']}",
                "--baseline-concurrency",
                str(spec.get("concurrency", DEFAULT_BASELINE_CONCURRENCY)),
                *(["--rows", *rows] if rows else []),
            ],
            cwd=REPO_MOUNT,
            check=True,
        )
    finally:
        server.terminate()
        data_volume.commit()


@app.local_entrypoint()
def evaluate(target: str, data: str, out: str, run: str = "", rows: str = "") -> None:
    """`rows`: comma-separated name=path pairs of saved prediction rows (paths under the data volume) to score beside a
    baseline, for example `rein=runs/rein-4b-r1-L18/eval-test/rows_rein.jsonl`."""
    files = [f"{DATA_MOUNT}/{path}" for path in data.split(",")]
    if target == "rein":
        evaluate_rein.remote(f"{DATA_MOUNT}/{run}", files, f"{DATA_MOUNT}/{out}")
    else:
        pairs = [
            f"{name}={DATA_MOUNT}/{path}" for name, path in (pair.split("=", 1) for pair in rows.split(",") if pair)
        ]
        evaluate_baseline.remote(target, files, f"{DATA_MOUNT}/{out}", pairs)


E2E_TIMEOUT_S = 3 * HOURS
E2E_DIR = f"{DATA_MOUNT}/e2e"


@app.function(image=image, gpu=GPU, volumes=VOLUMES, secrets=SECRETS, timeout=E2E_TIMEOUT_S)
def e2e_scenario(scenario: str, rein: str, generator: str, out: str, seeds: int) -> None:
    _run("python", "-c", KERNEL_CHECK)
    try:
        _run_committing(
            "python",
            "-m",
            "rein.e2e.scenario",
            "--scenario",
            scenario,
            "--rein",
            f"{DATA_MOUNT}/runs/{rein}",
            "--generator",
            generator,
            "--out",
            f"{E2E_DIR}/{out}",
            "--seeds",
            str(seeds),
        )
    finally:
        data_volume.commit()
        hf_volume.commit()


@app.local_entrypoint()
def e2e(scenarios: str, reins: str, generator: str, tag: str, seeds: int = 5) -> None:
    """Every scenario against every decider, one container each; outputs land in /e2e/<scenario>-<rein>-<tag>."""
    jobs = [
        (scenario, rein, generator, f"{Path(scenario).stem}-{rein}-{tag}", seeds)
        for scenario in scenarios.split(",")
        for rein in reins.split(",")
    ]
    list(e2e_scenario.starmap(jobs))
