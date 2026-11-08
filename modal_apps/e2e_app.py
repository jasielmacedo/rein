"""Phase 2, stage 5 on Modal: fit, compile and play the end-to-end sheets (DR-022, DR-024, DR-025).

- `prepare`: Qwen's own replies in each sheet's views, the states at the reading block, the rule readings, and the
  per-sheet fit for each example source;
- `build`: the tool neurons' strengths, the compiled checkpoint, and both checkpoints in GGUF Q4_K_M;
- `run`: the three arms in Transformers and in llama.cpp, with the sessions pulled to runs/phase2/e2e.
"""

import json
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

import modal
from images import (
    DATA_MOUNT,
    LLAMA_CPP_DIR,
    REPO_MOUNT,
    SECRETS,
    VOLUMES,
    JobClock,
    data_volume,
    hf_volume,
    llamacpp_image,
    llamacpp_run_image,
    torch_image,
    vllm_image,
)
from llamacpp import serving

APP_NAME = "rein-e2e"
MODEL_ID = "Qwen/Qwen3.5-4B"
RUNS = "phase2/runs/e2e"
READER_LAYER = 17
BLOCK = 20
COMPILER = "phase2/runs/full-nt/compiler-nt20-e16-bin"
H100, L4, CPU = "H100", "L4", "CPU"
HOURS = 60 * 60
CPU_CORES = 8.0
BIG_MEMORY_MB = 64 * 1024
QUANT = "Q4_K_M"
ARMS = ("prompt", "bases", "compiled")
ORIGINAL, COMPILED = "original", "compiled"

app = modal.App(APP_NAME)
torch_img = torch_image()
vllm_img = vllm_image()
llama_img = llamacpp_image()
llama_run_img = llamacpp_run_image()


def _remote(*parts: str) -> PurePosixPath:
    return PurePosixPath(DATA_MOUNT) / RUNS / PurePosixPath(*parts)


def _python(*args: str) -> None:
    subprocess.run(["python", "-m", *args], cwd=REPO_MOUNT, check=True)


def _snapshot() -> str:
    from huggingface_hub import snapshot_download

    return str(snapshot_download(MODEL_ID))


@app.function(image=vllm_img, gpu=H100, volumes=VOLUMES, secrets=SECRETS, timeout=2 * HOURS)
def replies(sheet: str) -> dict[str, Any]:
    clock = JobClock(H100)
    _python(
        "rein.genview.replies",
        "--views",
        str(_remote(sheet, "views.jsonl")),
        "--model",
        MODEL_ID,
        "--out",
        str(_remote(sheet, "views-replied.jsonl")),
    )
    data_volume.commit()
    hf_volume.commit()
    return {"cost": clock.report()}


@app.function(image=torch_img, gpu=H100, volumes=VOLUMES, secrets=SECRETS, timeout=2 * HOURS)
def states_and_readings(sheet: str) -> dict[str, Any]:
    clock = JobClock(H100)
    _python(
        "rein.neurons.extract",
        "--views",
        str(_remote(sheet, "views-replied.jsonl")),
        "--model",
        MODEL_ID,
        "--out",
        str(_remote(sheet, "states")),
        "--layers",
        str(BLOCK),
    )
    _python(
        "rein.compiler.readings",
        "--data",
        str(_remote(sheet, "records", "all")),
        "--bases",
        str(_remote(sheet, "bases.jsonl")),
        "--model",
        MODEL_ID,
        "--layer",
        str(READER_LAYER),
        "--out",
        str(_remote(sheet, "readings")),
        "--tool-binary",
        str(_remote(sheet, "tool_rules.json")),
    )
    data_volume.commit()
    hf_volume.commit()
    return {"cost": clock.report()}


@app.function(image=torch_img, cpu=CPU_CORES, memory=BIG_MEMORY_MB, volumes=VOLUMES, secrets=SECRETS, timeout=2 * HOURS)
def fit(sheet: str, source: str) -> dict[str, Any]:
    import os

    clock = JobClock(CPU)
    os.environ["OMP_NUM_THREADS"] = str(int(CPU_CORES))
    _python(
        "rein.sheets.fit",
        "--sheet",
        f"configs/sheets/{sheet}.yaml",
        "--states",
        str(_remote(sheet, "states")),
        "--readings",
        str(_remote(sheet, "readings")),
        "--data",
        str(_remote(sheet, "records", source)),
        "--compiler",
        f"{DATA_MOUNT}/{COMPILER}",
        "--out",
        str(_remote(sheet, "fits", source)),
    )
    data_volume.commit()
    return {"cost": clock.report()}


@app.function(image=torch_img, gpu=H100, volumes=VOLUMES, secrets=SECRETS, timeout=2 * HOURS)
def calibrate(sheets: list[str]) -> dict[str, Any]:
    clock = JobClock(H100)
    views = [str(_remote(sheet, "views.jsonl")) for sheet in sheets]
    _python(
        "rein.sheets.calibrate",
        "--views",
        *views,
        "--model",
        MODEL_ID,
        "--layer",
        str(BLOCK),
        "--out",
        str(_remote("calibration.json")),
    )
    data_volume.commit()
    hf_volume.commit()
    return {"cost": clock.report()}


@app.function(image=torch_img, cpu=CPU_CORES, memory=BIG_MEMORY_MB, volumes=VOLUMES, secrets=SECRETS, timeout=2 * HOURS)
def compile_checkpoint(sheets: list[str], source: str, steer: list[str], force: float, veto: float) -> dict[str, Any]:
    import shutil

    clock = JobClock(CPU)
    target = Path(str(_remote("checkpoint", COMPILED)))
    if target.exists():
        shutil.rmtree(target)
    _python(
        "rein.sheets.compile",
        "--model",
        _snapshot(),
        "--fits",
        *[str(_remote(sheet, "fits", source)) for sheet in sheets],
        "--vectors",
        str(_remote("vectors.safetensors")),
        "--steer",
        *steer,
        "--force",
        str(force),
        "--veto",
        str(veto),
        "--steer-names",
        str(_remote("steer_names.json")),
        "--out",
        str(target),
    )
    data_volume.commit()
    return {"cost": clock.report()}


@app.function(image=llama_img, gpu=L4, volumes=VOLUMES, secrets=SECRETS, timeout=3 * HOURS)
def gguf() -> dict[str, Any]:
    clock = JobClock(L4)
    out = Path(str(_remote("gguf")))
    out.mkdir(parents=True, exist_ok=True)
    for label, source in ((ORIGINAL, _snapshot()), (COMPILED, str(_remote("checkpoint", COMPILED)))):
        bf16 = out / f"{label}-bf16.gguf"
        subprocess.run(
            ["python", f"{LLAMA_CPP_DIR}/convert_hf_to_gguf.py", source, "--outtype", "bf16", "--outfile", str(bf16)],
            check=True,
        )
        subprocess.run(
            [f"{LLAMA_CPP_DIR}/build/bin/llama-quantize", str(bf16), str(out / f"{label}-{QUANT}.gguf"), QUANT],
            check=True,
        )
        bf16.unlink()
    data_volume.commit()
    return {"cost": clock.report()}


def _sheet_args(sheets: list[str]) -> list[str]:
    return ["--sheets", *[f"configs/sheets/{sheet}.yaml" for sheet in sheets]]


@app.function(image=torch_img, gpu=H100, volumes=VOLUMES, secrets=SECRETS, timeout=4 * HOURS)
def run_transformers(sheets: list[str], arm: str, seeds: int) -> dict[str, Any]:
    clock = JobClock(H100)
    model = str(_remote("checkpoint", COMPILED)) if arm == "compiled" else _snapshot()
    _python(
        "rein.sheets.e2e",
        *_sheet_args(sheets),
        "--arm",
        arm,
        "--runtime",
        "transformers",
        "--model",
        model,
        "--seeds",
        str(seeds),
        "--out",
        str(_remote("sessions")),
    )
    data_volume.commit()
    return {"cost": clock.report()}


@app.function(image=llama_run_img, gpu=L4, volumes=VOLUMES, secrets=SECRETS, timeout=4 * HOURS)
def run_llamacpp(sheets: list[str], arm: str, seeds: int) -> dict[str, Any]:
    clock = JobClock(L4)
    label = COMPILED if arm == "compiled" else ORIGINAL
    with serving(Path(str(_remote("gguf", f"{label}-{QUANT}.gguf"))), parallel=seeds * len(sheets)) as url:
        _python(
            "rein.sheets.e2e",
            *_sheet_args(sheets),
            "--arm",
            arm,
            "--runtime",
            "llamacpp",
            "--model",
            url,
            "--seeds",
            str(seeds),
            "--out",
            str(_remote("sessions")),
        )
    data_volume.commit()
    return {"cost": clock.report()}


@app.function(image=torch_img, gpu=H100, volumes=VOLUMES, secrets=SECRETS, timeout=2 * HOURS)
def firing(sheets: list[str], source: str) -> dict[str, Any]:
    clock = JobClock(H100)
    _python(
        "rein.sheets.firing",
        *_sheet_args(sheets),
        "--fits",
        *[str(_remote(sheet, "fits", source)) for sheet in sheets],
        "--sessions",
        str(_remote("sessions")),
        "--model",
        _snapshot(),
        "--out",
        str(_remote("firing.json")),
    )
    data_volume.commit()
    return {"cost": clock.report()}


def _pull(remote: str, local: Path) -> None:
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(b"".join(data_volume.read_file(remote)))


def _spent(costs: dict[str, Any]) -> float:
    return round(sum(float(entry["cost"]["usd"]) for entry in costs.values()), 2)


@app.local_entrypoint()
def prepare(sheets: str = "maren,ilsa", sources: str = "qwen") -> None:
    """Uploads each sheet's views and records, then replies, states, readings and one fit per example source."""
    names = [name for name in sheets.split(",") if name]
    picked = [source for source in sources.split(",") if source]
    with data_volume.batch_upload(force=True) as batch:
        for sheet in names:
            local = Path("data") / "phase2" / "sheets" / sheet
            for name in ("views.jsonl", "bases.jsonl", "tool_rules.json"):
                batch.put_file(str(local / name), f"/{RUNS}/{sheet}/{name}")
            for source in picked:
                batch.put_file(str(local / source / "train.jsonl"), f"/{RUNS}/{sheet}/records/{source}/train.jsonl")
                batch.put_file(str(local / "script.jsonl"), f"/{RUNS}/{sheet}/records/{source}/gold.jsonl")
            merged = local / "all" / "train.jsonl"
            merged.parent.mkdir(parents=True, exist_ok=True)
            merged.write_text(
                "".join((local / source / "train.jsonl").read_text(encoding="utf-8") for source in picked),
                encoding="utf-8",
            )
            batch.put_file(str(merged), f"/{RUNS}/{sheet}/records/all/train.jsonl")
            batch.put_file(str(local / "script.jsonl"), f"/{RUNS}/{sheet}/records/all/gold.jsonl")
    costs: dict[str, Any] = {}
    for sheet in names:
        costs[f"replies:{sheet}"] = replies.remote(sheet)
        costs[f"states:{sheet}"] = states_and_readings.remote(sheet)
    calls = {f"fit:{sheet}:{source}": fit.spawn(sheet, source) for sheet in names for source in picked}
    costs.update({name: call.get() for name, call in calls.items()})
    _pull_fits(names, picked)
    print(json.dumps({"costs": costs, "estimated_usd": _spent(costs)}, indent=2), flush=True)


@app.local_entrypoint()
def refit(sheets: str = "maren,ilsa", sources: str = "qwen") -> None:
    """Only the per-sheet fits, on states and readings already prepared."""
    names = [name for name in sheets.split(",") if name]
    picked = [source for source in sources.split(",") if source]
    calls = {f"fit:{sheet}:{source}": fit.spawn(sheet, source) for sheet in names for source in picked}
    costs = {name: call.get() for name, call in calls.items()}
    _pull_fits(names, picked)
    print(json.dumps({"costs": costs, "estimated_usd": _spent(costs)}, indent=2), flush=True)


def _pull_fits(names: list[str], sources: list[str]) -> None:
    for sheet in names:
        for source in sources:
            for name in ("neurons.safetensors", "neurons.json", "fit.json"):
                _pull(
                    f"/{RUNS}/{sheet}/fits/{source}/{name}",
                    Path("runs") / "phase2" / "e2e" / sheet / "fits" / source / name,
                )


@app.local_entrypoint()
def build(
    sheets: str = "maren,ilsa", source: str = "qwen", steer: str = "", force: float = 0.0, veto: float = 0.0
) -> None:
    """Calibrates the tool strengths (unless given), compiles the sheets, and converts both checkpoints to GGUF."""
    names = [name for name in sheets.split(",") if name]
    costs: dict[str, Any] = {}
    if not force or not veto:
        costs["calibrate"] = calibrate.remote(names)
        _pull(f"/{RUNS}/calibration.json", Path("runs") / "phase2" / "e2e" / "calibration.json")
        report = json.loads((Path("runs") / "phase2" / "e2e" / "calibration.json").read_text(encoding="utf-8"))
        force = force or report["force_ratio"] * report["residual_norm"]
        veto = veto or report["veto_ratio"] * report["residual_norm"]
    local_steer = Path("data") / "phase2" / "steer"
    with data_volume.batch_upload(force=True) as batch:
        batch.put_file(str(local_steer / "vectors.safetensors"), f"/{RUNS}/vectors.safetensors")
        batch.put_file(str(local_steer / "steer_names.json"), f"/{RUNS}/steer_names.json")
    costs["compile"] = compile_checkpoint.remote(
        names, source, [item for item in steer.split(",") if item], force, veto
    )
    costs["gguf"] = gguf.remote()
    print(
        json.dumps({"force": force, "veto": veto, "costs": costs, "estimated_usd": _spent(costs)}, indent=2), flush=True
    )


@app.local_entrypoint()
def run(
    sheets: str = "maren,ilsa", seeds: int = 5, runtimes: str = "transformers,llamacpp", source: str = "qwen"
) -> None:
    """Plays every arm in every runtime and pulls the sessions."""
    names = [name for name in sheets.split(",") if name]
    chosen = [runtime for runtime in runtimes.split(",") if runtime]
    calls = {}
    for arm in ARMS:
        if "transformers" in chosen:
            calls[f"transformers:{arm}"] = run_transformers.spawn(names, arm, seeds)
        if "llamacpp" in chosen:
            calls[f"llamacpp:{arm}"] = run_llamacpp.spawn(names, arm, seeds)
    costs = {name: call.get() for name, call in calls.items()}
    costs.update(_readback(names, source))
    print(json.dumps({"costs": costs, "estimated_usd": _spent(costs)}, indent=2), flush=True)


def _readback(names: list[str], source: str) -> dict[str, Any]:
    local = Path("runs") / "phase2" / "e2e"
    for entry in data_volume.listdir(f"/{RUNS}/sessions"):
        _pull(f"/{entry.path}", local / "sessions" / Path(entry.path).name)
    costs = {"firing": firing.remote(names, source)}
    _pull(f"/{RUNS}/firing.json", local / "firing.json")
    return costs


@app.local_entrypoint()
def readback(sheets: str = "maren,ilsa", source: str = "qwen") -> None:
    """Pulls the sessions and reads back what the compiled neurons decided in each of them."""
    costs = _readback([name for name in sheets.split(",") if name], source)
    print(json.dumps({"costs": costs, "estimated_usd": _spent(costs)}, indent=2), flush=True)
