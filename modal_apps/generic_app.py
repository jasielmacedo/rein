"""Phase 2, the generic checkpoint on Modal (DR-026): one checkpoint, fixed neurons, every agent's rules in the request.

- `build`: the category neurons fitted on the states with Qwen's own replies, the checkpoint compiled with the rules
  chat template, group 2 (does it stay Qwen when nothing fires), and both checkpoints in GGUF (Q8_0 and Q4_K_M);
- `run`: the end-to-end sheets, Qwen3.5-4B and the checkpoint with the very same prompt, in Transformers and
  llama.cpp, with the sessions pulled to runs/phase2/generic.
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
    llamacpp_image,
    llamacpp_run_image,
    ollama_image,
    torch_image,
)
from llamacpp import serving
from ollama import ollama_serving

APP_NAME = "rein-generic"
MODEL_ID = "Qwen/Qwen3.5-4B"
PHASE_RUNS = "phase2/runs"
DEFAULT_VARIANT = "generic"
DATA_NAME = "mix-v1-audited"
CATEGORIES = "phase2/record_categories.json"
GENERAL_TEXT = "phase2/general/wikitext-2-test.txt"
LAYER = 20
H100, L4, CPU = "H100", "L4", "CPU"
HOURS = 60 * 60
CPU_CORES = 8.0
BIG_MEMORY_MB = 64 * 1024
QUANTS = ("Q8_0", "Q4_K_M")
SLOT_CONTEXT = 32 * 1024
EXACT_TENSORS = rf"blk\.{LAYER}\.ffn_"
EXACT_SUFFIX = "-exact"
OLLAMA = "ollama"
ORIGINAL, GENERIC = "original", "generic"
RULES_ARM, GENERIC_ARM = "rules", "generic"
REASONED = "views-reasoned.jsonl"
PULLED = ("neurons/category_neurons.json", "neurons/category_neurons_report.json", "sameness/sameness.json")

app = modal.App(APP_NAME)
torch_img = torch_image()
llama_img = llamacpp_image()
llama_run_img = llamacpp_run_image()
ollama_img = ollama_image()


def _remote(variant: str, *parts: str) -> PurePosixPath:
    return PurePosixPath(DATA_MOUNT) / PHASE_RUNS / variant / PurePosixPath(*parts)


def _run(name: str) -> PurePosixPath:
    return PurePosixPath(DATA_MOUNT) / PHASE_RUNS / name


def _python(*args: str) -> None:
    subprocess.run(["python", "-m", *args], cwd=REPO_MOUNT, check=True)


def _snapshot() -> str:
    from huggingface_hub import snapshot_download

    return str(snapshot_download(MODEL_ID))


def _checkpoint(variant: str) -> str:
    return str(_remote(variant, "checkpoint"))


@app.function(image=torch_img, gpu=H100, volumes=VOLUMES, secrets=SECRETS, timeout=3 * HOURS)
def fit_neurons(variant: str, states_run: str, steer: list[str], gate_floor: float) -> dict[str, Any]:
    clock = JobClock(H100)
    data_volume.reload()
    _python(
        "rein.compiler.category_neurons",
        "--states",
        str(_run(states_run) / "states"),
        "--data",
        f"{DATA_MOUNT}/{DATA_NAME}",
        "--categories",
        f"{DATA_MOUNT}/{CATEGORIES}",
        "--layer",
        str(LAYER),
        "--out",
        str(_remote(variant, "neurons")),
        "--steer",
        *steer,
        "--gate-floor",
        str(gate_floor),
    )
    data_volume.commit()
    return {"cost": clock.report()}


@app.function(image=torch_img, cpu=CPU_CORES, memory=BIG_MEMORY_MB, volumes=VOLUMES, secrets=SECRETS, timeout=HOURS)
def compile_generic(variant: str, vectors_run: str, steer: list[str], force: float, veto: float) -> dict[str, Any]:
    import shutil

    clock = JobClock(CPU)
    data_volume.reload()
    target = Path(_checkpoint(variant))
    if target.exists():
        shutil.rmtree(target)
    _python(
        "rein.compiler.category_compile",
        "--model",
        _snapshot(),
        "--neurons",
        str(_remote(variant, "neurons")),
        "--vectors",
        str(_run(vectors_run) / "vectors.safetensors"),
        "--steer",
        *steer,
        "--force",
        str(force),
        "--veto",
        str(veto),
        "--out",
        str(target),
    )
    data_volume.commit()
    return {"cost": clock.report()}


@app.function(image=torch_img, gpu=H100, volumes=VOLUMES, secrets=SECRETS, timeout=2 * HOURS)
def sameness(variant: str, views_run: str) -> dict[str, Any]:
    clock = JobClock(H100)
    data_volume.reload()
    _python(
        "rein.compiler.sameness",
        "--original",
        _snapshot(),
        "--checkpoint",
        _checkpoint(variant),
        "--neurons",
        str(_remote(variant, "neurons")),
        "--views",
        str(_run(views_run) / REASONED),
        "--data",
        f"{DATA_MOUNT}/{DATA_NAME}",
        "--categories",
        f"{DATA_MOUNT}/{CATEGORIES}",
        "--text",
        f"{DATA_MOUNT}/{GENERAL_TEXT}",
        "--out",
        str(_remote(variant, "sameness")),
    )
    data_volume.commit()
    return {"cost": clock.report()}


@app.function(image=llama_img, gpu=L4, volumes=VOLUMES, secrets=SECRETS, timeout=3 * HOURS)
def gguf(variant: str, labels: list[str], exact_block: bool) -> dict[str, Any]:
    """Each checkpoint in GGUF at every quantization; with `exact_block`, a copy that keeps the FFN of the neurons'
    block in bf16 (the checkpoint's own precision), named with EXACT_SUFFIX."""
    clock = JobClock(L4)
    data_volume.reload()
    out = Path(str(_remote(variant, "gguf")))
    out.mkdir(parents=True, exist_ok=True)
    sources = {ORIGINAL: _snapshot, GENERIC: lambda: _checkpoint(variant)}
    quantize = f"{LLAMA_CPP_DIR}/build/bin/llama-quantize"
    for label in labels:
        bf16 = out / f"{label}-bf16.gguf"
        subprocess.run(
            [
                "python",
                f"{LLAMA_CPP_DIR}/convert_hf_to_gguf.py",
                sources[label](),
                "--outtype",
                "bf16",
                "--outfile",
                str(bf16),
            ],
            check=True,
        )
        for quant in QUANTS:
            if exact_block:
                keep = ["--tensor-type", f"{EXACT_TENSORS}=bf16"]
                target = out / f"{label}-{quant}{EXACT_SUFFIX}.gguf"
                subprocess.run([quantize, *keep, str(bf16), str(target), quant], check=True)
            else:
                subprocess.run([quantize, str(bf16), str(out / f"{label}-{quant}.gguf"), quant], check=True)
        bf16.unlink()
    data_volume.commit()
    return {"cost": clock.report()}


def _sheet_args(sheets: list[str]) -> list[str]:
    return ["--sheets", *[f"configs/sheets/{sheet}.yaml" for sheet in sheets]]


def _model_for(variant: str, arm: str) -> str:
    return _checkpoint(variant) if arm == GENERIC_ARM else _snapshot()


@app.function(image=torch_img, gpu=H100, volumes=VOLUMES, secrets=SECRETS, timeout=4 * HOURS)
def run_transformers(variant: str, sheets: list[str], arm: str, seeds: int) -> dict[str, Any]:
    clock = JobClock(H100)
    data_volume.reload()
    _python(
        "rein.sheets.e2e",
        *_sheet_args(sheets),
        "--arm",
        arm,
        "--runtime",
        "transformers",
        "--model",
        _model_for(variant, arm),
        "--seeds",
        str(seeds),
        "--out",
        str(_remote(variant, "sessions", "transformers")),
    )
    data_volume.commit()
    return {"cost": clock.report()}


@app.function(image=llama_run_img, gpu=L4, volumes=VOLUMES, secrets=SECRETS, timeout=4 * HOURS)
def run_llamacpp(variant: str, sheets: list[str], arm: str, seeds: int, quant: str) -> dict[str, Any]:
    clock = JobClock(L4)
    data_volume.reload()
    label = GENERIC if arm == GENERIC_ARM else ORIGINAL
    gguf_path = Path(str(_remote(variant, "gguf", f"{label}-{quant}.gguf")))
    with serving(gguf_path, context=SLOT_CONTEXT, parallel=seeds * len(sheets)) as url:
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
            str(_remote(variant, "sessions", f"llamacpp-{quant}")),
        )
    data_volume.commit()
    return {"cost": clock.report()}


@app.function(image=ollama_img, gpu=L4, volumes=VOLUMES, secrets=SECRETS, timeout=4 * HOURS)
def run_ollama(variant: str, sheets: list[str], arm: str, seeds: int, quant: str) -> dict[str, Any]:
    clock = JobClock(L4)
    data_volume.reload()
    label = GENERIC if arm == GENERIC_ARM else ORIGINAL
    gguf_path = Path(str(_remote(variant, "gguf", f"{label}-{quant}.gguf")))
    with ollama_serving(gguf_path, context=SLOT_CONTEXT, parallel=seeds * len(sheets)) as url:
        _python(
            "rein.sheets.e2e",
            *_sheet_args(sheets),
            "--arm",
            arm,
            "--runtime",
            "ollama",
            "--model",
            url,
            "--seeds",
            str(seeds),
            "--out",
            str(_remote(variant, "sessions", f"ollama-{quant}")),
        )
    data_volume.commit()
    return {"cost": clock.report()}


def _pull(remote: str, local: Path) -> None:
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(b"".join(data_volume.read_file(remote)))


def _spent(costs: dict[str, Any]) -> float:
    return round(sum(float(entry["cost"]["usd"]) for entry in costs.values()), 2)


@app.local_entrypoint()
def build(
    states_run: str = "full-block-reply",
    vectors_run: str = "steer-categories",
    steer: str = "",
    force: float = 15.76,
    veto: float = 15.76,
    steps: str = "fit,compile,sameness,gguf",
    variant: str = DEFAULT_VARIANT,
    gate_floor: float = 0.0,
) -> None:
    """Fits the neurons, compiles the checkpoint, measures group 2 and converts both checkpoints to GGUF."""
    chosen = [step for step in steps.split(",") if step]
    strengths = [item for item in steer.split(",") if item]
    categories = [item.split("=")[0] for item in strengths]
    with data_volume.batch_upload(force=True) as batch:
        batch.put_file(str(Path("data") / GENERAL_TEXT), f"/{GENERAL_TEXT}")
    costs: dict[str, Any] = {}
    if "fit" in chosen:
        costs["fit"] = fit_neurons.remote(variant, states_run, categories, gate_floor)
    if "compile" in chosen:
        costs["compile"] = compile_generic.remote(variant, vectors_run, strengths, force, veto)
    pending = gguf.spawn(variant, [ORIGINAL, GENERIC], False) if "gguf" in chosen else None
    if "sameness" in chosen:
        costs["sameness"] = sameness.remote(variant, states_run)
    if pending is not None:
        costs["gguf"] = pending.get()
    local = Path("runs") / "phase2" / variant
    for name in PULLED:
        try:
            _pull(f"/{PHASE_RUNS}/{variant}/{name}", local / name)
        except (FileNotFoundError, modal.exception.NotFoundError):
            continue
    print(json.dumps({"costs": costs, "estimated_usd": _spent(costs)}, indent=2), flush=True)


@app.function(image=torch_img, gpu=H100, volumes=VOLUMES, secrets=SECRETS, timeout=2 * HOURS)
def readback(variant: str, runtime: str, sheets: list[str]) -> dict[str, Any]:
    clock = JobClock(H100)
    data_volume.reload()
    _python(
        "rein.compiler.generic_readback",
        "--sessions",
        str(_remote(variant, "sessions", runtime)),
        *_sheet_args(sheets),
        "--checkpoint",
        _checkpoint(variant),
        "--neurons",
        str(_remote(variant, "neurons")),
        "--out",
        str(_remote(variant, "readback", runtime)),
    )
    data_volume.commit()
    return {"cost": clock.report()}


@app.local_entrypoint()
def firing(variant: str = DEFAULT_VARIANT, runtime: str = "transformers", sheets: str = "maren-long,ilsa-long") -> None:
    """Where the neurons fired in a runtime's sessions, read back through the checkpoint, pulled to runs/phase2."""
    cost = readback.remote(variant, runtime, [name for name in sheets.split(",") if name])
    name = "readback.jsonl"
    _pull(
        f"/{PHASE_RUNS}/{variant}/readback/{runtime}/{name}",
        Path("runs") / "phase2" / variant / "readback" / runtime / name,
    )
    print(json.dumps({"cost": cost}, indent=2), flush=True)


@app.local_entrypoint()
def exact_gguf(variant: str = DEFAULT_VARIANT) -> None:
    """The checkpoint in GGUF with its neurons' block kept in bf16 and everything else quantized."""
    cost = gguf.remote(variant, [GENERIC], True)
    print(json.dumps({"cost": cost}, indent=2), flush=True)


@app.local_entrypoint()
def run(
    sheets: str = "maren,ilsa",
    seeds: int = 5,
    runtimes: str = "transformers,llamacpp-Q4_K_M",
    variant: str = DEFAULT_VARIANT,
    arms: str = f"{RULES_ARM},{GENERIC_ARM}",
) -> None:
    """Plays both arms in every runtime and pulls the sessions."""
    names = [name for name in sheets.split(",") if name]
    calls = {}
    for runtime in [item for item in runtimes.split(",") if item]:
        for arm in [item for item in arms.split(",") if item]:
            if runtime == "transformers":
                calls[f"{runtime}:{arm}"] = run_transformers.spawn(variant, names, arm, seeds)
            elif runtime.startswith(OLLAMA):
                calls[f"{runtime}:{arm}"] = run_ollama.spawn(variant, names, arm, seeds, runtime.split("-", 1)[1])
            else:
                calls[f"{runtime}:{arm}"] = run_llamacpp.spawn(variant, names, arm, seeds, runtime.split("-", 1)[1])
    costs = {name: call.get() for name, call in calls.items()}
    local = Path("runs") / "phase2" / variant / "sessions"
    for folder in data_volume.listdir(f"/{PHASE_RUNS}/{variant}/sessions"):
        for entry in data_volume.listdir(f"/{folder.path}"):
            _pull(f"/{entry.path}", local / Path(folder.path).name / Path(entry.path).name)
    print(json.dumps({"costs": costs, "estimated_usd": _spent(costs)}, indent=2), flush=True)


@app.function(image=torch_img, gpu=H100, volumes=VOLUMES, secrets=SECRETS, timeout=HOURS)
def evaluate_neurons(variant: str, states_run: str, data: str, categories: str, split: str) -> dict[str, Any]:
    clock = JobClock(H100)
    data_volume.reload()
    _python(
        "rein.compiler.category_neurons",
        "--states",
        str(_run(states_run) / "states"),
        "--data",
        data,
        "--categories",
        categories,
        "--layer",
        str(LAYER),
        "--neurons",
        str(_remote(variant, "neurons")),
        "--out",
        str(_remote(variant, "evaluation", states_run)),
        "--split",
        split,
    )
    data_volume.commit()
    return {"cost": clock.report()}


@app.local_entrypoint()
def transfer(
    states_run: str = "door-block-v2-reply",
    variant: str = DEFAULT_VARIANT,
    data: str = f"{DATA_MOUNT}/{DATA_NAME}",
    categories: str = f"{DATA_MOUNT}/{CATEGORIES}",
    split: str = "development",
) -> None:
    """The fitted neurons, unchanged, on another run's states (another template or door): development split only."""
    cost = evaluate_neurons.remote(variant, states_run, data, categories, split)
    name = "category_neurons_evaluation.json"
    _pull(
        f"/{PHASE_RUNS}/{variant}/evaluation/{states_run}/{name}",
        Path("runs") / "phase2" / variant / "evaluation" / states_run / name,
    )
    print(json.dumps({"cost": cost}, indent=2), flush=True)
