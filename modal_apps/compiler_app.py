"""Phase 2, stage 1B on Modal: does a rule decision fit in FFN neurons of the frozen Qwen3.5-4B?"""

import json
import os
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import modal
from images import (
    DATA_MOUNT,
    REPO_MOUNT,
    SECRETS,
    VOLUMES,
    JobClock,
    data_volume,
    hf_volume,
    run_committing,
    torch_image,
    vllm_image,
)

APP_NAME = "rein-compiler"
MODEL_ID = "Qwen/Qwen3.5-4B"
DATA_NAME = "mix-v1-audited"
RUNS = "phase2/runs"
READER_LAYER = 17
TOOL_RULES = "phase2/tool_rules.json"
ORACLE_RULES = "rules.json"
DIAGNOSED_LAYERS = (14, 17, 20)
REASONING_TOKENS = 768
H100 = "H100"
CPU = "CPU"
FIT_CPUS = 8.0
FIT_MEMORY_MB = 32 * 1024
HOURS = 60 * 60
SWEEP_MEMORY_MB = 96 * 1024
PULLED = (
    "compiler/diagnosis.json",
    "compiler/diagnosis-train.json",
    "compiler/predictions-development.jsonl",
    "compiler/report.json",
    "compiler/compiler.json",
    "compiler/channels.safetensors",
    "compiler/compiler.safetensors",
)
STEPS = ("upload", "reasoning", "readings", "extract", "sweep", "diagnose")
CATEGORIES = "phase2/record_categories.json"
CATEGORY_DIR = "category-probe"
CATEGORY_PREDICTIONS = "predictions-development.jsonl"
STEER_ALPHAS = (0.0, 2.0, 4.0, 8.0)
CATEGORY_PULLED = ("category_probe.json", "category_probabilities.json", "diagnosis.json", CATEGORY_PREDICTIONS)

app = modal.App(APP_NAME)
torch_img = torch_image()
vllm_img = vllm_image()


def _compiler_dir(tag: str) -> str:
    return f"compiler-{tag}" if tag else "compiler"


def _posix(run: str) -> PurePosixPath:
    """A run's directory on the volume, spelled for the Linux containers even when built on Windows."""
    return PurePosixPath(DATA_MOUNT) / RUNS / run


def _remote(run: str) -> Path:
    return Path(DATA_MOUNT) / RUNS / run


def _python(*args: str) -> None:
    subprocess.run(["python", "-m", *args], cwd=REPO_MOUNT, check=True)


@app.function(image=vllm_img, gpu=H100, volumes=VOLUMES, secrets=SECRETS, timeout=4 * HOURS)
def reasoning(run: str, max_tokens: int = REASONING_TOKENS) -> dict[str, Any]:
    clock = JobClock(H100)
    base = _remote(run)
    _python(
        "rein.genview.reasoning",
        "--views",
        str(base / "views.jsonl"),
        "--model",
        MODEL_ID,
        "--out",
        str(base / "views-reasoned.jsonl"),
        "--max-tokens",
        str(max_tokens),
    )
    data_volume.commit()
    hf_volume.commit()
    return {"cost": clock.report()}


@app.function(image=torch_img, gpu=H100, volumes=VOLUMES, secrets=SECRETS, timeout=4 * HOURS)
def readings(run: str, layer: int = READER_LAYER, tool_binary: str = "") -> dict[str, Any]:
    clock = JobClock(H100)
    base = _remote(run)
    _python(
        "rein.compiler.readings",
        "--data",
        f"{DATA_MOUNT}/{DATA_NAME}",
        "--bases",
        str(base / "bases.jsonl"),
        "--model",
        MODEL_ID,
        "--layer",
        str(layer),
        "--out",
        str(base / "readings"),
        *(["--tool-binary", tool_binary] if tool_binary else []),
    )
    data_volume.commit()
    hf_volume.commit()
    return {"cost": clock.report()}


@app.function(image=torch_img, gpu=H100, volumes=VOLUMES, secrets=SECRETS, timeout=8 * HOURS, retries=2)
def extract(run: str, layers: list[int] | None = None) -> dict[str, Any]:
    clock = JobClock(H100)
    base = _remote(run)
    reasoned = base / "views-reasoned.jsonl"
    views = reasoned if reasoned.exists() else base / "views.jsonl"
    args = ["python", "-m", "rein.neurons.extract", "--views", str(views)]
    args += ["--model", MODEL_ID, "--out", str(base / "states")]
    if layers:
        args += ["--layers", *map(str, layers)]
    run_committing(args, data_volume)
    hf_volume.commit()
    return {"cost": clock.report()}


@app.function(image=torch_img, gpu=H100, volumes=VOLUMES, secrets=SECRETS, timeout=6 * HOURS, memory=SWEEP_MEMORY_MB)
def sweep(
    run: str,
    layers: list[int] | None = None,
    epochs: int = 4,
    readings_run: str = "",
    tag: str = "",
    extra: list[str] | None = None,
    with_states: list[str] | None = None,
) -> dict[str, Any]:
    clock = JobClock(H100)
    base = _remote(run)
    readings = _remote(readings_run or run) / "readings"
    states = [str(base / "states"), *(str(_remote(other) / "states") for other in with_states or [])]
    args = ["python", "-m", "rein.compiler.run", "--states", *states, "--readings", str(readings)]
    args += ["--data", f"{DATA_MOUNT}/{DATA_NAME}", "--out", str(base / _compiler_dir(tag)), "--epochs", str(epochs)]
    if layers:
        args += ["--layers", *map(str, layers)]
    run_committing([*args, *(extra or [])], data_volume)
    return {"cost": clock.report()}


@app.function(image=torch_img, gpu=H100, volumes=VOLUMES, secrets=SECRETS, timeout=2 * HOURS, memory=SWEEP_MEMORY_MB)
def diagnose(
    run: str, layers: list[int], readings_run: str = "", tag: str = "", split: str = "", tool_binary: str = ""
) -> dict[str, Any]:
    clock = JobClock(H100)
    base = _remote(run)
    _python(
        "rein.compiler.evaluate",
        "--states",
        str(base / "states"),
        "--readings",
        str(_remote(readings_run or run) / "readings"),
        "--data",
        f"{DATA_MOUNT}/{DATA_NAME}",
        "--compiler",
        str(base / _compiler_dir(tag)),
        "--layers",
        *map(str, layers),
        *(["--split", split] if split else []),
        *(["--tool-binary", tool_binary] if tool_binary else []),
    )
    data_volume.commit()
    return {"cost": clock.report()}


@app.function(image=torch_img, gpu=H100, volumes=VOLUMES, secrets=SECRETS, timeout=2 * HOURS, memory=SWEEP_MEMORY_MB)
def module(name: str, args: list[str]) -> dict[str, Any]:
    """Runs one project module against the volume, for analyses that need the stored states."""
    clock = JobClock(H100)
    data_volume.reload()
    _python(name, *args)
    data_volume.commit()
    return {"cost": clock.report()}


@app.function(image=torch_img, cpu=FIT_CPUS, memory=FIT_MEMORY_MB, volumes=VOLUMES, secrets=SECRETS, timeout=2 * HOURS)
def cpu_module(name: str, args: list[str]) -> dict[str, Any]:
    """Runs one project module that reads the stored states but needs no GPU, with as many threads as its CPUs: torch
    otherwise starts one per host core and small operations crawl."""
    clock = JobClock(CPU)
    os.environ["OMP_NUM_THREADS"] = str(int(FIT_CPUS))
    data_volume.reload()
    _python(name, *args)
    data_volume.commit()
    return {"cost": clock.report()}


@app.function(image=vllm_img, gpu=H100, volumes=VOLUMES, secrets=SECRETS, timeout=2 * HOURS)
def vllm_module(name: str, args: list[str]) -> dict[str, Any]:
    """Runs one project module that needs vLLM against the volume."""
    clock = JobClock(H100)
    data_volume.reload()
    _python(name, *args)
    data_volume.commit()
    hf_volume.commit()
    return {"cost": clock.report()}


def _upload(run: str) -> None:
    local = Path("data") / "phase2"
    with data_volume.batch_upload(force=True) as batch:
        batch.put_file(str(local / "views" / f"{run}.jsonl"), f"/{RUNS}/{run}/views.jsonl")
        batch.put_file(str(local / "bases" / "bases.jsonl"), f"/{RUNS}/{run}/bases.jsonl")
    present = {entry.path for entry in data_volume.listdir(f"/{DATA_NAME}")} if _exists(f"/{DATA_NAME}") else set()
    if not present:
        with data_volume.batch_upload(force=True) as batch:
            batch.put_directory(str(Path("data") / DATA_NAME), f"/{DATA_NAME}")


def _exists(path: str) -> bool:
    try:
        data_volume.listdir(path)
    except (FileNotFoundError, modal.exception.NotFoundError):
        return False
    return True


def _pull(run: str, tag: str = "") -> Path:
    out = Path("runs") / "phase2" / run
    for name in PULLED:
        name = name.replace("compiler/", f"{_compiler_dir(tag)}/", 1)
        remote = f"/{RUNS}/{run}/{name}"
        if not _exists_file(remote):
            continue
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            for chunk in data_volume.read_file(remote):
                handle.write(chunk)
    return out


def _exists_file(path: str) -> bool:
    parent, _, name = path.rpartition("/")
    return _exists(parent) and any(entry.path.rpartition("/")[2] == name for entry in data_volume.listdir(parent))


@app.local_entrypoint()
def stage1b(
    run: str = "1b",
    steps: str = ",".join(STEPS),
    layers: str = "",
    epochs: int = 4,
    readings_run: str = "",
    tag: str = "",
    split: str = "",
    extra: str = "",
    diagnose_layers: str = "",
    with_states: str = "",
    tool_binary: bool = False,
) -> None:
    chosen = [step for step in steps.split(",") if step]
    unknown = sorted(set(chosen) - set(STEPS))
    if unknown:
        raise SystemExit(f"unknown steps {unknown}")
    picked_layers = [int(layer) for layer in layers.split(",") if layer] or None
    costs: dict[str, Any] = {}
    if "upload" in chosen:
        _upload(run)
    tool_rules = f"{DATA_MOUNT}/{TOOL_RULES}" if tool_binary else ""
    if tool_binary:
        with data_volume.batch_upload(force=True) as batch:
            batch.put_file(str(Path("data") / TOOL_RULES), f"/{TOOL_RULES}")
        extra = f"{extra} --tool-binary {tool_rules}"
    pending_readings = readings.spawn(run, READER_LAYER, tool_rules) if "readings" in chosen else None
    if "reasoning" in chosen:
        costs["reasoning"] = reasoning.remote(run)
    if "extract" in chosen:
        costs["extract"] = extract.remote(run, picked_layers)
    if pending_readings is not None:
        costs["readings"] = pending_readings.get()
    if "sweep" in chosen:
        costs["sweep"] = sweep.remote(
            run, picked_layers, epochs, readings_run, tag, extra.split(), [r for r in with_states.split(",") if r]
        )
    if "diagnose" in chosen:
        diagnosed = [int(layer) for layer in diagnose_layers.split(",") if layer] or picked_layers
        costs["diagnose"] = diagnose.remote(
            run, diagnosed or list(DIAGNOSED_LAYERS), readings_run, tag, split, tool_rules
        )
    if {"sweep", "diagnose"} & set(chosen):
        print(f"pulled the compiler report into {_pull(run, tag)}", flush=True)
    spent = sum(float(entry["cost"]["usd"]) for entry in costs.values())
    print(json.dumps({"costs": costs, "estimated_usd": round(spent, 2)}, indent=2), flush=True)


@app.local_entrypoint()
def oracle(
    run: str = "oracle",
    compiler_run: str = "1b",
    compiler_tag: str = "nt20-f1",
    layers: str = "14,17,20",
    skip_extract: bool = False,
) -> None:
    """Uploads the oracle cases, extracts their states and scores per-rule detectors against a trained compiler."""
    local = Path("data") / "phase2"
    with data_volume.batch_upload(force=True) as batch:
        batch.put_file(str(local / "views" / f"{run}.jsonl"), f"/{RUNS}/{run}/views.jsonl")
        batch.put_file(str(local / "oracle" / "train.jsonl"), f"/{RUNS}/{run}/records/train.jsonl")
    picked = [int(layer) for layer in layers.split(",") if layer]
    costs = {} if skip_extract else {"extract": extract.remote(run, picked)}
    base = _posix(run)
    compiler_dir = _posix(compiler_run) / _compiler_dir(compiler_tag)
    out = base / "oracle.json"
    costs["probe"] = module.remote(
        "rein.compiler.oracle_probe",
        [
            "--states",
            str(base / "states"),
            "--readings",
            str(_posix(compiler_run) / "readings"),
            "--data",
            str(base / "records"),
            "--compiler",
            str(compiler_dir),
            "--layers",
            *map(str, picked),
            "--out",
            str(out),
        ],
    )
    target = Path("runs") / "phase2" / run / "oracle.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"".join(data_volume.read_file(f"/{RUNS}/{run}/oracle.json")))
    print(json.dumps({"costs": costs, "report": str(target)}, indent=2), flush=True)


@app.local_entrypoint()
def native_tools(run: str = "full-nt") -> None:
    """Measures how Qwen3.5 calls tools on its own in the development cases of tool rules."""
    out = _posix(run) / "native_tools.json"
    cost = vllm_module.remote(
        "rein.compiler.native_tools",
        [
            "--views",
            str(_posix(run) / "views.jsonl"),
            "--data",
            f"{DATA_MOUNT}/{DATA_NAME}",
            "--model",
            MODEL_ID,
            "--out",
            str(out),
        ],
    )
    target = Path("runs") / "phase2" / run / "native_tools.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"".join(data_volume.read_file(f"/{RUNS}/{run}/native_tools.json")))
    print(json.dumps({"cost": cost, "report": str(target)}, indent=2), flush=True)


@app.local_entrypoint()
def sheet_fit(
    run: str = "oracle",
    compiler_run: str = "full-nt",
    compiler_tag: str = "",
    readings_run: str = "full-nt-bin",
    tool_binary: bool = False,
) -> None:
    """Fits each oracle rule's neurons from the compiler's and compares them with the compiler and the oracle."""
    with data_volume.batch_upload(force=True) as batch:
        batch.put_file(str(Path("data") / "phase2" / "oracle" / ORACLE_RULES), f"/{RUNS}/{run}/{ORACLE_RULES}")
    compiler_dir = _posix(compiler_run) / _compiler_dir(compiler_tag)
    name = f"sheet_fit-{compiler_tag}.json"
    cost = cpu_module.remote(
        "rein.compiler.sheet_fit",
        [
            "--states",
            str(_posix(run) / "states"),
            "--readings",
            str(_posix(readings_run) / "readings"),
            "--data",
            str(_posix(run) / "records"),
            "--rules",
            str(_posix(run) / ORACLE_RULES),
            "--compiler",
            str(compiler_dir),
            "--out",
            str(_posix(run) / name),
            *(["--tool-binary"] if tool_binary else []),
        ],
    )
    target = Path("runs") / "phase2" / run / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"".join(data_volume.read_file(f"/{RUNS}/{run}/{name}")))
    print(json.dumps({"cost": cost, "report": str(target)}, indent=2), flush=True)


@app.local_entrypoint()
def gold_eval(
    prepared: str = "data/phase2/gold",
    run: str = "gold",
    compiler_run: str = "full-nt",
    compiler_tag: str = "nt20-e16-bin",
    layer: int = 20,
) -> None:
    """Scores a compiler on the hand-written gold set, as prepared by `python -m rein.gold.prepare`."""
    local = Path(prepared)
    with data_volume.batch_upload(force=True) as batch:
        batch.put_file(str(local / "views.jsonl"), f"/{RUNS}/{run}/views.jsonl")
        batch.put_file(str(local / "gold.jsonl"), f"/{RUNS}/{run}/records/gold.jsonl")
        batch.put_file(str(local / "bases.jsonl"), f"/{RUNS}/{run}/bases.jsonl")
        batch.put_file(str(local / "tool_rules.json"), f"/{RUNS}/{run}/tool_rules.json")
    base = _posix(run)
    tool_rules = ["--tool-binary", str(base / "tool_rules.json")]
    costs = {"extract": extract.remote(run, [layer])}
    costs["readings"] = module.remote(
        "rein.compiler.readings",
        [
            "--data",
            str(base / "records"),
            "--bases",
            str(base / "bases.jsonl"),
            "--model",
            MODEL_ID,
            "--layer",
            str(READER_LAYER),
            "--out",
            str(base / "readings"),
            *tool_rules,
        ],
    )
    out = base / compiler_tag
    costs["diagnose"] = module.remote(
        "rein.compiler.evaluate",
        [
            "--states",
            str(base / "states"),
            "--readings",
            str(base / "readings"),
            "--data",
            str(base / "records"),
            "--compiler",
            str(_posix(compiler_run) / _compiler_dir(compiler_tag)),
            "--layers",
            str(layer),
            "--split",
            "gold",
            "--out",
            str(out),
            *tool_rules,
        ],
    )
    target = Path("runs") / "phase2" / run / compiler_tag
    target.mkdir(parents=True, exist_ok=True)
    for name in ("diagnosis-gold.json", "predictions-gold.jsonl"):
        (target / name).write_bytes(b"".join(data_volume.read_file(f"/{RUNS}/{run}/{compiler_tag}/{name}")))
    spent = sum(float(entry["cost"]["usd"]) for entry in costs.values())
    print(json.dumps({"costs": costs, "estimated_usd": round(spent, 2), "report": str(target)}, indent=2), flush=True)


@app.local_entrypoint()
def sheet_writer(sheets: str = "maren,ilsa") -> None:
    """Qwen3.5-4B writes each sheet's example cases (no teacher); the outputs come back to data/phase2/sheets."""
    names = [name for name in sheets.split(",") if name]
    local = Path("data") / "phase2" / "sheets"
    with data_volume.batch_upload(force=True) as batch:
        for name in names:
            batch.put_file(str(local / name / "qwen-prompts.jsonl"), f"/{RUNS}/sheets/{name}/qwen-prompts.jsonl")
    remote = _posix("sheets")
    calls = {
        name: vllm_module.spawn(
            "rein.sheets.qwen_writer",
            [
                "--prompts",
                str(remote / name / "qwen-prompts.jsonl"),
                "--model",
                MODEL_ID,
                "--out",
                str(remote / name / "qwen-written.jsonl"),
            ],
        )
        for name in names
    }
    costs = {name: call.get() for name, call in calls.items()}
    for name in names:
        target = local / name / "qwen-written.jsonl"
        target.write_bytes(b"".join(data_volume.read_file(f"/{RUNS}/sheets/{name}/qwen-written.jsonl")))
    spent = sum(float(entry["cost"]["usd"]) for entry in costs.values())
    print(json.dumps({"costs": costs, "estimated_usd": round(spent, 2)}, indent=2), flush=True)


@app.local_entrypoint()
def steer(sheets: str = "maren,ilsa", source: str = "qwen") -> None:
    """Behavior vectors: Qwen's hold and give-in pairs, the auditor's verdicts, the vectors, and the strength sweep."""
    names = [name for name in sheets.split(",") if name]
    local = Path("data") / "phase2" / "steer"
    remote = _posix("steer")
    sheet_args = ["--sheet", *[f"configs/sheets/{name}.yaml" for name in names]]
    record_args = ["--records", *[f"data/phase2/sheets/{name}/{source}/train.jsonl" for name in names]]
    with data_volume.batch_upload(force=True) as batch:
        batch.put_file(str(local / "prompts.jsonl"), f"/{RUNS}/steer/prompts.jsonl")
    costs = {
        "sample": vllm_module.remote(
            "rein.sheets.steer_gpu",
            ["sample", "--pairs", str(remote / "prompts.jsonl"), "--model", MODEL_ID, "--out", str(remote)],
        )
    }
    (local / "pairs.jsonl").write_bytes(b"".join(data_volume.read_file(f"/{RUNS}/steer/pairs.jsonl")))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "rein.sheets.steer",
            "judge",
            *sheet_args,
            *record_args,
            "--pairs",
            str(local / "pairs.jsonl"),
        ],
        check=True,
    )
    with data_volume.batch_upload(force=True) as batch:
        batch.put_file(str(local / "pairs.jsonl"), f"/{RUNS}/steer/pairs.jsonl")
    for step in ("vectors", "sweep"):
        costs[step] = module.remote(
            "rein.sheets.steer_gpu",
            [step, "--pairs", str(remote / "pairs.jsonl"), "--model", MODEL_ID, "--out", str(remote)],
        )
    for name in ("vectors.safetensors", "vectors.json", "sweep.jsonl"):
        (local / name).write_bytes(b"".join(data_volume.read_file(f"/{RUNS}/steer/{name}")))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "rein.sheets.steer",
            "sweep",
            *sheet_args,
            *record_args,
            "--pairs",
            str(local / "sweep.jsonl"),
        ],
        check=True,
    )
    spent = sum(float(entry["cost"]["usd"]) for entry in costs.values())
    print(json.dumps({"costs": costs, "estimated_usd": round(spent, 2)}, indent=2), flush=True)


@app.local_entrypoint()
def category_probe(
    arms: str = "abl-rules,abl-bases,abl-none",
    readings_run: str = "1b",
    layer: int = 20,
    doors: str = "",
    gpu: bool = False,
) -> None:
    """One generic detector per rule category on each arm's stored states, beside that arm's per-rule compiler when it
    has one, and scored on every other door's states too."""
    names = [name for name in arms.split(",") if name]
    door_names = [name for name in doors.split(",") if name]
    local_data = Path("data")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "rein.compiler.category_probe",
            "map",
            "--data",
            str(local_data / DATA_NAME),
            "--synthetic",
            str(local_data / "synthetic"),
            "--out",
            str(local_data / CATEGORIES),
        ],
        check=True,
    )
    with data_volume.batch_upload(force=True) as batch:
        batch.put_file(str(local_data / CATEGORIES), f"/{CATEGORIES}")
    data = f"{DATA_MOUNT}/{DATA_NAME}"
    compared = [name for name in names if _exists(f"/{RUNS}/{name}/{_compiler_dir('')}")]
    pending = [name for name in compared if not _exists_file(f"/{RUNS}/{name}/{CATEGORY_DIR}/{CATEGORY_PREDICTIONS}")]
    evaluations = {
        name: cpu_module.spawn(
            "rein.compiler.evaluate",
            [
                "--states",
                str(_posix(name) / "states"),
                "--readings",
                str(_posix(readings_run) / "readings"),
                "--data",
                data,
                "--compiler",
                str(_posix(name) / _compiler_dir("")),
                "--layers",
                str(layer),
                "--out",
                str(_posix(name) / CATEGORY_DIR),
            ],
        )
        for name in pending
    }
    costs = {f"evaluate:{name}": call.get() for name, call in evaluations.items()}
    probes = {}
    for name in names:
        other_doors = [
            arg for door in door_names if door != name for arg in ("--door", f"{door}={_posix(door) / 'states'}")
        ]
        comparison = (
            ["--compiler-predictions", str(_posix(name) / CATEGORY_DIR / CATEGORY_PREDICTIONS)]
            if name in compared
            else []
        )
        probes[name] = (module if gpu else cpu_module).spawn(
            "rein.compiler.category_probe",
            [
                "probe",
                "--states",
                str(_posix(name) / "states"),
                "--data",
                data,
                "--categories",
                f"{DATA_MOUNT}/{CATEGORIES}",
                "--layer",
                str(layer),
                "--out",
                str(_posix(name) / CATEGORY_DIR),
                *other_doors,
                *comparison,
            ],
        )
    costs |= {f"probe:{name}": call.get() for name, call in probes.items()}
    for name in names:
        target = Path("runs") / "phase2" / name / CATEGORY_DIR
        target.mkdir(parents=True, exist_ok=True)
        for file in CATEGORY_PULLED:
            remote = f"/{RUNS}/{name}/{CATEGORY_DIR}/{file}"
            if _exists_file(remote):
                (target / file).write_bytes(b"".join(data_volume.read_file(remote)))
    spent = sum(float(entry["cost"]["usd"]) for entry in costs.values())
    print(json.dumps({"costs": costs, "estimated_usd": round(spent, 2)}, indent=2), flush=True)


@app.local_entrypoint()
def reply_states(run: str = "full-block-reply", source: str = "full-block", layer: int = 20) -> None:
    """Qwen3.5's own reply added to each non-thinking view of `source` (with positions inside it), then the states."""
    reasoned = "views-reasoned.jsonl"
    costs = {
        "replies": vllm_module.remote(
            "rein.genview.replies",
            ["--views", str(_posix(source) / reasoned), "--model", MODEL_ID, "--out", str(_posix(run) / reasoned)],
        )
    }
    costs["extract"] = extract.remote(run, [layer])
    spent = sum(float(entry["cost"]["usd"]) for entry in costs.values())
    print(json.dumps({"costs": costs, "estimated_usd": round(spent, 2)}, indent=2), flush=True)


@app.local_entrypoint()
def category_steer(run: str = "steer-categories", views: str = "data/phase2/views/full-block.jsonl") -> None:
    """Behavior vectors per rule category: prompts here, Qwen's pairs on vLLM, the auditor's verdicts here, then the
    vectors and the strength sweep on development attacks, judged here too."""
    local = Path("data") / "phase2" / run
    remote = _posix(run)
    step = [sys.executable, "-m", "rein.compiler.category_steer"]
    data = ["--data", str(Path("data") / DATA_NAME)]
    subprocess.run([*step, "prompts", *data, "--views", views, "--pairs", str(local / "prompts.jsonl")], check=True)
    with data_volume.batch_upload(force=True) as batch:
        batch.put_file(str(local / "prompts.jsonl"), f"/{RUNS}/{run}/prompts.jsonl")
    gpu_args = ["--model", MODEL_ID, "--out", str(remote)]
    costs = {
        "sample": vllm_module.remote(
            "rein.sheets.steer_gpu", ["sample", "--pairs", str(remote / "prompts.jsonl"), *gpu_args]
        )
    }
    (local / "pairs.jsonl").write_bytes(b"".join(data_volume.read_file(f"/{RUNS}/{run}/pairs.jsonl")))
    subprocess.run([*step, "judge", *data, "--pairs", str(local / "pairs.jsonl")], check=True)
    with data_volume.batch_upload(force=True) as batch:
        batch.put_file(str(local / "pairs.jsonl"), f"/{RUNS}/{run}/pairs.jsonl")
    pairs = ["--pairs", str(remote / "pairs.jsonl")]
    costs["vectors"] = module.remote("rein.sheets.steer_gpu", ["vectors", *pairs, *gpu_args])
    costs["sweep"] = module.remote(
        "rein.sheets.steer_gpu", ["sweep", *pairs, *gpu_args, "--alphas", *map(str, STEER_ALPHAS)]
    )
    for name in ("vectors.safetensors", "vectors.json", "sweep.jsonl"):
        (local / name).write_bytes(b"".join(data_volume.read_file(f"/{RUNS}/{run}/{name}")))
    subprocess.run([*step, "sweep", *data, "--pairs", str(local / "sweep.jsonl")], check=True)
    spent = sum(float(entry["cost"]["usd"]) for entry in costs.values())
    print(json.dumps({"costs": costs, "estimated_usd": round(spent, 2)}, indent=2), flush=True)


@app.local_entrypoint()
def view_replies(name: str = "tool-confusion") -> None:
    """Qwen3.5's own reply to each non-thinking view of data/phase2/views/{name}.jsonl, pulled back as token ids."""
    with data_volume.batch_upload(force=True) as batch:
        batch.put_file(str(Path("data") / "phase2" / "views" / f"{name}.jsonl"), f"/{RUNS}/{name}/views.jsonl")
    remote = _posix(name)
    cost = vllm_module.remote(
        "rein.genview.replies",
        ["--views", str(remote / "views.jsonl"), "--model", MODEL_ID, "--out", str(remote / "views-replied.jsonl")],
    )
    target = Path("runs") / "phase2" / name / "views-replied.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"".join(data_volume.read_file(f"/{RUNS}/{name}/views-replied.jsonl")))
    print(json.dumps({"cost": cost, "replies": str(target)}), flush=True)


@app.local_entrypoint()
def steer_duration(run: str = "steer-categories", conditions: str = "0:all,2:all,2:1,4:1,8:1,16:1,4:8") -> None:
    """How long the behavior vector must stay on: replies with it for a number of positions from the reply opening,
    judged here by the auditor."""
    local = Path("data") / "phase2" / run
    remote = _posix(run)
    cost = module.remote(
        "rein.sheets.steer_gpu",
        [
            "duration",
            "--pairs",
            str(remote / "pairs.jsonl"),
            "--model",
            MODEL_ID,
            "--out",
            str(remote),
            "--conditions",
            *[item for item in conditions.split(",") if item],
        ],
    )
    name = "duration.jsonl"
    (local / name).write_bytes(b"".join(data_volume.read_file(f"/{RUNS}/{run}/{name}")))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "rein.compiler.category_steer",
            "sweep",
            "--data",
            str(Path("data") / DATA_NAME),
            "--pairs",
            str(local / name),
        ],
        check=True,
    )
    print(json.dumps({"cost": cost}, indent=2), flush=True)


@app.local_entrypoint()
def tool_probe(out: str = "tools-probe") -> None:
    """Tool detectors by block: fitted on the synthetic records, scored on end-to-end turns and on the gold set."""
    labels = "turn_labels.json"
    with data_volume.batch_upload(force=True) as batch:
        batch.put_file(str(Path("data") / "phase2" / "views" / labels), f"/{RUNS}/tools-e2e/{labels}")
    cost = module.remote(
        "rein.compiler.tool_probe",
        [
            "--synthetic",
            str(_posix("tools-synthetic") / "states"),
            "--data",
            f"{DATA_MOUNT}/{DATA_NAME}",
            "--categories",
            f"{DATA_MOUNT}/{CATEGORIES}",
            "--e2e",
            str(_posix("tools-e2e") / "states"),
            "--e2e-labels",
            str(_posix("tools-e2e") / labels),
            "--gold",
            str(_posix("tools-gold") / "states"),
            "--gold-data",
            str(_posix("gold-block") / "records"),
            "--gold-categories",
            str(_posix("gold-block") / "categories.json"),
            "--out",
            str(_posix(out)),
        ],
    )
    target = Path("runs") / "phase2" / out
    target.mkdir(parents=True, exist_ok=True)
    for name in ("tool_probe.json", "tool_scores.json"):
        (target / name).write_bytes(b"".join(data_volume.read_file(f"/{RUNS}/{out}/{name}")))
    print(json.dumps({"cost": cost, "report": str(target)}, indent=2), flush=True)
