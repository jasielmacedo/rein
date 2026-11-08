"""The soft-token decider inside one GGUF, on Modal.

- `build`: a run's trained rows merged into a plain copy of its base checkpoint (every other tensor identical; with
  `--untie`, the base's table also becomes the copy's own output head), then converted and quantized by llama.cpp, the
  bf16 file kept beside the quantized ones (`quantize` redoes only that last step, skipping the files already there);
- `play_sheets`: the sheets' scripts against the `none` arm (persona only) and the `rein` arm (the same persona, the
  decider asked in Rein's layout through the very same llama-server's logprobs, the app acting on the answers), with
  the sessions left on the volume for the judge;
- `check_silence`: the soft tokens' output mass over held-out conversations;
- `check_agreement`: the GGUF's decisions through llama-server against the run in PyTorch, on held-out records.
"""

import json
import shutil
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
    torch_image,
)
from llamacpp import serving

APP_NAME = "rein-soft"
SOFT_DIR = "soft"
RUNS_DIR = "runs"
UNTIED_SUFFIX = "-untied"
CHECKPOINT, GGUF, SESSIONS = "checkpoint", "gguf", "sessions"
SILENCE_FILE = "silence-test.json"
TEST_DATA = "mix-v1-audited/test.jsonl"
SILENCE_LIMIT = 2000
SILENCE_TOKENS = 4096
AGREEMENT_FILE = "agreement-{quant}.json"
AGREEMENT_RECORDS = 20
LABEL = "soft"
QUANTS = ("Q8_0", "Q4_K_M")
DEFAULT_QUANT = "Q4_K_M"
CONFIG_FILE = "rein_config.json"
CONVERT_TYPE = "bf16"
SLOT_CONTEXT = 32 * 1024
SHEETS_PER_SERVER = 2
STREAMS_PER_SESSION = 2
HOURS = 60 * 60
L4, CPU = "L4", "CPU"
CPU_CORES = 8.0
BIG_MEMORY_MB = 64 * 1024
NONE_ARM, REIN_ARM = "none", "rein"
DEFAULT_SHEETS = ",".join(f"test-v1/test-{index:02d}" for index in range(10))

app = modal.App(APP_NAME)
torch_img = torch_image()
llama_img = llamacpp_image()
llama_run_img = llamacpp_run_image()


def _store(run: str, untie: bool) -> str:
    """Where a run's exports live: the tied and the untied copies side by side."""
    return f"{run}{UNTIED_SUFFIX}" if untie else run


def _remote(store: str, *parts: str) -> PurePosixPath:
    return PurePosixPath(DATA_MOUNT) / SOFT_DIR / store / PurePosixPath(*parts)


def _run_dir(run: str) -> str:
    return f"{DATA_MOUNT}/{RUNS_DIR}/{run}"


def _python(*args: str) -> None:
    subprocess.run(["python", "-m", *args], cwd=REPO_MOUNT, check=True)


def _sheet_args(sheets: list[str]) -> list[str]:
    return ["--sheets", *[f"configs/sheets/{sheet}.yaml" for sheet in sheets]]


@app.function(image=torch_img, cpu=CPU_CORES, memory=BIG_MEMORY_MB, volumes=VOLUMES, secrets=SECRETS, timeout=HOURS)
def merge(run: str, untie: bool) -> dict[str, Any]:
    clock = JobClock(CPU)
    data_volume.reload()
    out = str(_remote(_store(run, untie), CHECKPOINT))
    flags = ["--untie"] if untie else []
    _python("rein.export.soft_merge", "--run", _run_dir(run), "--out", out, "--verify", *flags)
    data_volume.commit()
    return {"cost": clock.report()}


@app.function(image=llama_img, gpu=L4, volumes=VOLUMES, secrets=SECRETS, timeout=2 * HOURS)
def gguf(run: str, untie: bool) -> dict[str, Any]:
    """The llama.cpp binaries are built against CUDA, so converting and quantizing run next to a GPU too."""
    clock = JobClock(L4)
    data_volume.reload()
    store = _store(run, untie)
    out = Path(str(_remote(store, GGUF)))
    out.mkdir(parents=True, exist_ok=True)
    full = out / f"{LABEL}-{CONVERT_TYPE}.gguf"
    if not full.exists():
        subprocess.run(
            [
                "python",
                f"{LLAMA_CPP_DIR}/convert_hf_to_gguf.py",
                str(_remote(store, CHECKPOINT)),
                "--outtype",
                CONVERT_TYPE,
                "--outfile",
                str(full),
            ],
            check=True,
        )
    for quant in QUANTS:
        target = out / f"{LABEL}-{quant}.gguf"
        if not target.exists():
            subprocess.run([f"{LLAMA_CPP_DIR}/build/bin/llama-quantize", str(full), str(target), quant], check=True)
    shutil.copy(Path(_run_dir(run)) / CONFIG_FILE, out / CONFIG_FILE)
    data_volume.commit()
    return {"cost": clock.report(), "sizes": {path.name: path.stat().st_size for path in out.glob("*.gguf")}}


def _sessions_dir(quant: str, tag: str) -> str:
    """The sessions folder of one play: the runtime and quantization, and a tag telling plays of the same file apart."""
    return f"llamacpp-{quant}-{tag}" if tag else f"llamacpp-{quant}"


@app.function(image=llama_run_img, gpu=L4, volumes=VOLUMES, secrets=SECRETS, timeout=4 * HOURS)
def play(
    run: str, sheets: list[str], arms: list[str], seeds: int, quant: str, untie: bool, tag: str = ""
) -> dict[str, Any]:
    clock = JobClock(L4)
    data_volume.reload()
    store = _store(run, untie)
    gguf_path = Path(str(_remote(store, GGUF, f"{LABEL}-{quant}.gguf")))
    groups = [sheets[start : start + SHEETS_PER_SERVER] for start in range(0, len(sheets), SHEETS_PER_SERVER)]
    for group in groups:
        slots = seeds * len(group) * STREAMS_PER_SESSION
        with serving(gguf_path, context=SLOT_CONTEXT, parallel=slots) as url:
            for arm in arms:
                extra = ["--decider", _run_dir(run), "--decider-url", url] if arm == REIN_ARM else []
                _python(
                    "rein.sheets.e2e",
                    *_sheet_args(group),
                    "--arm",
                    arm,
                    "--runtime",
                    "llamacpp",
                    "--model",
                    url,
                    "--seeds",
                    str(seeds),
                    "--out",
                    str(_remote(store, SESSIONS, _sessions_dir(quant, tag))),
                    *extra,
                )
                data_volume.commit()
    return {"cost": clock.report()}


@app.function(image=torch_img, gpu=L4, volumes=VOLUMES, secrets=SECRETS, timeout=2 * HOURS)
def silence(run: str, data: str, limit: int, max_tokens: int, untie: bool) -> dict[str, Any]:
    """The soft tokens' output mass over held-out conversations rendered as the generator sees them."""
    clock = JobClock(L4)
    data_volume.reload()
    store = _store(run, untie)
    out = _remote(store, SILENCE_FILE)
    _python(
        "rein.eval.soft_silence",
        "--checkpoint",
        str(_remote(store, CHECKPOINT)),
        "--data",
        f"{DATA_MOUNT}/{data}",
        "--limit",
        str(limit),
        "--max-tokens",
        str(max_tokens),
        "--out",
        str(out),
    )
    data_volume.commit()
    report = json.loads(Path(str(out)).read_text(encoding="utf-8"))
    return {"cost": clock.report(), **{key: report[key] for key in report if key != "worst_positions"}}


@app.function(image=llama_run_img, gpu=L4, volumes=VOLUMES, secrets=SECRETS, timeout=HOURS)
def agree(run: str, records: int, quant: str, untie: bool) -> dict[str, Any]:
    """The decisions through llama-server on the GGUF against the run in PyTorch, both on this GPU."""
    clock = JobClock(L4)
    data_volume.reload()
    store = _store(run, untie)
    out = _remote(store, AGREEMENT_FILE.format(quant=quant))
    with serving(Path(str(_remote(store, GGUF, f"{LABEL}-{quant}.gguf"))), context=SLOT_CONTEXT) as url:
        _python(
            "rein.eval.soft_agreement",
            "--run",
            _run_dir(run),
            "--data",
            f"{DATA_MOUNT}/{TEST_DATA}",
            "--url",
            url,
            "--records",
            str(records),
            "--out",
            str(out),
        )
    data_volume.commit()
    report = json.loads(Path(str(out)).read_text(encoding="utf-8"))
    return {"cost": clock.report(), **report["summary"]}


@app.local_entrypoint()
def build(run: str, untie: bool = False) -> None:
    costs = {"merge": merge.remote(run, untie), "gguf": gguf.remote(run, untie)}
    print(json.dumps({"store": _store(run, untie), **costs}, indent=2))


@app.local_entrypoint()
def quantize(run: str, untie: bool = False) -> None:
    print(json.dumps({"store": _store(run, untie), "gguf": gguf.remote(run, untie)}, indent=2))


@app.local_entrypoint()
def check_silence(
    run: str,
    data: str = TEST_DATA,
    limit: int = SILENCE_LIMIT,
    max_tokens: int = SILENCE_TOKENS,
    untie: bool = False,
) -> None:
    print(json.dumps(silence.remote(run, data, limit, max_tokens, untie), indent=2))


@app.local_entrypoint()
def check_agreement(
    run: str, records: int = AGREEMENT_RECORDS, quant: str = DEFAULT_QUANT, untie: bool = False
) -> None:
    print(json.dumps(agree.remote(run, records, quant, untie), indent=2))


@app.local_entrypoint()
def play_sheets(
    run: str,
    sheets: str = DEFAULT_SHEETS,
    arms: str = f"{NONE_ARM},{REIN_ARM}",
    seeds: int = 3,
    quant: str = DEFAULT_QUANT,
    untie: bool = False,
    tag: str = "",
) -> None:
    names = [name for name in sheets.split(",") if name]
    cost = play.remote(run, names, [arm for arm in arms.split(",") if arm], seeds, quant, untie, tag)
    sessions = str(_remote(_store(run, untie), SESSIONS, _sessions_dir(quant, tag)))
    print(json.dumps({"cost": cost, "sessions": sessions}, indent=2))
