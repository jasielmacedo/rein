"""Option A on Modal (after DR-027): a rule module of new decoder layers inside Qwen3.5-4B's own forward pass.

- `compatibility`: Qwen3.5-4B with a group of zero-output layers inserted (36 layers) must compute what Qwen computes,
  in Transformers (logits and greedy text), vLLM, llama.cpp (bf16, Q8_0, Q4_K_M) and Ollama.
- `decision`: test A2. Only the inserted layers are trained to decide per rule category (`rein.module.train`), on the
  views the generic detectors were scored on, and their states go through the detectors' own probe.
- `pairs`: Qwen3.5-4B's own replies at every non-thinking reply opening of the training and development views, for the
  module's action (`rein.module.pairs`); the auditor then judges them from the local machine.
- `action`: the module trained to act from the decision module (`rein.module.act`), installed into the checkpoint, then
  group 2 against Qwen and the checkpoint's own replies at the development reply openings, judged like Qwen's.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import modal
from images import (
    DATA_MOUNT,
    LLAMA_CPP_DIR,
    PEFT_SPEC,
    REPO_MOUNT,
    SECRETS,
    VOLUMES,
    JobClock,
    data_volume,
    hf_volume,
    llamacpp_image,
    ollama_image,
    run_committing,
    running_gpu,
    torch_image,
    vllm_image,
)
from llamacpp import CONTEXT, serving
from ollama import ollama_serving

APP_NAME = "rein-module"
MODEL_ID = "Qwen/Qwen3.5-4B"
RUNS = "phase2/runs/module"
PROMPTS_PATH = "phase2/prompts/portability.jsonl"
INSERT_AT = 20
ORIGINAL, INSERTED = "original", "inserted"
LABELS = (ORIGINAL, INSERTED)
GENERATION_TOKENS = 32
QUANTS = ("Q8_0", "Q4_K_M")
PRECISIONS = ("bf16", *QUANTS)
OLLAMA_QUANT = "Q4_K_M"
H100, L4, CPU = "H100", "L4", "CPU"
HOURS = 60 * 60
BUILD_CPUS = 8.0
BUILD_MEMORY_MB = 64 * 1024
OUTPUT_TAIL_CHARS = 4000
A100 = "A100-80GB"
LARGE_GPUS = [H100, A100]
OLLAMA = "ollama"
DATA_NAME = "mix-v1-audited"
CATEGORIES = "phase2/record_categories.json"
DECISION_VIEWS = "phase2/runs/full-block/views-reasoned.jsonl"
DECISION_RUN = "a2"
DECISION_EPOCHS = 2
TRAIN_REPORT = "train.json"
STATES_DIR = "states"
PROBE_DIR = "category-probe"
PAIRS_RUN = "pairs-v2"
PAIR_VIEWS = "phase2/views/module-v2.jsonl"
SAMPLES_FILE = "samples.jsonl"
PAIRED_FILES = ("pairs.jsonl", "anchors.jsonl")
ACT_RUN = "a3"
ACT_EPOCHS = 2
ACT_KL_WEIGHT = 1.0
ACT_DECISION_WEIGHT = 1.0
ACT_LR = 5e-5
ALLOCATOR = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
CHECKPOINT_DIR = "checkpoint"
BEHAVIOR_DIR = "behavior"
SAMENESS_DIR = "sameness-v2"
SAMENESS_VIEWS = "phase2/views/module-v2-replied.jsonl"
GENERAL_TEXT = "phase2/general/wikitext-2-test.txt"
DEVELOPMENT = "development"
LONG_RUN = "long"
LONG_VIEWS = "phase2/module/long/views-long.jsonl"
FILLER_CHECKED = "phase2/module/long/filler-checked.jsonl"
SYNTHETIC = "synthetic"
NEMOTRON_VIEWS = "phase2/module/nemotron/views.jsonl"
BEHAVIOR_SETS = {"rein": (PAIR_VIEWS, True), "long": (LONG_VIEWS, True), "nemotron": (NEMOTRON_VIEWS, False)}
TEACHER_CHECKPOINT = "runs/rein-4b-r1-L18"
TEACHER_LABELS = "phase2/module/teacher/teacher.jsonl"
OPENINGS_FILE = "openings.jsonl"
FIRST_VIEWS = "phase2/module/first/views-first.jsonl"
FIRST_RECORDS = "phase2/module/first/records"
RESTYLED_VIEWS = "phase2/module/restyled/views-restyled.jsonl"
OPENING_VIEWS = (PAIR_VIEWS, LONG_VIEWS, NEMOTRON_VIEWS, FIRST_VIEWS, RESTYLED_VIEWS)
ACT_OPENING_WEIGHT = 1.0
ACT_READ_LR_FACTOR = 1.0
CALIBRATION_VIEWS = "phase2/views/module-v2-calibration.jsonl"
GOLD_DIR = "phase2/gold-block"
GOLD_FILES = ("views.jsonl", "gold.jsonl", "categories.json")
GOLD_SPLIT = "gold"
GOLD_RUNS = "gold"
GOLD_TEACHER = "phase2/module/teacher/gold-teacher.jsonl"
QWEN = "qwen"

app = modal.App(APP_NAME)
torch_img = torch_image()
teacher_img = torch_image((PEFT_SPEC,))
vllm_img = vllm_image()
llama_img = llamacpp_image()
ollama_img = ollama_image()


def _base() -> Path:
    return Path(DATA_MOUNT) / RUNS


def _prompts() -> Path:
    return Path(DATA_MOUNT) / PROMPTS_PATH


def _snapshot() -> str:
    from huggingface_hub import snapshot_download

    return str(snapshot_download(MODEL_ID))


def _checkpoint(label: str) -> str:
    return _snapshot() if label == ORIGINAL else str(_base() / INSERTED)


def _run(*args: str) -> str:
    result = subprocess.run(list(args), cwd=REPO_MOUNT, capture_output=True, text=True)
    output = result.stdout + result.stderr
    if result.returncode:
        print(output[-OUTPUT_TAIL_CHARS:], flush=True)
        raise subprocess.CalledProcessError(result.returncode, list(args), output)
    return output


def _generate(runtime: str, model: str, out: Path) -> Path:
    _run(
        "python",
        "-m",
        "rein.neurons.runtimes",
        "--runtime",
        runtime,
        "--model",
        model,
        "--prompts",
        str(_prompts()),
        "--out",
        str(out),
        "--max-tokens",
        str(GENERATION_TOKENS),
    )
    return out


def _compare(files: dict[str, Path]) -> dict[str, float]:
    from rein.neurons.runtimes import compare, load_run

    return compare(load_run(files[ORIGINAL]).generations, load_run(files[INSERTED]).generations)


@app.function(image=torch_img, cpu=BUILD_CPUS, memory=BUILD_MEMORY_MB, volumes=VOLUMES, secrets=SECRETS, timeout=HOURS)
def build(at: int) -> dict[str, Any]:
    from rein.neurons.layers import insert_layers

    clock = JobClock(CPU)
    target = _base() / INSERTED
    shutil.rmtree(target, ignore_errors=True)
    insertion = insert_layers(Path(_snapshot()), target, at)
    data_volume.commit()
    hf_volume.commit()
    return {"layers": insertion.layers_after, "inserted": list(insertion.new_layers), "cost": clock.report()}


@app.function(image=torch_img, gpu=LARGE_GPUS, volumes=VOLUMES, secrets=SECRETS, timeout=2 * HOURS)
def transformers_check() -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM

    from rein.neurons.portability import kl_stats
    from rein.neurons.runtimes import load_prompts

    clock = JobClock(running_gpu(H100))
    data_volume.reload()

    def load(label: str) -> Any:
        return AutoModelForCausalLM.from_pretrained(_checkpoint(label), dtype=torch.bfloat16).to("cuda").eval()

    original, inserted = load(ORIGINAL), load(INSERTED)
    prompts = load_prompts(_prompts())
    identical_logits, sequences, starts = 0, [], []
    with torch.no_grad():
        for prompt in prompts:
            tokens = torch.tensor([list(prompt.ids)], device="cuda")
            identical_logits += int(torch.equal(original(input_ids=tokens).logits, inserted(input_ids=tokens).logits))
            sequences.append(original.generate(tokens, max_new_tokens=GENERATION_TOKENS, do_sample=False)[0].tolist())
            starts.append(len(prompt.ids))
    report = {
        "prompts": len(prompts),
        "identical_prompt_logits": identical_logits / max(1, len(prompts)),
        "kl_inserted_vs_original": kl_stats(original, inserted, sequences, starts),
    }
    del original, inserted
    torch.cuda.empty_cache()
    outputs = _base() / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    files = {
        label: _generate("transformers", _checkpoint(label), outputs / f"transformers-{label}.json") for label in LABELS
    }
    report["greedy"] = _compare(files)
    data_volume.commit()
    report["cost"] = clock.report()
    return report


@app.function(image=vllm_img, gpu=H100, volumes=VOLUMES, secrets=SECRETS, timeout=2 * HOURS)
def vllm_check() -> dict[str, Any]:
    clock = JobClock(H100)
    data_volume.reload()
    outputs = _base() / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    files = {label: _generate("vllm", _checkpoint(label), outputs / f"vllm-{label}.json") for label in LABELS}
    data_volume.commit()
    return {"greedy": _compare(files), "cost": clock.report()}


@app.function(image=llama_img, gpu=L4, volumes=VOLUMES, secrets=SECRETS, timeout=4 * HOURS)
def llamacpp_check() -> dict[str, Any]:
    clock = JobClock(L4)
    data_volume.reload()
    gguf = _base() / "gguf"
    outputs = _base() / "outputs"
    gguf.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"precisions": {}, "sizes_bytes": {}}
    for label in LABELS:
        bf16 = gguf / f"{label}-bf16.gguf"
        _run(
            "python",
            f"{LLAMA_CPP_DIR}/convert_hf_to_gguf.py",
            _checkpoint(label),
            "--outtype",
            "bf16",
            "--outfile",
            str(bf16),
        )
        for quant in QUANTS:
            _run(f"{LLAMA_CPP_DIR}/build/bin/llama-quantize", str(bf16), str(gguf / f"{label}-{quant}.gguf"), quant)
    for path in sorted(gguf.glob("*.gguf")):
        report["sizes_bytes"][path.name] = path.stat().st_size
    for precision in PRECISIONS:
        files = {}
        for label in LABELS:
            with serving(gguf / f"{label}-{precision}.gguf") as url:
                files[label] = _generate("llamacpp", url, outputs / f"llamacpp-{precision}-{label}.json")
        report["precisions"][precision] = _compare(files)
    for label in LABELS:
        (gguf / f"{label}-bf16.gguf").unlink()
    data_volume.commit()
    report["cost"] = clock.report()
    return report


@app.function(image=ollama_img, gpu=L4, volumes=VOLUMES, secrets=SECRETS, timeout=2 * HOURS)
def ollama_check() -> dict[str, Any]:
    clock = JobClock(L4)
    data_volume.reload()
    outputs = _base() / "outputs"
    files = {}
    for label in LABELS:
        with ollama_serving(_base() / "gguf" / f"{label}-{OLLAMA_QUANT}.gguf", CONTEXT, 1) as url:
            files[label] = _generate(OLLAMA, url, outputs / f"ollama-{OLLAMA_QUANT}-{label}.json")
    data_volume.commit()
    return {"greedy": _compare(files), "quant": OLLAMA_QUANT, "cost": clock.report()}


@app.function(image=torch_img, gpu=LARGE_GPUS, volumes=VOLUMES, secrets=SECRETS, timeout=8 * HOURS, retries=2)
def decide(run: str, epochs: int) -> dict[str, Any]:
    """Trains the inserted layers to decide, stores their states and scores them with the generic detectors' probe; a
    restarted container resumes from the last saved step."""
    clock = JobClock(running_gpu(H100))
    data_volume.reload()
    out = _base() / run
    data = f"{DATA_MOUNT}/{DATA_NAME}"
    categories = f"{DATA_MOUNT}/{CATEGORIES}"
    train = ["python", "-m", "rein.module.train", "--checkpoint", str(_base() / INSERTED)]
    train += ["--views", f"{DATA_MOUNT}/{DECISION_VIEWS}", "--data", data, "--categories", categories]
    run_committing([*train, "--out", str(out), "--epochs", str(epochs)], data_volume)
    read_layer = json.loads((out / TRAIN_REPORT).read_text(encoding="utf-8"))["read_layer"]
    probe = ["python", "-m", "rein.compiler.category_probe", "probe", "--states", str(out / STATES_DIR)]
    probe += ["--data", data, "--categories", categories, "--layer", str(read_layer), "--out", str(out / PROBE_DIR)]
    run_committing(probe, data_volume)
    return {"cost": clock.report()}


@app.function(image=vllm_img, gpu=LARGE_GPUS, volumes=VOLUMES, secrets=SECRETS, timeout=4 * HOURS)
def sample_pairs(run: str, views: str, labelled: bool) -> dict[str, Any]:
    """Qwen's replies at the views' reply openings; `labelled` keeps only Rein records with category labels."""
    clock = JobClock(running_gpu(H100))
    data_volume.reload()
    sample = ["python", "-m", "rein.module.pairs", "sample", "--views", f"{DATA_MOUNT}/{views}", "--model", MODEL_ID]
    if labelled:
        sample += ["--data", f"{DATA_MOUNT}/{DATA_NAME}", "--categories", f"{DATA_MOUNT}/{CATEGORIES}"]
    run_committing([*sample, "--out", str(_base() / run / SAMPLES_FILE)], data_volume)
    hf_volume.commit()
    return {"cost": clock.report()}


@app.function(image=torch_img, gpu=LARGE_GPUS, volumes=VOLUMES, secrets=SECRETS, timeout=8 * HOURS, retries=2)
def act(
    run: str,
    start: str,
    pairs: list[str],
    epochs: int,
    kl_weight: float,
    decision_weight: float,
    lr: float,
    taught: bool,
    read_inside: bool,
    openings: str,
    opening_weight: float,
    read_lr_factor: float,
) -> dict[str, Any]:
    """Trains the module to act, from zero output or from the module of `start`, and writes the checkpoint with it;
    `taught` puts Rein-r1's soft labels on the decision read and scores the module's own decisions beside Rein-r1's,
    and `openings` adds the action at every training reply opening with a target."""
    from rein.module.train import MODULE_FILE

    clock = JobClock(running_gpu(H100))
    data_volume.reload()
    os.environ.update(ALLOCATOR)
    out = _base() / run
    train = ["python", "-m", "rein.module.act", "--checkpoint", str(_base() / INSERTED)]
    train += ["--start", str(_base() / start)] if start else []
    train += ["--pairs", *(str(_base() / folder) for folder in pairs), "--data", f"{DATA_MOUNT}/{DATA_NAME}"]
    train += ["--categories", f"{DATA_MOUNT}/{CATEGORIES}", "--out", str(out), "--epochs", str(epochs)]
    train += ["--kl-weight", str(kl_weight), "--decision-weight", str(decision_weight), "--lr", str(lr)]
    train += ["--teacher", f"{DATA_MOUNT}/{TEACHER_LABELS}"] if taught else []
    train += ["--read-inside"] if read_inside else []
    train += ["--read-lr-factor", str(read_lr_factor)]
    if openings:
        train += ["--openings", str(_base() / openings / OPENINGS_FILE), "--opening-weight", str(opening_weight)]
        present = [f"{DATA_MOUNT}/{views}" for views in OPENING_VIEWS if Path(f"{DATA_MOUNT}/{views}").exists()]
        train += ["--opening-views", *present]
    run_committing(train, data_volume)
    target = out / CHECKPOINT_DIR
    shutil.rmtree(target, ignore_errors=True)
    install = ["python", "-m", "rein.module.install", "--source", str(_base() / INSERTED)]
    run_committing([*install, "--weights", str(out / MODULE_FILE), "--target", str(target)], data_volume)
    if taught:
        score = ["python", "-m", "rein.module.decisions", "--checkpoint", str(target), "--run", str(out), "--views"]
        score += [f"{DATA_MOUNT}/{PAIR_VIEWS}", f"{DATA_MOUNT}/{LONG_VIEWS}", "--data", f"{DATA_MOUNT}/{DATA_NAME}"]
        score += ["--categories", f"{DATA_MOUNT}/{CATEGORIES}", "--teacher", f"{DATA_MOUNT}/{TEACHER_LABELS}"]
        score += ["--calibration-views", f"{DATA_MOUNT}/{CALIBRATION_VIEWS}"]
        run_committing(score, data_volume)
    return {"cost": clock.report()}


@app.function(image=torch_img, gpu=LARGE_GPUS, volumes=VOLUMES, secrets=SECRETS, timeout=2 * HOURS)
def module_sameness(run: str) -> dict[str, Any]:
    """Group 2: KL to Qwen on quiet development conversations and on general text, and both perplexities."""
    clock = JobClock(running_gpu(H100))
    data_volume.reload()
    check = ["python", "-m", "rein.compiler.sameness", "--original", _snapshot()]
    check += ["--checkpoint", str(_base() / run / CHECKPOINT_DIR), "--views", f"{DATA_MOUNT}/{SAMENESS_VIEWS}"]
    check += ["--data", f"{DATA_MOUNT}/{DATA_NAME}", "--categories", f"{DATA_MOUNT}/{CATEGORIES}"]
    check += ["--text", f"{DATA_MOUNT}/{GENERAL_TEXT}", "--out", str(_base() / run / SAMENESS_DIR)]
    run_committing(check, data_volume)
    return {"cost": clock.report()}


@app.function(image=vllm_img, gpu=LARGE_GPUS, volumes=VOLUMES, secrets=SECRETS, timeout=4 * HOURS)
def module_behavior(run: str, sets: list[str]) -> dict[str, Any]:
    """The checkpoint's own replies at the development reply openings of each set, drawn as Qwen's were."""
    clock = JobClock(running_gpu(H100))
    data_volume.reload()
    for name in sets:
        views, labelled = BEHAVIOR_SETS[name]
        sample = ["python", "-m", "rein.module.pairs", "sample", "--views", f"{DATA_MOUNT}/{views}", "--model"]
        sample += [str(_base() / run / CHECKPOINT_DIR), "--splits", DEVELOPMENT]
        if labelled:
            sample += ["--data", f"{DATA_MOUNT}/{DATA_NAME}", "--categories", f"{DATA_MOUNT}/{CATEGORIES}"]
        run_committing([*sample, "--out", str(_base() / run / BEHAVIOR_DIR / name / SAMPLES_FILE)], data_volume)
    return {"cost": clock.report()}


@app.function(image=vllm_img, gpu=LARGE_GPUS, volumes=VOLUMES, secrets=SECRETS, timeout=4 * HOURS)
def gold_replies(models: list[str]) -> dict[str, Any]:
    """Each model's replies at the reply openings of the user's hand-written gold set (`qwen` is Qwen itself, any
    other name an action run's checkpoint), drawn as the development replies are."""
    clock = JobClock(running_gpu(H100))
    data_volume.reload()
    for name in models:
        model = MODEL_ID if name == QWEN else str(_base() / name / CHECKPOINT_DIR)
        sample = ["python", "-m", "rein.module.pairs", "sample", "--views", f"{DATA_MOUNT}/{GOLD_DIR}/{GOLD_FILES[0]}"]
        sample += ["--model", model, "--splits", GOLD_SPLIT, "--data", f"{DATA_MOUNT}/{GOLD_DIR}"]
        sample += ["--categories", f"{DATA_MOUNT}/{GOLD_DIR}/{GOLD_FILES[2]}"]
        run_committing([*sample, "--out", str(_base() / GOLD_RUNS / name / SAMPLES_FILE)], data_volume)
    hf_volume.commit()
    return {"cost": clock.report()}


@app.function(image=torch_img, gpu=LARGE_GPUS, volumes=VOLUMES, secrets=SECRETS, timeout=2 * HOURS)
def gold_decisions(runs: list[str]) -> dict[str, Any]:
    """Each run's decision read on the gold set's reply openings, at the temperature its development scoring fitted,
    beside Rein-r1's labels of them."""
    from rein.module.decisions import DECISIONS_FILE

    clock = JobClock(running_gpu(H100))
    data_volume.reload()
    for run in runs:
        folder = _base() / run
        temperature = json.loads((folder / DECISIONS_FILE).read_text(encoding="utf-8"))["temperature"]
        score = ["python", "-m", "rein.module.decisions", "--checkpoint", str(folder / CHECKPOINT_DIR)]
        score += ["--run", str(folder), "--views", f"{DATA_MOUNT}/{GOLD_DIR}/{GOLD_FILES[0]}", "--split", GOLD_SPLIT]
        score += ["--data", f"{DATA_MOUNT}/{GOLD_DIR}", "--categories", f"{DATA_MOUNT}/{GOLD_DIR}/{GOLD_FILES[2]}"]
        score += ["--teacher", f"{DATA_MOUNT}/{GOLD_TEACHER}", "--temperature", str(temperature)]
        run_committing([*score, "--out", str(_base() / GOLD_RUNS / run)], data_volume)
    return {"cost": clock.report()}


@app.function(image=vllm_img, gpu=LARGE_GPUS, volumes=VOLUMES, secrets=SECRETS, timeout=6 * HOURS)
def long_play() -> dict[str, Any]:
    """Qwen answers the checked filler of every long conversation, then the long views are written."""
    clock = JobClock(running_gpu(H100))
    data_volume.reload()
    play = ["python", "-m", "rein.module.longviews", "play", "--data", f"{DATA_MOUNT}/{DATA_NAME}"]
    play += ["--synthetic", f"{DATA_MOUNT}/{SYNTHETIC}", "--filler", f"{DATA_MOUNT}/{FILLER_CHECKED}"]
    run_committing([*play, "--model", MODEL_ID, "--out", f"{DATA_MOUNT}/{LONG_VIEWS}"], data_volume)
    hf_volume.commit()
    return {"cost": clock.report()}


def _pull(remote: str, local: Path) -> None:
    local.parent.mkdir(parents=True, exist_ok=True)
    with local.open("wb") as handle:
        for chunk in data_volume.read_file(remote):
            handle.write(chunk)


@app.local_entrypoint()
def decision(epochs: int = DECISION_EPOCHS, run: str = DECISION_RUN) -> None:
    """Test A2: can the inserted layers decide like Rein-r1? Pulls the reports to runs/ and the weights to
    runs/checkpoints/."""
    from rein.compiler.category_probe import PROBABILITIES_FILE
    from rein.compiler.category_probe import REPORT_FILE as PROBE_REPORT
    from rein.module.train import HEAD_FILE, MODULE_FILE

    with data_volume.batch_upload(force=True) as batch:
        batch.put_file(str(Path("data") / CATEGORIES), f"/{CATEGORIES}")
    result = decide.remote(run, epochs)
    remote = f"/{RUNS}/{run}"
    local = Path("runs") / "phase2" / "module" / run
    for name in (TRAIN_REPORT, f"{PROBE_DIR}/{PROBE_REPORT}", f"{PROBE_DIR}/{PROBABILITIES_FILE}"):
        _pull(f"{remote}/{name}", local / name)
    for name in (MODULE_FILE, HEAD_FILE):
        _pull(f"{remote}/{name}", Path("runs") / "checkpoints" / "phase2" / f"module-{run}" / name)
    print(json.dumps({"cost": result["cost"], "pulled": str(local)}), flush=True)


@app.local_entrypoint()
def pairs(run: str = PAIRS_RUN, views: str = PAIR_VIEWS, labelled: bool = True) -> None:
    """Qwen's own replies for the module's action, on views under data/ rendered with the checkpoint's rules template,
    pulled to data/phase2/module/ for the judge."""
    with data_volume.batch_upload(force=True) as batch:
        batch.put_file(str(Path("data") / views), f"/{views}")
    result = sample_pairs.remote(run, views, labelled)
    local = Path("data") / "phase2" / "module" / run / SAMPLES_FILE
    _pull(f"/{RUNS}/{run}/{SAMPLES_FILE}", local)
    print(json.dumps({"cost": result["cost"], "pulled": str(local)}), flush=True)


@app.function(image=teacher_img, gpu=LARGE_GPUS, volumes=VOLUMES, secrets=SECRETS, timeout=6 * HOURS, retries=2)
def teacher_labels(views: list[str], data: str, categories: str, out: str) -> dict[str, Any]:
    """Rein-r1's soft labels per category at every view's reply opening, read from the records of `data`; appended as
    it goes, so a restart resumes."""
    clock = JobClock(running_gpu(H100))
    data_volume.reload()
    label = ["python", "-m", "rein.module.teacher", "--teacher", f"{DATA_MOUNT}/{TEACHER_CHECKPOINT}", "--views"]
    label += [f"{DATA_MOUNT}/{views_file}" for views_file in views]
    label += ["--filler", f"{DATA_MOUNT}/{FILLER_CHECKED}", "--data", f"{DATA_MOUNT}/{data}"]
    label += ["--categories", f"{DATA_MOUNT}/{categories}", "--out", f"{DATA_MOUNT}/{out}"]
    run_committing(label, data_volume)
    hf_volume.commit()
    return {"cost": clock.report()}


@app.local_entrypoint()
def teach(
    views: str = f"{PAIR_VIEWS},{LONG_VIEWS}",
    data: str = DATA_NAME,
    categories: str = CATEGORIES,
    out: str = TEACHER_LABELS,
) -> None:
    """Rein-r1 labels the views (the short Rein views and the long ones by default) on the records of `data` (the
    first-turn cuts live in their own folder); the views and a records folder of our own are uploaded first, and the
    labels are pulled to data/."""
    chosen = [name for name in views.split(",") if name]
    with data_volume.batch_upload(force=True) as batch:
        for views_file in chosen:
            batch.put_file(str(Path("data") / views_file), f"/{views_file}")
        if data != DATA_NAME:
            batch.put_directory(str(Path("data") / data), f"/{data}")
    result = teacher_labels.remote(chosen, data, categories, out)
    _pull(f"/{out}", Path("data") / out)
    print(json.dumps({"cost": result["cost"], "pulled": str(Path("data") / out)}), flush=True)


@app.local_entrypoint()
def gold(models: str = f"{QWEN},{ACT_RUN}") -> None:
    """The gold set's replies of each model, pulled to data/ for the local judge (`rein.module.pairs judge` with
    `--gold-project`)."""
    chosen = [name for name in models.split(",") if name]
    with data_volume.batch_upload(force=True) as batch:
        for name in GOLD_FILES:
            batch.put_file(str(Path("data") / GOLD_DIR / name), f"/{GOLD_DIR}/{name}")
    result = gold_replies.remote(chosen)
    for name in chosen:
        local = Path("data") / "phase2" / "module" / GOLD_RUNS / name / SAMPLES_FILE
        _pull(f"/{RUNS}/{GOLD_RUNS}/{name}/{SAMPLES_FILE}", local)
    print(json.dumps({"cost": result["cost"], "models": chosen}), flush=True)


@app.local_entrypoint()
def gold_read(runs: str = ACT_RUN) -> None:
    """Rein-r1 labels the gold set's reply openings, then each run's decision read is scored on them; the labels go to
    data/, the scores to runs/."""
    from rein.module.decisions import DECISIONS_FILE, FIRE_FILE

    chosen = [name for name in runs.split(",") if name]
    with data_volume.batch_upload(force=True) as batch:
        for name in GOLD_FILES:
            batch.put_file(str(Path("data") / GOLD_DIR / name), f"/{GOLD_DIR}/{name}")
    gold_views = f"{GOLD_DIR}/{GOLD_FILES[0]}"
    taught = teacher_labels.remote([gold_views], GOLD_DIR, f"{GOLD_DIR}/{GOLD_FILES[2]}", GOLD_TEACHER)
    _pull(f"/{GOLD_TEACHER}", Path("data") / GOLD_TEACHER)
    scored = gold_decisions.remote(chosen)
    for run in chosen:
        for name in (DECISIONS_FILE, FIRE_FILE):
            _pull(f"/{RUNS}/{GOLD_RUNS}/{run}/{name}", Path("runs") / "phase2" / "module" / GOLD_RUNS / run / name)
    print(json.dumps({"costs": {"teacher": taught["cost"], "decisions": scored["cost"]}}), flush=True)


@app.local_entrypoint()
def long_views() -> None:
    """Uploads the checked filler and the synthetic scenarios, plays the long conversations and pulls the views."""
    from rein.genview import constants as gc

    with data_volume.batch_upload(force=True) as batch:
        batch.put_file(str(Path("data") / FILLER_CHECKED), f"/{FILLER_CHECKED}")
        for scenarios in sorted((Path("data") / SYNTHETIC).glob(gc.SCENARIO_GLOB)):
            batch.put_file(str(scenarios), f"/{scenarios.relative_to(Path('data')).as_posix()}")
    result = long_play.remote()
    _pull(f"/{LONG_VIEWS}", Path("data") / LONG_VIEWS)
    print(json.dumps({"cost": result["cost"], "pulled": str(Path("data") / LONG_VIEWS)}), flush=True)


@app.local_entrypoint()
def action(
    run: str = ACT_RUN,
    start: str = "",
    pairs: str = PAIRS_RUN,
    sets: str = "rein",
    epochs: int = ACT_EPOCHS,
    kl_weight: float = ACT_KL_WEIGHT,
    decision_weight: float = ACT_DECISION_WEIGHT,
    lr: float = ACT_LR,
    taught: bool = False,
    read_inside: bool = False,
    openings: str = "",
    opening_weight: float = ACT_OPENING_WEIGHT,
    read_lr_factor: float = ACT_READ_LR_FACTOR,
) -> None:
    """Uploads the pairs and openings built locally, trains the module to act, then measures group 2 and draws the
    checkpoint's development replies for the auditor; reports go to runs/, the weights to runs/checkpoints/."""
    from rein.compiler.sameness import REPORT_FILE as SAMENESS_REPORT
    from rein.module.act import REPORT_FILE as ACT_REPORT
    from rein.module.decisions import DECISIONS_FILE, FIRE_FILE
    from rein.module.train import HEAD_FILE, MODULE_FILE

    folders = [name for name in pairs.split(",") if name]
    chosen = [name for name in sets.split(",") if name]
    with data_volume.batch_upload(force=True) as batch:
        for folder in folders:
            for name in PAIRED_FILES:
                local = Path("data") / "phase2" / "module" / folder / name
                batch.put_file(str(local), f"/{RUNS}/{folder}/{name}")
        batch.put_file(str(Path("data") / SAMENESS_VIEWS), f"/{SAMENESS_VIEWS}")
        shared = {BEHAVIOR_SETS[name][0] for name in chosen}
        shared |= set(OPENING_VIEWS) if openings else set()
        shared |= {CALIBRATION_VIEWS} if taught else set()
        for views in sorted(shared):
            if (Path("data") / views).exists():
                batch.put_file(str(Path("data") / views), f"/{views}")
        if openings:
            local = Path("data") / "phase2" / "module" / openings / OPENINGS_FILE
            batch.put_file(str(local), f"/{RUNS}/{openings}/{OPENINGS_FILE}")
    settings = (epochs, kl_weight, decision_weight, lr, taught, read_inside, openings, opening_weight, read_lr_factor)
    trained = act.remote(run, start, folders, *settings)
    costs = {"act": trained}
    calls = {"sameness": module_sameness.spawn(run), "behavior": module_behavior.spawn(run, chosen)}
    costs |= {name: call.get() for name, call in calls.items()}
    remote = f"/{RUNS}/{run}"
    local = Path("runs") / "phase2" / "module" / run
    _pull(f"{remote}/{ACT_REPORT}", local / ACT_REPORT)
    for name in (DECISIONS_FILE, FIRE_FILE) if taught else ():
        _pull(f"{remote}/{name}", local / name)
    _pull(f"{remote}/{SAMENESS_DIR}/{SAMENESS_REPORT}", local / SAMENESS_DIR / SAMENESS_REPORT)
    for name in chosen:
        local_samples = Path("data") / "phase2" / "module" / run / name / SAMPLES_FILE
        _pull(f"{remote}/{BEHAVIOR_DIR}/{name}/{SAMPLES_FILE}", local_samples)
    for name in (MODULE_FILE, HEAD_FILE):
        _pull(f"{remote}/{name}", Path("runs") / "checkpoints" / "phase2" / f"module-{run}" / name)
    spent = round(sum(float(entry["cost"]["usd"]) for entry in costs.values()), 2)
    print(json.dumps({"costs": costs, "estimated_usd": spent}), flush=True)


@app.local_entrypoint()
def measure(run: str = ACT_RUN) -> None:
    """Group 2 again for a trained run, on the views rendered with the checkpoint's rules template."""
    from rein.compiler.sameness import REPORT_FILE as SAMENESS_REPORT

    with data_volume.batch_upload(force=True) as batch:
        batch.put_file(str(Path("data") / SAMENESS_VIEWS), f"/{SAMENESS_VIEWS}")
    cost = module_sameness.remote(run)
    local = Path("runs") / "phase2" / "module" / run / SAMENESS_DIR / SAMENESS_REPORT
    _pull(f"/{RUNS}/{run}/{SAMENESS_DIR}/{SAMENESS_REPORT}", local)
    print(json.dumps({"cost": cost["cost"], "report": str(local)}), flush=True)


def _spent(report: dict[str, Any]) -> float:
    return round(
        sum(float(entry.get("cost", {}).get("usd", 0.0)) for entry in report.values() if isinstance(entry, dict)), 2
    )


@app.local_entrypoint()
def compatibility(at: int = INSERT_AT, only: str = "") -> None:
    """Builds the 36-layer checkpoint and checks it against Qwen3.5-4B in every runtime. `only` (comma-separated
    checks, ollama included) reruns some of them on the stored checkpoint and merges them into the saved report."""
    local = Path("runs") / "phase2" / "module" / "compatibility.json"
    picked = {name for name in only.split(",") if name}
    report: dict[str, Any] = json.loads(local.read_text(encoding="utf-8")) if picked and local.exists() else {}
    with data_volume.batch_upload(force=True) as batch:
        batch.put_file(str(Path("data") / PROMPTS_PATH), f"/{PROMPTS_PATH}")
    if not picked:
        report["build"] = build.remote(at)
        print(json.dumps(report["build"]), flush=True)
    checks = {"transformers": transformers_check, "vllm": vllm_check, "llamacpp": llamacpp_check}
    calls = {name: check.spawn() for name, check in checks.items() if not picked or name in picked}
    for name, call in calls.items():
        try:
            report[name] = call.get()
        except Exception as exc:
            report[name] = {"error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps({name: report[name]}, default=str), flush=True)
    if (not picked or OLLAMA in picked) and "error" not in report.get("llamacpp", {"error": "not run"}):
        try:
            report[OLLAMA] = ollama_check.remote()
        except Exception as exc:
            report[OLLAMA] = {"error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps({OLLAMA: report[OLLAMA]}, default=str), flush=True)
    report["estimated_usd"] = _spent(report)
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"report": str(local), "estimated_usd": report["estimated_usd"]}), flush=True)
