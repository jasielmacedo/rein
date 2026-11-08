"""Phase 2, stage 1A on Modal: an expanded Qwen3.5-4B checkpoint must run unchanged in every runtime."""

import json
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path
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
    mlx_image,
    sglang_image,
    torch_image,
    vllm_image,
)
from llamacpp import serving

from rein.neurons.constants import SIGNAL_TOKENS

APP_NAME = "rein-portability"
MODEL_ID = "Qwen/Qwen3.5-4B"
TINY = "tiny"
TINY_VOCAB = 248_320
TINY_SHARD = "200MB"
PHASE_DIR = f"{DATA_MOUNT}/phase2"
PROMPTS_PATH = "phase2/prompts/portability.jsonl"
CHECKPOINTS_FILE = "checkpoints.json"
ORIGINAL, ZERO, FORCE_TOOL_CALL, FORCE_SIGNAL = "original", "zero", "force-tool-call", "force-signal"
STATS_PROMPTS = 16
FORCED_PROMPTS = 8
GENERATION_TOKENS = 32
FORCED_TOKENS = 4
THROUGHPUT_TOKENS = 128
THROUGHPUT_REPEAT = 4
MLX_PROMPTS = 4
MLX_TOKENS = 8
QUANTS = ("Q8_0", "Q4_K_M")
LLAMA_PORT = 8080
LLAMA_CONTEXT = 8192
H100, L4, CPU = "H100", "L4", "CPU"
HOURS = 60 * 60
BUILD_MEMORY_MB = 64 * 1024
MLX_MEMORY_MB = 32 * 1024
MLX_CPUS = 8.0
FALLBACK_MARKER = "fallback"
OUTPUT_TAIL_CHARS = 4000
FORCED = {FORCE_TOOL_CALL: "<tool_call>", FORCE_SIGNAL: SIGNAL_TOKENS[0]}

app = modal.App(APP_NAME)
torch_img = torch_image()
vllm_img = vllm_image()
sglang_img = sglang_image()
llama_img = llamacpp_image()
mlx_img = mlx_image()


def _base(name: str) -> Path:
    return Path(PHASE_DIR) / "portability" / name


def _prompts() -> Path:
    return Path(DATA_MOUNT) / PROMPTS_PATH


def _checkpoints(name: str) -> dict[str, str]:
    return json.loads((_base(name) / CHECKPOINTS_FILE).read_text(encoding="utf-8"))


def _fresh(directory: Path) -> bool:
    """True when `directory` must be built: absent, or left half-written by a lost container."""
    if (directory / "config.json").exists():
        return False
    shutil.rmtree(directory, ignore_errors=True)
    return True


def _forces_another_token(directory: Path, token: str) -> bool:
    """True when a forcing checkpoint from an earlier run forces a different token than the one asked now."""
    manifest = directory / "rein_neurons.json"
    if not manifest.exists():
        return False
    roles = [neuron["role"] for neuron in json.loads(manifest.read_text(encoding="utf-8"))["neurons"]]
    return roles != [f"force:{token}"]


def _size(directory: str) -> int:
    return sum(path.stat().st_size for path in Path(directory).glob("*.safetensors"))


def _run(*args: str) -> str:
    """Runs a command from the repository and returns its output, printing the tail when it fails."""
    result = subprocess.run(list(args), cwd=REPO_MOUNT, capture_output=True, text=True)
    output = result.stdout + result.stderr
    if result.returncode:
        print(output[-OUTPUT_TAIL_CHARS:], flush=True)
        raise subprocess.CalledProcessError(result.returncode, list(args), output)
    return output


def _generate(runtime: str, model: str, out: Path, tokens: int, prompts: Path | None = None) -> dict[str, Any]:
    _run(
        "python",
        "-m",
        "rein.neurons.runtimes",
        "--runtime",
        runtime,
        "--model",
        model,
        "--prompts",
        str(prompts or _prompts()),
        "--out",
        str(out),
        "--max-tokens",
        str(tokens),
    )
    return json.loads(out.read_text(encoding="utf-8"))


def _subset(source: Path, target: Path, count: int, repeat: int = 1) -> Path:
    lines = source.read_text(encoding="utf-8").splitlines()[:count]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines * repeat) + "\n", encoding="utf-8")
    return target


def _versions(*modules: str) -> dict[str, str]:
    found = {}
    for module in modules:
        try:
            found[module] = (
                _run("python", "-c", f"import {module}; print({module}.__version__)").strip().splitlines()[-1]
            )
        except subprocess.CalledProcessError:
            found[module] = "unavailable"
    return found


def _forced_rates(tokenizer_dir: str, files: dict[str, Path]) -> dict[str, dict[str, float]]:
    """How often each forcing checkpoint opened its reply with its token, by id and as visible text."""
    from transformers import AutoTokenizer

    from rein.neurons.runtimes import forced_rate, load_run

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    return {
        label: forced_rate(
            load_run(path).generations, int(tokenizer.convert_tokens_to_ids(FORCED[label])), FORCED[label]
        )
        for label, path in files.items()
        if label in FORCED
    }


@app.function(image=torch_img, gpu=H100, volumes=VOLUMES, secrets=SECRETS, timeout=2 * HOURS, memory=BUILD_MEMORY_MB)
def build(name: str) -> dict[str, Any]:
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    from rein.model.backbone import load_text_backbone, placed
    from rein.neurons.checkpoint import expand_checkpoint, original_mismatches
    from rein.neurons.constants import NEURON_ALIGN
    from rein.neurons.portability import SIGNAL_TOKEN, TOOL_CALL_TOKEN, forcing_checkpoint, last_layer_stats
    from rein.neurons.runtimes import load_prompts
    from rein.neurons.tiny import tiny_qwen_checkpoint

    clock = JobClock(H100)
    base = _base(name)
    if name == TINY:
        source = base / ORIGINAL
        shutil.rmtree(base, ignore_errors=True)
        tokenizer_files = Path(snapshot_download(MODEL_ID, allow_patterns=["tokenizer*", "chat_template.jinja"]))
        tiny_qwen_checkpoint(source, TINY_VOCAB, TINY_SHARD, tokenizer_from=tokenizer_files)
    else:
        source = Path(snapshot_download(MODEL_ID))
    prompts = load_prompts(_prompts())[:STATS_PROMPTS]
    backbone = placed(load_text_backbone(str(source), dtype=torch.bfloat16), torch.device("cuda"))
    stats = last_layer_stats(backbone, [prompt.ids for prompt in prompts])
    embedding = backbone.embed_tokens.weight.detach().float().cpu()
    norm_weight = backbone.norm.weight.detach().float().cpu()
    layer = len(backbone.layers) - 1
    del backbone
    tokenizer = AutoTokenizer.from_pretrained(str(source))
    if _fresh(base / ZERO):
        expand_checkpoint(source, base / ZERO, NEURON_ALIGN)
    for label, token in ((FORCE_TOOL_CALL, TOOL_CALL_TOKEN), (FORCE_SIGNAL, SIGNAL_TOKEN)):
        if _forces_another_token(base / label, token):
            shutil.rmtree(base / label)
        if _fresh(base / label):
            token_id = int(tokenizer.convert_tokens_to_ids(token))
            forcing_checkpoint(source, base / label, token, token_id, embedding, norm_weight, stats, layer)
    mismatches = original_mismatches(source, base / ZERO)
    checkpoints = {ORIGINAL: str(source), ZERO: str(base / ZERO)}
    checkpoints |= {FORCE_TOOL_CALL: str(base / FORCE_TOOL_CALL), FORCE_SIGNAL: str(base / FORCE_SIGNAL)}
    (base / CHECKPOINTS_FILE).write_text(json.dumps(checkpoints, indent=2), encoding="utf-8")
    data_volume.commit()
    hf_volume.commit()
    return {
        "checkpoints": checkpoints,
        "original_tensors_changed": len(mismatches),
        "sizes_bytes": {label: _size(path) for label, path in checkpoints.items()},
        "last_layer_residual_norm": stats.residual_norm,
        "cost": clock.report(),
    }


@app.function(image=torch_img, gpu=H100, volumes=VOLUMES, secrets=SECRETS, timeout=2 * HOURS)
def transformers_check(name: str) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM

    from rein.neurons.portability import kl_stats
    from rein.neurons.runtimes import load_prompts, transformers_generate

    clock = JobClock(H100)
    checkpoints = _checkpoints(name)
    prompts = load_prompts(_prompts())

    def load(path: str, attention: str = "sdpa") -> torch.nn.Module:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation=attention)
        return model.to("cuda").eval()

    original, eager, zero = load(checkpoints[ORIGINAL]), load(checkpoints[ORIGINAL], "eager"), load(checkpoints[ZERO])
    sequences, starts, modes = [], [], []
    with torch.no_grad():
        for prompt in prompts:
            tokens = torch.tensor([list(prompt.ids)], device="cuda")
            sequences.append(original.generate(tokens, max_new_tokens=GENERATION_TOKENS, do_sample=False)[0].tolist())
            starts.append(len(prompt.ids))
            modes.append(prompt.prompt_id.rsplit("|", 1)[-1])
    report: dict[str, Any] = {"kl": {}}
    for mode in sorted(set(modes)):
        picked = [index for index, value in enumerate(modes) if value == mode]
        chosen = [sequences[index] for index in picked]
        offsets = [starts[index] for index in picked]
        report["kl"][mode] = {
            "zero_vs_original": kl_stats(original, zero, chosen, offsets),
            "noise_floor_sdpa_vs_eager": kl_stats(original, eager, chosen, offsets),
        }
    del original, eager, zero
    torch.cuda.empty_cache()
    forced = prompts[:FORCED_PROMPTS]
    outputs = _base(name) / "outputs"
    files = {}
    for label in FORCED:
        run = transformers_generate(checkpoints[label], forced, FORCED_TOKENS)
        files[label] = outputs / f"transformers-{label}.json"
        files[label].parent.mkdir(parents=True, exist_ok=True)
        payload = {"seconds": run.seconds, "generations": [asdict(item) for item in run.generations]}
        files[label].write_text(json.dumps(payload), encoding="utf-8")
    report["forced"] = _forced_rates(checkpoints[ORIGINAL], files)
    report["cost"] = clock.report()
    return report


def _runtime_check(runtime: str, name: str, labels: tuple[str, ...], gpu: str) -> dict[str, Any]:
    from rein.neurons.runtimes import compare, load_run

    clock = JobClock(gpu)
    checkpoints = _checkpoints(name)
    outputs = _base(name) / "outputs"
    report: dict[str, Any] = {"files": {}}
    for label in labels:
        tokens = GENERATION_TOKENS if label in (ORIGINAL, ZERO) else FORCED_TOKENS
        prompts = (
            _prompts() if label in (ORIGINAL, ZERO) else _subset(_prompts(), outputs / "forced.jsonl", FORCED_PROMPTS)
        )
        out = outputs / f"{runtime}-{label}.json"
        _generate(runtime, checkpoints[label], out, tokens, prompts)
        report["files"][label] = str(out)
    report["zero_vs_original"] = compare(
        load_run(Path(report["files"][ORIGINAL])).generations, load_run(Path(report["files"][ZERO])).generations
    )
    report["forced"] = _forced_rates(
        checkpoints[ORIGINAL], {label: Path(report["files"][label]) for label in labels if label in FORCED}
    )
    data_volume.commit()
    report["cost"] = clock.report()
    return report


def _throughput(runtime: str, name: str) -> dict[str, float]:
    checkpoints = _checkpoints(name)
    outputs = _base(name) / "outputs"
    batch = _subset(_prompts(), outputs / "throughput.jsonl", count=10**6, repeat=THROUGHPUT_REPEAT)
    speeds = {}
    for label in (ORIGINAL, ZERO):
        run = _generate(
            runtime, checkpoints[label], outputs / f"{runtime}-speed-{label}.json", THROUGHPUT_TOKENS, batch
        )
        speeds[label] = float(run["tokens_per_s"])
    return speeds


@app.function(image=vllm_img, gpu=H100, volumes=VOLUMES, secrets=SECRETS, timeout=2 * HOURS)
def vllm_check(name: str) -> dict[str, Any]:
    report = _runtime_check("vllm", name, (ORIGINAL, ZERO, FORCE_TOOL_CALL, FORCE_SIGNAL), H100)
    if name != TINY:
        report["tokens_per_s"] = _throughput("vllm", name)
    report["versions"] = _versions("vllm", "transformers")
    return report


@app.function(image=sglang_img, gpu=H100, volumes=VOLUMES, secrets=SECRETS, timeout=2 * HOURS)
def sglang_check(name: str) -> dict[str, Any]:
    report = _runtime_check("sglang", name, (ORIGINAL, ZERO, FORCE_SIGNAL), H100)
    report["versions"] = _versions("sglang", "transformers")
    return report


@app.function(image=mlx_img, cpu=MLX_CPUS, memory=MLX_MEMORY_MB, volumes=VOLUMES, secrets=SECRETS, timeout=3 * HOURS)
def mlx_check(name: str) -> dict[str, Any]:
    from rein.neurons.runtimes import compare, load_run

    clock = JobClock(CPU)
    checkpoints = _checkpoints(name)
    outputs = _base(name) / "outputs"
    prompts = _subset(_prompts(), outputs / "mlx.jsonl", MLX_PROMPTS)
    files = {}
    for label in (ORIGINAL, ZERO, FORCE_SIGNAL):
        files[label] = outputs / f"mlx-{label}.json"
        _generate("mlx", checkpoints[label], files[label], MLX_TOKENS, prompts)
    data_volume.commit()
    return {
        "zero_vs_original": compare(load_run(files[ORIGINAL]).generations, load_run(files[ZERO]).generations),
        "forced": _forced_rates(checkpoints[ORIGINAL], {FORCE_SIGNAL: files[FORCE_SIGNAL]}),
        "versions": _versions("mlx", "mlx_lm"),
        "cost": clock.report(),
    }


def _serve_and_generate(gguf: Path, out: Path, tokens: int, prompts: Path) -> dict[str, Any]:
    with serving(gguf, LLAMA_PORT, LLAMA_CONTEXT) as url:
        return _generate("llamacpp", url, out, tokens, prompts)


@app.function(image=llama_img, gpu=L4, volumes=VOLUMES, secrets=SECRETS, timeout=4 * HOURS)
def llamacpp_check(name: str) -> dict[str, Any]:
    from rein.neurons.runtimes import compare, load_run

    clock = JobClock(L4)
    checkpoints = _checkpoints(name)
    gguf_dir = _base(name) / "gguf"
    outputs = _base(name) / "outputs"
    gguf_dir.mkdir(parents=True, exist_ok=True)
    forced_prompts = _subset(_prompts(), outputs / "forced.jsonl", FORCED_PROMPTS)
    report: dict[str, Any] = {"quantize_fallback_lines": {}, "precisions": {}, "sizes_bytes": {}}
    for label, path in checkpoints.items():
        target = gguf_dir / f"{label}-bf16.gguf"
        if not target.exists():
            _run(
                "python", f"{LLAMA_CPP_DIR}/convert_hf_to_gguf.py", path, "--outtype", "bf16", "--outfile", str(target)
            )
        report["sizes_bytes"][target.name] = target.stat().st_size
    for label in (ORIGINAL, ZERO, FORCE_SIGNAL):
        for quant in QUANTS:
            target = gguf_dir / f"{label}-{quant}.gguf"
            log = _run(
                f"{LLAMA_CPP_DIR}/build/bin/llama-quantize", str(gguf_dir / f"{label}-bf16.gguf"), str(target), quant
            )
            report["quantize_fallback_lines"][target.name] = sum(FALLBACK_MARKER in line for line in log.splitlines())
            report["sizes_bytes"][target.name] = target.stat().st_size
    for precision in ("bf16", *QUANTS):
        runs = {}
        for label in (ORIGINAL, ZERO, FORCE_SIGNAL, FORCE_TOOL_CALL):
            gguf = gguf_dir / f"{label}-{precision}.gguf"
            if not gguf.exists():
                continue
            forced = label in (FORCE_SIGNAL, FORCE_TOOL_CALL)
            out = outputs / f"llamacpp-{precision}-{label}.json"
            runs[label] = _serve_and_generate(
                gguf, out, FORCED_TOKENS if forced else GENERATION_TOKENS, forced_prompts if forced else _prompts()
            )
            runs[label]["file"] = str(out)
        report["precisions"][precision] = {
            "zero_vs_original": compare(
                load_run(Path(runs[ORIGINAL]["file"])).generations, load_run(Path(runs[ZERO]["file"])).generations
            ),
            "tokens_per_s": {label: runs[label]["tokens_per_s"] for label in (ORIGINAL, ZERO)},
            "forced": _forced_rates(
                checkpoints[ORIGINAL], {label: Path(runs[label]["file"]) for label in FORCED if label in runs}
            ),
        }
    report["llama_cpp_commit"] = _run("git", "-C", LLAMA_CPP_DIR, "rev-parse", "HEAD").strip()
    shutil.rmtree(gguf_dir, ignore_errors=True)
    data_volume.commit()
    report["cost"] = clock.report()
    return report


@app.function(image=torch_img, volumes=VOLUMES, timeout=HOURS)
def cleanup(name: str) -> list[str]:
    """Removes the large checkpoints of a portability run and keeps its outputs."""
    removed = []
    for label in (ZERO, FORCE_TOOL_CALL, FORCE_SIGNAL, "gguf") + ((ORIGINAL,) if name == TINY else ()):
        target = _base(name) / label
        if target.exists():
            shutil.rmtree(target)
            removed.append(str(target))
    data_volume.commit()
    return removed


CHECKS = {
    "transformers": transformers_check,
    "vllm": vllm_check,
    "sglang": sglang_check,
    "llamacpp": llamacpp_check,
    "mlx": mlx_check,
}


@app.local_entrypoint()
def portability(name: str = TINY, checks: str = ",".join(CHECKS), keep: bool = False) -> None:
    local_prompts = Path("data") / PROMPTS_PATH
    with data_volume.batch_upload(force=True) as batch:
        batch.put_file(str(local_prompts), f"/{PROMPTS_PATH}")
    report: dict[str, Any] = {"name": name, "build": build.remote(name)}
    print(json.dumps(report["build"], indent=2), flush=True)
    calls = {check: CHECKS[check].spawn(name) for check in checks.split(",") if check}
    for check, call in calls.items():
        try:
            report[check] = call.get()
        except Exception as exc:
            report[check] = {"error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps({check: report[check]}, indent=2, default=str), flush=True)
    if not keep:
        report["removed"] = cleanup.remote(name)
    spent = sum(float(entry.get("cost", {}).get("usd", 0.0)) for entry in report.values() if isinstance(entry, dict))
    report["estimated_usd"] = round(spent, 2)
    out = Path("runs") / "phase2" / "portability" / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"estimated cost US${spent:.2f}; report in {out / 'report.json'}", flush=True)
