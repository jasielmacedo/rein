"""Images, volumes and price table shared by the phase 2 Modal apps."""

import time

import modal

PYTHON_VERSION = "3.12"
TORCH_VERSION = "2.13.0"
FLA_VERSION = "0.6.0"
CAUSAL_CONV1D_VERSION = "1.7.0"
CUDA_DEVEL_IMAGE = "nvidia/cuda:13.0.0-devel-ubuntu22.04"
VLLM_IMAGE = "vllm/vllm-openai:v0.27.1"
SGLANG_IMAGE = "lmsysorg/sglang:dev"
LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp.git"
LLAMA_CPP_TAG = "b8967"
LLAMA_CPP_DIR = "/opt/llama.cpp"
HOPPER_ARCH = "9.0"
LLAMA_CUDA_ARCHS = "89;90"
LLAMA_CMAKE_FLAGS = (
    f"-DGGML_CUDA=ON -DGGML_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES='{LLAMA_CUDA_ARCHS}' -DLLAMA_CURL=ON "
    "-DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined"
)
TRANSFORMERS_SPEC = "transformers>=5.17,<6"
CONV_BUILD_ENV = {
    "TORCH_CUDA_ARCH_LIST": HOPPER_ARCH,
    "CAUSAL_CONV1D_FORCE_BUILD": "TRUE",
    "MAX_JOBS": "8",
    "CC": "gcc",
    "CXX": "g++",
}

DATA_MOUNT = "/data"
HF_MOUNT = "/hf-cache"
REPO_MOUNT = "/root/rein"
HF_SECRET_NAME = "huggingface"

USD_PER_HOUR = {"H100": 3.95, "A100-80GB": 2.50, "L40S": 1.95, "L4": 0.80, "CPU": 0.20}
PRICED_GPUS = {"H100": "H100", "A100": "A100-80GB", "L40S": "L40S", "L4": "L4"}

data_volume = modal.Volume.from_name("rein-data", create_if_missing=True)
hf_volume = modal.Volume.from_name("rein-hf-cache", create_if_missing=True)
VOLUMES = {DATA_MOUNT: data_volume, HF_MOUNT: hf_volume}
SECRETS = [modal.Secret.from_name(HF_SECRET_NAME)]

RUNTIME_ENV = {
    "HF_HOME": HF_MOUNT,
    "TOKENIZERS_PARALLELISM": "false",
    "PYTHONPATH": f"{REPO_MOUNT}/src",
    "PYTHONUNBUFFERED": "1",
}
PROJECT_PACKAGES = (
    "pydantic>=2.9",
    "httpx>=0.28",
    "pyyaml>=6.0",
    "python-dotenv>=1.0",
    "huggingface_hub>=0.34",
    "safetensors>=0.5",
    "numpy",
)
LANGUAGE_DETECTOR = "lingua-language-detector>=2.0"
OLLAMA_INSTALLER = "https://ollama.com/install.sh"
JINJA_SPEC = "jinja2>=3.1"
PEFT_SPEC = "peft>=0.17"


def _with_project(image: modal.Image) -> modal.Image:
    return (
        image.env(RUNTIME_ENV)
        .add_local_dir("src", f"{REPO_MOUNT}/src")
        .add_local_dir("configs", f"{REPO_MOUNT}/configs")
        .add_local_python_source("images", "llamacpp", "ollama")
    )


def torch_image(extra: tuple[str, ...] = ()) -> modal.Image:
    """The training image of phase 1: torch 2.13 with the fast Gated DeltaNet kernels built for Hopper, plus any
    `extra` packages."""
    base = (
        modal.Image.from_registry(CUDA_DEVEL_IMAGE, add_python=PYTHON_VERSION)
        .apt_install("build-essential")
        .pip_install(
            f"torch=={TORCH_VERSION}",
            TRANSFORMERS_SPEC,
            "accelerate>=1.10",
            f"flash-linear-attention=={FLA_VERSION}",
            "ninja>=1.11",
            "packaging>=24",
            "pytest>=8.3",
            *PROJECT_PACKAGES,
            *extra,
        )
        .env(CONV_BUILD_ENV)
        .run_commands(f"pip install --no-build-isolation causal-conv1d=={CAUSAL_CONV1D_VERSION}")
    )
    return _with_project(base)


def vllm_image() -> modal.Image:
    base = (
        modal.Image.from_registry(VLLM_IMAGE)
        .entrypoint([])
        .run_commands("ln -sf $(which python3) /usr/bin/python")
        .pip_install(*PROJECT_PACKAGES, LANGUAGE_DETECTOR)
    )
    return _with_project(base)


def sglang_image() -> modal.Image:
    base = (
        modal.Image.from_registry(SGLANG_IMAGE)
        .entrypoint([])
        .run_commands("ln -sf $(which python3) /usr/bin/python")
        .pip_install(*PROJECT_PACKAGES)
    )
    return _with_project(base)


def _llamacpp_base() -> modal.Image:
    return (
        modal.Image.from_registry(CUDA_DEVEL_IMAGE, add_python=PYTHON_VERSION)
        .apt_install("build-essential", "cmake", "git", "libcurl4-openssl-dev")
        .run_commands(
            f"git clone --depth 1 --branch {LLAMA_CPP_TAG} {LLAMA_CPP_REPO} {LLAMA_CPP_DIR}",
            f"cd {LLAMA_CPP_DIR} && cmake -B build {LLAMA_CMAKE_FLAGS} "
            "&& cmake --build build --config Release -j 16 --target llama-server llama-quantize",
            f"pip install -r {LLAMA_CPP_DIR}/requirements/requirements-convert_hf_to_gguf.txt",
        )
        .pip_install(*PROJECT_PACKAGES)
    )


def llamacpp_image() -> modal.Image:
    """llama.cpp with its own converter's requirements, for converting, quantizing and serving."""
    return _with_project(_llamacpp_base())


def llamacpp_run_image() -> modal.Image:
    """llama.cpp plus a Transformers that renders Qwen3.5's chat template and loads a run's PyTorch decider next to
    the server, for driving end-to-end sessions and checking the GGUF's decisions against PyTorch."""
    return _with_project(_llamacpp_base().pip_install(TRANSFORMERS_SPEC, PEFT_SPEC))


def ollama_image() -> modal.Image:
    """Ollama (its own installer, latest release) plus a Transformers that renders Qwen3.5's chat template."""
    base = (
        modal.Image.debian_slim(python_version=PYTHON_VERSION)
        .apt_install("curl", "zstd", "ca-certificates")
        .run_commands(f"curl -fsSL {OLLAMA_INSTALLER} | sh")
        .pip_install(TRANSFORMERS_SPEC, JINJA_SPEC, *PROJECT_PACKAGES)
    )
    return _with_project(base)


def mlx_image() -> modal.Image:
    base = modal.Image.debian_slim(python_version=PYTHON_VERSION).pip_install(
        "mlx[cpu]", "mlx-lm", TRANSFORMERS_SPEC, *PROJECT_PACKAGES
    )
    return _with_project(base)


class JobClock:
    """Wall time and estimated cost of one Modal job, from the price table above."""

    def __init__(self, gpu: str) -> None:
        self.gpu = gpu
        self.started = time.monotonic()

    def report(self) -> dict[str, float | str]:
        seconds = time.monotonic() - self.started
        return {"gpu": self.gpu, "seconds": round(seconds, 1), "usd": round(seconds / 3600 * USD_PER_HOUR[self.gpu], 3)}


def running_gpu(default: str) -> str:
    """The price table's name for the GPU this container got, for functions that accept more than one."""
    import subprocess

    try:
        query = ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
        name = subprocess.run(query, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return default
    return next((label for key, label in PRICED_GPUS.items() if key in name), default)


VOLUME_COMMIT_EVERY_S = 120


def run_committing(args: list[str], volume: modal.Volume) -> None:
    """Runs a long job from the repository and commits `volume` every two minutes, so a lost container keeps its
    progress."""
    import subprocess

    process = subprocess.Popen(args, cwd=REPO_MOUNT)
    running = True
    while running:
        try:
            process.wait(timeout=VOLUME_COMMIT_EVERY_S)
            running = False
        except subprocess.TimeoutExpired:
            volume.commit()
    volume.commit()
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, args)
