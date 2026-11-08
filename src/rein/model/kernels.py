import importlib
from dataclasses import dataclass

FAST_KERNELS = {
    "chunk_gated_delta_rule": ("fla.ops.gated_delta_rule", "chunk_gated_delta_rule"),
    "fused_recurrent_gated_delta_rule": ("fla.ops.gated_delta_rule", "fused_recurrent_gated_delta_rule"),
    "causal_conv1d_fn": ("causal_conv1d", "causal_conv1d_fn"),
    "causal_conv1d_update": ("causal_conv1d", "causal_conv1d_update"),
}


@dataclass(frozen=True)
class KernelStatus:
    name: str
    available: bool
    error: str | None


class FastKernelsMissing(RuntimeError):
    pass


def _probe(name: str, module_name: str, attribute: str) -> KernelStatus:
    """Imports the kernel the way transformers does, but keeps the error transformers would swallow."""
    try:
        module = importlib.import_module(module_name)
        getattr(module, attribute)
    except Exception as exc:
        return KernelStatus(name, False, f"{type(exc).__name__}: {exc}")
    return KernelStatus(name, True, None)


def kernel_report() -> list[KernelStatus]:
    return [_probe(name, module, attribute) for name, (module, attribute) in FAST_KERNELS.items()]


def require_fast_kernels() -> list[KernelStatus]:
    report = kernel_report()
    missing = [status for status in report if status.name in FAST_KERNELS and not status.available]
    if missing:
        details = "; ".join(f"{status.name}: {status.error}" for status in missing)
        raise FastKernelsMissing(f"DeltaNet fast kernels are not usable, refusing to train slowly ({details})")
    return report
