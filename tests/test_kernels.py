import importlib.util

import pytest

from rein.model.kernels import FAST_KERNELS, FastKernelsMissing, kernel_report, require_fast_kernels

FLA_INSTALLED = importlib.util.find_spec("fla") is not None


def test_report_covers_every_kernel_and_keeps_errors() -> None:
    report = {status.name: status for status in kernel_report()}
    assert set(FAST_KERNELS) <= set(report)
    for status in report.values():
        assert status.available == (status.error is None)


@pytest.mark.skipif(FLA_INSTALLED, reason="fla is installed here")
def test_gate_refuses_to_train_without_fast_kernels() -> None:
    with pytest.raises(FastKernelsMissing, match="chunk_gated_delta_rule"):
        require_fast_kernels()
