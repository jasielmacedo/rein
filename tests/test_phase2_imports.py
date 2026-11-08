import subprocess
import sys

import pytest

PHASE2_ENTRY_POINTS = (
    "rein.genview.build",
    "rein.genview.reasoning",
    "rein.genview.prompts",
    "rein.neurons.extract",
    "rein.neurons.portability",
    "rein.neurons.runtimes",
    "rein.compiler.readings",
    "rein.compiler.run",
)
ABSENT_IN_PHASE2_IMAGES = ("peft", "lingua")


@pytest.mark.parametrize("module", PHASE2_ENTRY_POINTS)
def test_phase2_entry_points_import_without_phase1_only_packages(module: str) -> None:
    """The phase 2 images ship neither peft (LoRA) nor lingua (data generation)."""
    blocked = "; ".join(f"sys.modules[{name!r}] = None" for name in ABSENT_IN_PHASE2_IMAGES)
    code = f"import sys; {blocked}; import {module}"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]
