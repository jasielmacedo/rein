from pathlib import Path

import torch
from safetensors.torch import save_file

from rein.compiler.model import Compiler
from rein.compiler.run import COMPILER_FILE, load_compiler_weights

HIDDEN = 8
RANK = 2


def _flat_name(key: str) -> str:
    """How the compiler named its weights before the rule encoder became a module of its own."""
    if not key.startswith("encoder."):
        return key
    return key.removeprefix("encoder.").replace("blocks.", "encoder.", 1)


def test_a_compiler_saved_with_the_older_names_loads_unchanged(tmp_path: Path) -> None:
    saved = Compiler(HIDDEN, [1], rank=RANK)
    save_file(
        {_flat_name(key): value.clone() for key, value in saved.state_dict().items()}, str(tmp_path / COMPILER_FILE)
    )
    loaded = Compiler(HIDDEN, [1], rank=RANK)
    load_compiler_weights(loaded, tmp_path)
    assert all(torch.equal(saved.state_dict()[key], value) for key, value in loaded.state_dict().items())
