"""The checkpoint a runtime loads: the inserted one with the trained module's tensors in place of the copies."""

import argparse
import json
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file

from rein.neurons import constants as c
from rein.neurons.checkpoint import shard_names


def install_module(source: Path, weights: Path, target: Path) -> int:
    """Copies `source` with every tensor named in `weights` replaced; each must keep its shape and dtype."""
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"{target} is not empty")
    trained = load_file(str(weights))
    target.mkdir(parents=True, exist_ok=True)
    placed = 0
    for name in shard_names(source):
        tensors = {}
        with safe_open(str(source / name), framework="pt") as reader:
            metadata = reader.metadata() or c.SAFETENSORS_METADATA
            names = reader.keys()
            for key in names:
                tensor = reader.get_tensor(key)
                if key in trained:
                    new = trained[key]
                    if new.shape != tensor.shape or new.dtype != tensor.dtype:
                        raise ValueError(f"{key}: {tuple(new.shape)} {new.dtype} cannot replace {tuple(tensor.shape)}")
                    tensor, placed = new, placed + 1
                tensors[key] = tensor
        save_file(tensors, str(target / name), metadata=metadata)
    if placed != len(trained):
        raise ValueError(f"{len(trained) - placed} trained tensors have no place in {source}")
    for path in source.iterdir():
        if path.is_file() and path.suffix != c.SAFETENSORS_SUFFIX:
            shutil.copy2(path, target / path.name)
    return placed


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the trained rule module into the inserted checkpoint.")
    parser.add_argument("--source", type=Path, required=True, help="the checkpoint with the inserted layers")
    parser.add_argument("--weights", type=Path, required=True, help="the module's trained tensors")
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    placed = install_module(args.source, args.weights, args.target)
    print(json.dumps({"placed": placed, "target": str(args.target)}), flush=True)


if __name__ == "__main__":
    main()
