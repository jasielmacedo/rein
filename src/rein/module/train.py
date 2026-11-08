"""Option A, test A2: trains only the rule module, the decoder layers inserted into Qwen3.5-4B (DR-027 follow-up), to
decide per rule category whether a rule fires, with every original weight frozen around it.

As approved for option A, the module's full-attention layer is trained whole and its linear-attention layers only in
their MLPs: their token mixers keep a zero output. The decision is read where the generic detectors read theirs, the
MLP input of the first original layer after the module, so the only difference against them is what the module adds to
the residual stream; before training it adds nothing and the read is theirs exactly. The recipe is Rein-r1's:
cross-entropy plus 0.25 Brier, AdamW, one cycle.
"""

import argparse
import json
import math
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from rein.compiler.category_probe import CALIBRATION, TRAIN
from rein.genview.build import ViewRow, load_all_records
from rein.model.backbone import load_text_backbone, placed, truncate_backbone
from rein.model.kernels import require_fast_kernels
from rein.module.master import MasterWeights
from rein.module.targets import OPTIONS, LabelTable, batch_targets, label_table, labelled_views, option_targets
from rein.neurons.extract import MlpInputCapture, batches, extract_to, load_views, padded
from rein.neurons.layers import MANIFEST_FILE
from rein.train.config import TrainConfig
from rein.train.loss import LossParts, decision_loss

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

REIN_R1 = TrainConfig.model_fields
CHECKPOINT_PREFIX = "model.language_model."
MODULE_FILE = "module.safetensors"
HEAD_FILE = "head.safetensors"
REPORT_FILE = "train.json"
RESUME_FILE = "resume.pt"
PARTIAL_SUFFIX = ".partial"
STATES_DIR = "states"
RESUME_EVERY_STEPS = 800
LINEAR_ATTENTION = "linear_attention"
ZEROED_MIXER_PARTS = ("linear_attn.", "input_layernorm.")

Logger = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class ModuleSettings:
    epochs: int = REIN_R1["epochs"].default
    lr: float = REIN_R1["lora_lr"].default
    head_lr: float = REIN_R1["head_lr"].default
    weight_decay: float = REIN_R1["weight_decay"].default
    warmup_fraction: float = REIN_R1["warmup_fraction"].default
    grad_clip: float = REIN_R1["grad_clip"].default
    brier_weight: float = REIN_R1["brier_weight"].default
    batch_tokens: int = REIN_R1["batch_tokens"].default
    seed: int = REIN_R1["seed"].default
    log_every_steps: int = REIN_R1["log_every_steps"].default
    resume_every_steps: int = RESUME_EVERY_STEPS


@dataclass(frozen=True)
class ModulePlace:
    layers: range
    read_layer: int


@dataclass
class Progress:
    epoch: int = 0
    batch: int = 0
    step: int = 0
    best_loss: float = math.inf
    best_epoch: int = -1
    history: list[dict[str, Any]] = field(default_factory=list)


def module_place(checkpoint: Path) -> ModulePlace:
    """The inserted layers, from the checkpoint's manifest, and the first original layer after them."""
    manifest = json.loads((checkpoint / MANIFEST_FILE).read_text(encoding="utf-8"))
    start, size = int(manifest["inserted_at"]), int(manifest["group"])
    return ModulePlace(range(start, start + size), start + size)


def trained_part(layer_type: str, name: str) -> bool:
    """Whether a parameter of a module layer is trained: all of a full-attention layer, and the MLP side of a
    linear-attention layer, whose token mixer stays at zero output."""
    return layer_type != LINEAR_ATTENTION or not name.startswith(ZEROED_MIXER_PARTS)


def prepared_backbone(
    checkpoint: Path, place: ModulePlace, device: torch.device, dtype: torch.dtype, whole: bool = False
) -> Qwen3_5TextModel:
    """The decoder up to the read layer (every layer when `whole`), frozen except for the module's trained parts."""
    backbone = load_text_backbone(str(checkpoint), dtype=dtype)
    backbone = placed(backbone if whole else truncate_backbone(backbone, place.read_layer + 1), device)
    for param in backbone.parameters():
        param.requires_grad_(False)
    types = list(backbone.config.layer_types or [])
    for layer in place.layers:
        for name, param in backbone.layers[layer].named_parameters():
            param.requires_grad_(trained_part(types[layer], name))
    return backbone


def module_parameters(backbone: Qwen3_5TextModel) -> dict[str, nn.Parameter]:
    """The trainable parameters under their names in the checkpoint."""
    return {f"{CHECKPOINT_PREFIX}{name}": param for name, param in backbone.named_parameters() if param.requires_grad}


def new_head(hidden_size: int, categories: int, seed: int) -> nn.Linear:
    torch.manual_seed(seed)
    return nn.Linear(hidden_size, categories * OPTIONS)


def readout_loss(head: nn.Linear, states: torch.Tensor, labels: torch.Tensor, brier_weight: float) -> LossParts:
    """Cross-entropy plus Brier of the decision read, every position weighing the same."""
    logits = head(states.float()).reshape(-1, OPTIONS)
    one_hot, owners = option_targets(labels.to(logits.device))
    return decision_loss(logits, torch.ones_like(logits, dtype=torch.bool), one_hot, owners, brier_weight)


def batch_loss(
    backbone: Qwen3_5TextModel,
    capture: MlpInputCapture,
    head: nn.Linear,
    batch: Sequence[ViewRow],
    table: LabelTable,
    rng: random.Random,
    brier_weight: float,
) -> tuple[LossParts, int]:
    """The loss over one batch's decision and silence positions, and how many positions it covers."""
    device = head.weight.device
    targets = batch_targets(batch, table, rng)
    ids, mask = padded(batch, int(backbone.config.pad_token_id or 0))
    capture.aim(targets.rows.to(device), targets.columns.to(device))
    backbone(input_ids=ids.to(device), attention_mask=mask.to(device), use_cache=False)
    return readout_loss(head, capture.stacked()[:, 0], targets.labels, brier_weight), len(targets)


@torch.no_grad()
def mean_loss(
    backbone: Qwen3_5TextModel,
    capture: MlpInputCapture,
    head: nn.Linear,
    views: Sequence[ViewRow],
    table: LabelTable,
    settings: ModuleSettings,
) -> float:
    """The loss per position over `views`, with the same silences at every call."""
    total, positions = 0.0, 0
    for index, batch in enumerate(batches(views, settings.batch_tokens)):
        parts, count = batch_loss(
            backbone, capture, head, batch, table, random.Random(f"{settings.seed}:held:{index}"), settings.brier_weight
        )
        total += float(parts.total) * count
        positions += count
    return total / max(1, positions)


def _settings_key(settings: "DataclassInstance", trained: Sequence[str]) -> str:
    return json.dumps({"settings": asdict(settings), "trained": sorted(trained)}, sort_keys=True)


def save_resume(
    out: Path,
    settings: "DataclassInstance",
    progress: Progress,
    master: MasterWeights,
    head: nn.Linear,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> None:
    payload = {
        "settings": _settings_key(settings, list(master.names)),
        "progress": asdict(progress),
        "master": [weight.detach().cpu() for weight in master.master],
        "head": head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    partial = out / f"{RESUME_FILE}{PARTIAL_SUFFIX}"
    torch.save(payload, partial)
    partial.replace(out / RESUME_FILE)


def load_resume(
    out: Path,
    settings: "DataclassInstance",
    master: MasterWeights,
    head: nn.Linear,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> Progress | None:
    """The saved point of an interrupted run with the same settings, restored into the weights and the optimizer."""
    path = out / RESUME_FILE
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["settings"] != _settings_key(settings, list(master.names)):
        return None
    master.load(payload["master"])
    head.load_state_dict(payload["head"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    return Progress(**payload["progress"])


def save_weights(out: Path, params: Mapping[str, nn.Parameter], head: nn.Linear) -> None:
    save_file({name: param.detach().cpu().contiguous() for name, param in params.items()}, str(out / MODULE_FILE))
    save_file(
        {name: value.detach().cpu().contiguous() for name, value in head.state_dict().items()}, str(out / HEAD_FILE)
    )


@torch.no_grad()
def load_weights(out: Path, params: Mapping[str, nn.Parameter], head: nn.Linear) -> None:
    saved = load_file(str(out / MODULE_FILE))
    for name, param in params.items():
        param.copy_(saved[name])
    head.load_state_dict(load_file(str(out / HEAD_FILE)))


def train_module(
    backbone: Qwen3_5TextModel,
    place: ModulePlace,
    head: nn.Linear,
    train_views: Sequence[ViewRow],
    held_views: Sequence[ViewRow],
    table: LabelTable,
    settings: ModuleSettings,
    out: Path,
    log: Logger = print,
) -> Progress:
    """Trains the module and the read on `train_views`; after each epoch the weights with the lowest loss on
    `held_views` are kept in `out`. An interrupted run resumes from its last saved point."""
    out.mkdir(parents=True, exist_ok=True)
    params = module_parameters(backbone)
    master = MasterWeights(params)
    optimizer = torch.optim.AdamW(
        [{"params": master.master, "lr": settings.lr}, {"params": list(head.parameters()), "lr": settings.head_lr}],
        weight_decay=settings.weight_decay,
    )
    plan = list(batches(train_views, settings.batch_tokens))
    total_steps = max(1, settings.epochs * len(plan))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[settings.lr, settings.head_lr],
        total_steps=total_steps,
        pct_start=settings.warmup_fraction,
    )
    progress = load_resume(out, settings, master, head, optimizer, scheduler) or Progress()
    log({"event": "plan", "batches_per_epoch": len(plan), "steps": total_steps, "resumed_at_step": progress.step})
    capture = MlpInputCapture(backbone, [place.read_layer], detach=False)
    started = time.monotonic()
    running = torch.zeros(3)
    tokens = positions = 0
    try:
        for epoch in range(progress.epoch, settings.epochs):
            order = random.Random(f"{settings.seed}:{epoch}").sample(plan, len(plan))
            for index in range(progress.batch, len(order)):
                rng = random.Random(f"{settings.seed}:{epoch}:{index}")
                parts, count = batch_loss(backbone, capture, head, order[index], table, rng, settings.brier_weight)
                parts.total.backward()
                master.pull_gradients()
                torch.nn.utils.clip_grad_norm_([*master.master, *head.parameters()], settings.grad_clip)
                optimizer.step()
                if progress.step < total_steps - 1:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                master.push_weights()
                progress.step, progress.batch = progress.step + 1, index + 1
                running += torch.stack([parts.total.detach(), parts.ce.detach(), parts.brier.detach()]).cpu()
                tokens += sum(len(view.ids) for view in order[index])
                positions += count
                if progress.step % settings.log_every_steps == 0:
                    loss, ce, brier = (running / settings.log_every_steps).tolist()
                    elapsed = time.monotonic() - started
                    log(
                        {
                            "epoch": epoch,
                            "step": progress.step,
                            "loss": round(loss, 5),
                            "ce": round(ce, 5),
                            "brier": round(brier, 5),
                            "positions": positions,
                            "tokens_per_s": round(tokens / max(elapsed, 1e-9)),
                            "elapsed_s": round(elapsed, 1),
                        }
                    )
                    running.zero_()
                if progress.step % settings.resume_every_steps == 0:
                    save_resume(out, settings, progress, master, head, optimizer, scheduler)
            held = mean_loss(backbone, capture, head, held_views, table, settings)
            progress.history.append({"epoch": epoch, "step": progress.step, "held_loss": held})
            log({"epoch": epoch, "held_loss": round(held, 5)})
            if held < progress.best_loss:
                progress.best_loss, progress.best_epoch = held, epoch
                save_weights(out, params, head)
            progress.epoch, progress.batch = epoch + 1, 0
            save_resume(out, settings, progress, master, head, optimizer, scheduler)
    finally:
        capture.close()
    return progress


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the inserted rule module to decide per rule category.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Qwen3.5 with the module's layers inserted")
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--categories", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=ModuleSettings.epochs)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(require_fast_kernels(), flush=True)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    settings = ModuleSettings(epochs=args.epochs)
    table = label_table(load_all_records(args.data), json.loads(args.categories.read_text(encoding="utf-8")))
    views = load_views(args.views)
    place = module_place(args.checkpoint)
    backbone = prepared_backbone(args.checkpoint, place, device, dtype)
    head = new_head(backbone.config.hidden_size, len(table.categories), settings.seed).to(device)
    train_views, held_views = labelled_views(views, table, TRAIN), labelled_views(views, table, CALIBRATION)

    def log(entry: dict[str, Any]) -> None:
        print(json.dumps(entry), flush=True)

    log({"event": "ready", "views": len(views), "train": len(train_views), "held": len(held_views)})
    progress = train_module(backbone, place, head, train_views, held_views, table, settings, args.out, log)
    load_weights(args.out, module_parameters(backbone), head)
    report = {
        "checkpoint": str(args.checkpoint),
        "settings": asdict(settings),
        "categories": list(table.categories),
        "module_layers": list(place.layers),
        "read_layer": place.read_layer,
        "progress": asdict(progress),
    }
    (args.out / REPORT_FILE).write_text(json.dumps(report, indent=2), encoding="utf-8")
    model_name = f"{args.checkpoint}+{MODULE_FILE}"
    log(extract_to(backbone, views, [place.read_layer], args.out / STATES_DIR, model_name))


if __name__ == "__main__":
    main()
