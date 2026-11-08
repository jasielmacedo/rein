"""Option A, the action: trains the rule module (its full-attention layer and its MLPs, Qwen frozen) so that replies do
the right thing where a rule acts and stay as Qwen wrote them elsewhere, while it keeps a decision read. It starts from
the inserted layers at zero output, which is Qwen exactly: the module trained only to decide (test A2) writes features
large enough to move Qwen's replies by 1.5 nats per token, so it is no start for a module that must leave them alone.

- Preference (DPO) on two of Qwen's own replies to one prompt, one right and one wrong (`rein.module.pairs`), against
  the checkpoint with the module bypassed, which computes what Qwen computes. Each kind of mistake (a call not needed,
  the call needed, the rule) weighs the same in total, so the many unneeded calls do not teach the module to stop
  calling.
- KL to Qwen over the whole vocabulary at every reply token of the anchors, Qwen's replies where every sample was right,
  and at positions drawn from every prompt: the module must leave the reading of the conversation alone, not only the
  reply (trained on reply tokens alone, it moved the prompt's predictions by 0.36 nats per token).
- The decision read at the reply opening, on Rein-r1's soft labels when a teacher is given.
- The action at the reply opening (`rein.module.opening_loss`): the call's first tokens where the turn needs a tool,
  and no `<tool_call>` where it needs none, on every training view with such a target.

Every batch of every kind is seen once per epoch, in one shuffled order.
"""

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from rein.genview.build import load_all_records
from rein.model.kernels import require_fast_kernels
from rein.module.losses import (
    Teacher,
    bypassed,
    decision_loss_of,
    divergence,
    last_hidden,
    log_probs_at,
    sequence_sums,
    stacked,
)
from rein.module.master import MasterWeights
from rein.module.material import ANCHORS_FILE, PAIRS_FILE, read_rows
from rein.module.opening_loss import Opening, load_openings, opening_losses
from rein.module.targets import LabelTable, label_table
from rein.module.train import (
    REIN_R1,
    ModulePlace,
    Progress,
    load_resume,
    load_weights,
    module_parameters,
    module_place,
    new_head,
    prepared_backbone,
    save_resume,
    save_weights,
)
from rein.neurons.extract import MLP_HIDDEN, MlpInputCapture, length_batches

DPO_BETA = 0.1
REPORT_FILE = "act.json"
CHOSEN, REJECTED, REPLY, KIND = "chosen", "rejected", "reply", "kind"
PAIR_STREAM, ANCHOR_STREAM, OPENING_STREAM = "pairs", "anchors", "openings"
WRITER_PARTS = ("o_proj", "down_proj")

Logger = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class ActSettings:
    epochs: int = 1
    lr: float = REIN_R1["lora_lr"].default
    head_lr: float = REIN_R1["head_lr"].default
    beta: float = DPO_BETA
    kl_weight: float = 1.0
    decision_weight: float = 1.0
    weight_decay: float = REIN_R1["weight_decay"].default
    warmup_fraction: float = REIN_R1["warmup_fraction"].default
    grad_clip: float = REIN_R1["grad_clip"].default
    brier_weight: float = REIN_R1["brier_weight"].default
    batch_tokens: int = 8192
    accumulation: int = 4
    seed: int = REIN_R1["seed"].default
    log_every_steps: int = REIN_R1["log_every_steps"].default
    resume_every_steps: int = 200
    checkpointing: bool = True
    read_inside: bool = False
    opening_weight: float = 1.0
    read_lr_factor: float = 1.0


@dataclass(frozen=True)
class Example:
    record_id: str
    prompt: tuple[int, ...]
    replies: tuple[tuple[int, ...], ...]
    weight: float = 1.0
    view_id: str = ""

    @property
    def width(self) -> int:
        return len(self.prompt) + max(len(reply) for reply in self.replies)


def load_examples(paths: Sequence[Path], fields: Sequence[str], balance: str | None = None) -> list[Example]:
    """The examples of pairs or anchors files; with `balance`, weights that give each value of that field the same
    total, averaging 1."""
    rows = [row for path in paths for row in read_rows(path) if all(row[field] for field in fields)]
    counts = Counter(row[balance] for row in rows) if balance else Counter()
    return [
        Example(
            row["record_id"],
            tuple(row["prompt"]),
            tuple(tuple(row[field]) for field in fields),
            len(rows) / (len(counts) * counts[row[balance]]) if balance else 1.0,
            str(row.get("view_id", "")),
        )
        for row in rows
    ]


def load_teacher(path: Path, categories: Sequence[str]) -> dict[str, tuple[float, ...]]:
    """Rein-r1's chance that some rule of each category fires, per view, in the read's category order (a category
    the record has no rule of does not fire)."""
    return {
        row["view_id"]: tuple(float(row["fire"].get(category, 0.0)) for category in categories)
        for row in read_rows(path)
    }


def pair_losses(
    backbone: Qwen3_5TextModel,
    place: ModulePlace,
    capture: MlpInputCapture,
    head: nn.Linear,
    pairs: Sequence[Example],
    table: LabelTable,
    settings: ActSettings,
    rng: random.Random,
    teacher: Teacher | None = None,
) -> dict[str, torch.Tensor]:
    """DPO on each pair against the bypassed module, the KL inside the shared prompt, and the decision read at the
    reply opening."""
    count = len(pairs)
    pieces = [(pair.prompt, pair.replies[0]) for pair in pairs] + [(pair.prompt, pair.replies[1]) for pair in pairs]
    batch = stacked(pieces, int(backbone.config.pad_token_id or 0), rng, count)
    device = head.weight.device
    capture.aim(torch.arange(count, device=device), batch.openings[:count].to(device))
    with torch.no_grad(), bypassed(backbone, place.layers):
        hidden = last_hidden(backbone, batch)
        reference = sequence_sums(log_probs_at(backbone, hidden, batch.rows, batch.columns), batch, 2 * count)
        reference_prompt = log_probs_at(backbone, hidden, batch.prompt_rows, batch.prompt_columns)
        del hidden
    hidden = last_hidden(backbone, batch)
    policy = sequence_sums(log_probs_at(backbone, hidden, batch.rows, batch.columns), batch, 2 * count)
    prompt_kl = divergence(reference_prompt, log_probs_at(backbone, hidden, batch.prompt_rows, batch.prompt_columns))
    margin = settings.beta * ((policy[:count] - reference[:count]) - (policy[count:] - reference[count:]))
    decision = decision_loss_of(head, capture, pairs, table, teacher, settings.brier_weight)
    weights = torch.tensor([pair.weight for pair in pairs], device=device)
    return {
        "preference": (weights * -F.logsigmoid(margin)).mean(),
        "prompt_kl": prompt_kl,
        "decision": decision,
        "margin": margin.detach().mean(),
        "won": (margin.detach() > 0).float().mean(),
    }


def anchor_losses(
    backbone: Qwen3_5TextModel,
    place: ModulePlace,
    capture: MlpInputCapture,
    head: nn.Linear,
    anchors: Sequence[Example],
    table: LabelTable,
    settings: ActSettings,
    rng: random.Random,
    teacher: Teacher | None = None,
) -> dict[str, torch.Tensor]:
    """KL from Qwen to the checkpoint at every reply token and at positions drawn from the prompt, and the decision
    read at the reply opening."""
    pieces = [(anchor.prompt, anchor.replies[0]) for anchor in anchors]
    batch = stacked(pieces, int(backbone.config.pad_token_id or 0), rng, len(pieces))
    rows, columns = torch.cat([batch.rows, batch.prompt_rows]), torch.cat([batch.columns, batch.prompt_columns])
    device = head.weight.device
    capture.aim(torch.arange(len(anchors), device=device), batch.openings.to(device))
    with torch.no_grad(), bypassed(backbone, place.layers):
        reference = log_probs_at(backbone, last_hidden(backbone, batch), rows, columns)
    kl = divergence(reference, log_probs_at(backbone, last_hidden(backbone, batch), rows, columns))
    decision = decision_loss_of(head, capture, anchors, table, teacher, settings.brier_weight)
    return {"kl": kl, "anchor_decision": decision}


def decision_capture(backbone: Qwen3_5TextModel, place: ModulePlace, inside: bool) -> MlpInputCapture:
    """Where the decision is read: inside the module (the hidden units of its last MLP, so the decision need not be
    written into the residual stream) or at the MLP input of the first original layer after it, as in test A2."""
    if inside:
        return MlpInputCapture(backbone, [place.layers[-1]], detach=False, part=MLP_HIDDEN)
    return MlpInputCapture(backbone, [place.read_layer], detach=False)


def read_size(backbone: Qwen3_5TextModel, inside: bool) -> int:
    return int(backbone.config.intermediate_size if inside else backbone.config.hidden_size)


def writes_residual(name: str) -> bool:
    """Whether a module parameter writes into the residual stream (the attention's and the MLP's output
    projections); the others only read, and may learn faster without moving Qwen more."""
    return any(f".{part}." in name for part in WRITER_PARTS)


def pair_width(pair: Example) -> int:
    return len(pair.replies) * pair.width


def single_width(item: Example | Opening) -> int:
    return item.width


def train_action(
    backbone: Qwen3_5TextModel,
    place: ModulePlace,
    head: nn.Linear,
    pairs: Sequence[Example],
    anchors: Sequence[Example],
    table: LabelTable,
    settings: ActSettings,
    out: Path,
    log: Logger = print,
    teacher: Teacher | None = None,
    openings: Sequence[Opening] = (),
) -> Progress:
    """Every batch of pairs, anchors and openings once per epoch, in one shuffled order; `accumulation` of them make one
    optimizer step. The weights are saved at the end and an interrupted run resumes from its last saved point."""
    plan: list[tuple[str, Sequence[Any]]] = [
        *((PAIR_STREAM, batch) for batch in length_batches(pairs, settings.batch_tokens, pair_width)),
        *((ANCHOR_STREAM, batch) for batch in length_batches(anchors, settings.batch_tokens, single_width)),
        *((OPENING_STREAM, batch) for batch in length_batches(openings, settings.batch_tokens, single_width)),
    ]
    if not plan:
        raise ValueError("the action needs pairs, anchors or openings")
    out.mkdir(parents=True, exist_ok=True)
    params = module_parameters(backbone)
    master = MasterWeights(params)
    writing = [weight for name, weight in zip(master.names, master.master, strict=True) if writes_residual(name)]
    reading = [weight for name, weight in zip(master.names, master.master, strict=True) if not writes_residual(name)]
    rates = [settings.lr, settings.lr * settings.read_lr_factor, settings.head_lr]
    groups = [writing, reading, list(head.parameters())]
    optimizer = torch.optim.AdamW(
        [{"params": params, "lr": rate} for params, rate in zip(groups, rates, strict=True)],
        weight_decay=settings.weight_decay,
    )
    total_steps = max(1, settings.epochs * -(-len(plan) // settings.accumulation))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=rates, total_steps=total_steps, pct_start=settings.warmup_fraction
    )
    progress = load_resume(out, settings, master, head, optimizer, scheduler) or Progress()
    log({"event": "plan", "batches": dict(Counter(kind for kind, _ in plan)), "steps": total_steps})
    capture = decision_capture(backbone, place, settings.read_inside)
    if settings.checkpointing:
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        backbone.train()

    def micro_batch(
        kind: str, batch: Sequence[Any], rng: random.Random
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if kind == PAIR_STREAM:
            parts = pair_losses(backbone, place, capture, head, batch, table, settings, rng, teacher)
            moved = settings.kl_weight * parts["prompt_kl"] + settings.decision_weight * parts["decision"]
            return parts["preference"] + moved, parts
        if kind == ANCHOR_STREAM:
            parts = anchor_losses(backbone, place, capture, head, batch, table, settings, rng, teacher)
            return settings.kl_weight * parts["kl"] + settings.decision_weight * parts["anchor_decision"], parts
        parts = opening_losses(backbone, place, capture, head, batch, table, settings.brier_weight, rng, teacher)
        steady = settings.kl_weight * parts["opening_prompt_kl"] + settings.decision_weight * parts["opening_decision"]
        return settings.opening_weight * parts["opening"] + steady, parts

    running: dict[str, float] = defaultdict(float)
    seen: Counter[str] = Counter()
    started = time.monotonic()
    try:
        for epoch in range(progress.epoch, settings.epochs):
            order = random.Random(f"{settings.seed}:{epoch}").sample(plan, len(plan))
            for index in range(progress.batch, len(order)):
                kind, batch = order[index]
                total, parts = micro_batch(kind, batch, random.Random(f"{settings.seed}:{epoch}:{index}"))
                (total / settings.accumulation).backward()
                master.pull_gradients()
                for name, value in parts.items():
                    running[name] += float(value.detach())
                    seen[name] += 1
                last = index + 1 == len(order)
                if (index + 1) % settings.accumulation and not last:
                    continue
                torch.nn.utils.clip_grad_norm_([*master.master, *head.parameters()], settings.grad_clip)
                optimizer.step()
                if progress.step < total_steps - 1:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                master.push_weights()
                progress.step, progress.batch = progress.step + 1, index + 1
                if progress.step % settings.log_every_steps == 0:
                    entry = {name: round(value / seen[name], 5) for name, value in running.items()}
                    elapsed = round(time.monotonic() - started, 1)
                    log({"epoch": epoch, "step": progress.step, **entry, "elapsed_s": elapsed})
                    running.clear()
                    seen.clear()
                if progress.step % settings.resume_every_steps == 0:
                    save_resume(out, settings, progress, master, head, optimizer, scheduler)
            progress.epoch, progress.batch = epoch + 1, 0
            save_resume(out, settings, progress, master, head, optimizer, scheduler)
    finally:
        capture.close()
    save_weights(out, params, head)
    return progress


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the rule module to act on its decisions.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Qwen3.5 with the module's layers inserted")
    parser.add_argument("--start", type=Path, default=None, help="a trained module and read to start from")
    parser.add_argument("--pairs", type=Path, nargs="+", required=True, help="folders with pairs and anchors")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--categories", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=ActSettings.epochs)
    parser.add_argument("--kl-weight", type=float, default=ActSettings.kl_weight)
    parser.add_argument("--decision-weight", type=float, default=ActSettings.decision_weight)
    parser.add_argument("--lr", type=float, default=ActSettings.lr, help="the module's peak learning rate")
    parser.add_argument("--teacher", type=Path, default=None, help="Rein-r1's soft labels per view")
    parser.add_argument("--read-inside", action="store_true", help="read the decision inside the module")
    parser.add_argument("--openings", type=Path, default=None, help="the training reply openings' targets")
    parser.add_argument("--opening-views", type=Path, nargs="*", default=[], help="the views with their prompts")
    parser.add_argument("--opening-weight", type=float, default=ActSettings.opening_weight)
    parser.add_argument(
        "--read-lr-factor", type=float, default=ActSettings.read_lr_factor, help="lr times this for the reading parts"
    )
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(require_fast_kernels(), flush=True)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    settings = ActSettings(
        epochs=args.epochs,
        kl_weight=args.kl_weight,
        decision_weight=args.decision_weight,
        lr=args.lr,
        read_inside=args.read_inside,
        opening_weight=args.opening_weight,
        read_lr_factor=args.read_lr_factor,
    )
    table = label_table(load_all_records(args.data), json.loads(args.categories.read_text(encoding="utf-8")))
    place = module_place(args.checkpoint)
    backbone = prepared_backbone(args.checkpoint, place, device, dtype, whole=True)
    head = new_head(read_size(backbone, settings.read_inside), len(table.categories), settings.seed).to(device)
    if args.start is not None:
        load_weights(args.start, module_parameters(backbone), head)
    pairs = load_examples([folder / PAIRS_FILE for folder in args.pairs], (CHOSEN, REJECTED), KIND)
    anchors = load_examples([folder / ANCHORS_FILE for folder in args.pairs], (REPLY,))

    def log(entry: dict[str, Any]) -> None:
        print(json.dumps(entry), flush=True)

    openings = load_openings(args.openings, args.opening_views) if args.openings else []
    teacher = load_teacher(args.teacher, table.categories) if args.teacher else None
    seen = [*(item.view_id for item in (*pairs, *anchors)), *(opening.view_id for opening in openings)]
    taught = sum(view_id in teacher for view_id in seen) if teacher else 0
    targets = dict(Counter(str(opening.calls) for opening in openings))
    log({"event": "ready", "pairs": len(pairs), "anchors": len(anchors), "openings": targets, "taught": taught})
    progress = train_action(backbone, place, head, pairs, anchors, table, settings, args.out, log, teacher, openings)
    report = {
        "settings": asdict(settings),
        "start": str(args.start),
        "teacher": str(args.teacher),
        "openings": targets,
        "categories": list(table.categories),
        "progress": asdict(progress),
    }
    (args.out / REPORT_FILE).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
