"""The module's own decision on development conversations: the trained read at each view's reply opening, scored on the
rule in play (the category of the rule the record's case was written for) beside Rein-r1 on the same views, with how
often the two decide alike, how far their chances sit apart and how well each separates firing from quiet (AUC). The
read's temperature is fitted on the calibration split, never on the development views."""

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean

import torch
from safetensors.torch import load_file
from torch import nn
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from rein.compiler.category_probe import DEVELOPMENT, PLAY_THRESHOLD, category_labels
from rein.genview.build import NO_THINKING, ViewRow, load_all_records
from rein.module.act import REPORT_FILE as ACT_REPORT
from rein.module.act import decision_capture, load_teacher, read_size
from rein.module.material import LONG, is_long
from rein.module.targets import OPTIONS
from rein.module.train import HEAD_FILE, module_place, new_head, prepared_backbone
from rein.neurons.extract import DEFAULT_TOKEN_BUDGET, MlpInputCapture, batches, load_views, padded
from rein.schema import TrainingRecord

DECISIONS_FILE = "decisions.json"
FIRE_FILE = "fire.jsonl"
SHORT = "short"
HEAD_SEED = 0
TEMPERATURE_RANGE = (0.05, 20.0)
TEMPERATURE_STEPS = 400
TIE = 0.5

Fire = Mapping[str, Sequence[float]]
Truths = Mapping[str, tuple[int, bool]]
Labels = Mapping[str, Mapping[int, int]]


@torch.no_grad()
def module_odds(
    backbone: Qwen3_5TextModel,
    capture: MlpInputCapture,
    head: nn.Linear,
    views: Sequence[ViewRow],
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> dict[str, tuple[float, ...]]:
    """Per view, the read's log-odds that each category fires at the reply opening (the view's last token)."""
    device = head.weight.device
    pad_id = int(backbone.config.pad_token_id or 0)
    found: dict[str, tuple[float, ...]] = {}
    for batch in batches(views, token_budget):
        ids, mask = padded(batch, pad_id)
        openings = torch.tensor([len(view.ids) - 1 for view in batch], device=device)
        capture.aim(torch.arange(len(batch), device=device), openings)
        backbone(input_ids=ids.to(device), attention_mask=mask.to(device), use_cache=False)
        logits = head(capture.stacked()[:, 0].float()).reshape(len(batch), -1, OPTIONS)
        odds = (logits[..., 1] - logits[..., 0]).cpu().tolist()
        found |= {view.view_id: tuple(row) for view, row in zip(batch, odds, strict=True)}
    return found


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value)) if value >= 0 else math.exp(value) / (1.0 + math.exp(value))


def softplus(value: float) -> float:
    return max(value, 0.0) + math.log1p(math.exp(-abs(value)))


def chances(odds: Fire, temperature: float) -> dict[str, tuple[float, ...]]:
    return {view_id: tuple(sigmoid(value / temperature) for value in values) for view_id, values in odds.items()}


def fitted_temperature(odds: Fire, labels: Labels) -> float:
    """The temperature with the lowest log-loss of the read on labelled views, over a grid in log space."""
    pairs = [
        (odds[view_id][index], label)
        for view_id, known in labels.items()
        if view_id in odds
        for index, label in known.items()
    ]
    if not pairs:
        return 1.0
    low, high = TEMPERATURE_RANGE
    grid = [low * (high / low) ** (step / (TEMPERATURE_STEPS - 1)) for step in range(TEMPERATURE_STEPS)]
    return min(
        grid, key=lambda t: fmean(softplus(-value / t) if label else softplus(value / t) for value, label in pairs)
    )


def view_labels(
    views: Sequence[ViewRow],
    records: Mapping[str, TrainingRecord],
    mapping: Mapping[str, Mapping[str, str]],
    categories: Sequence[str],
) -> dict[str, dict[int, int]]:
    """Per view, the label of every category its record has rules of, by the read's index."""
    found: dict[str, dict[int, int]] = {}
    for view in views:
        record = records.get(view.record_id)
        if record is None:
            continue
        labels = category_labels(record, mapping.get(view.record_id, {}))
        found[view.view_id] = {
            list(categories).index(name): value for name, value in labels.items() if name in categories
        }
    return found


def rule_in_play(
    views: Sequence[ViewRow],
    records: Mapping[str, TrainingRecord],
    mapping: Mapping[str, Mapping[str, str]],
    categories: Sequence[str],
) -> dict[str, tuple[int, bool]]:
    """Per view: the read's index of the category of its rule in play, and whether some rule of it fires."""
    found: dict[str, tuple[int, bool]] = {}
    for view in views:
        record = records.get(view.record_id)
        if record is None:
            continue
        category = (record.meta or {}).get("category")
        labels = category_labels(record, mapping.get(view.record_id, {}))
        if category in labels and category in categories:
            found[view.view_id] = (list(categories).index(category), bool(labels[category]))
    return found


def separation(fire: Fire, truths: Truths, shared: Sequence[str]) -> float | None:
    """The chance that a firing view gets a higher chance than a quiet one on its rule in play (AUC)."""
    positives = [fire[view_id][truths[view_id][0]] for view_id in shared if truths[view_id][1]]
    negatives = [fire[view_id][truths[view_id][0]] for view_id in shared if not truths[view_id][1]]
    if not positives or not negatives:
        return None
    wins = sum(float(p > n) + TIE * float(p == n) for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def play_scores(
    fire: Fire, teacher: Fire, truths: Truths, threshold: float = PLAY_THRESHOLD
) -> dict[str, float | None]:
    """On the views both models read: each one's accuracy and AUC on the rule in play, how often they decide alike
    there, and the mean distance between their chances over every category."""
    shared = [view_id for view_id in truths if view_id in fire and view_id in teacher]
    if not shared:
        return {"n": 0}
    ours = [fire[view_id][truths[view_id][0]] >= threshold for view_id in shared]
    theirs = [teacher[view_id][truths[view_id][0]] >= threshold for view_id in shared]
    truth = [truths[view_id][1] for view_id in shared]
    gaps = [fmean(abs(a - b) for a, b in zip(fire[view_id], teacher[view_id], strict=True)) for view_id in shared]
    return {
        "n": len(shared),
        "module": fmean(a == t for a, t in zip(ours, truth, strict=True)),
        "rein_r1": fmean(b == t for b, t in zip(theirs, truth, strict=True)),
        "agreement": fmean(a == b for a, b in zip(ours, theirs, strict=True)),
        "mean_gap": fmean(gaps),
        "module_auc": separation(fire, truths, shared),
        "rein_r1_auc": separation(teacher, truths, shared),
    }


def scores_by_set(fire: Fire, teacher: Fire, truths: Truths) -> dict[str, dict[str, float | None]]:
    """The scores of the short Rein conversations and of the long ones, apart."""
    return {
        name: play_scores(fire, teacher, {view_id: t for view_id, t in truths.items() if is_long(view_id) == long})
        for name, long in ((SHORT, False), (LONG, True))
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="The module's own decisions on development views, beside Rein-r1's.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="the checkpoint with the trained module")
    parser.add_argument("--run", type=Path, required=True, help="the action run folder, with its report and read head")
    parser.add_argument("--views", type=Path, nargs="+", required=True)
    parser.add_argument("--calibration-views", type=Path, default=None, help="views to fit the read's temperature on")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--categories", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True, help="Rein-r1's soft labels per view")
    parser.add_argument("--split", default=DEVELOPMENT, help="the views' split to score")
    parser.add_argument("--temperature", type=float, default=None, help="a temperature fitted earlier, kept")
    parser.add_argument("--out", type=Path, default=None, help="where the scores go (the run folder by default)")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    report = json.loads((args.run / ACT_REPORT).read_text(encoding="utf-8"))
    categories, inside = tuple(report["categories"]), bool(report["settings"]["read_inside"])
    place = module_place(args.checkpoint)
    backbone = prepared_backbone(args.checkpoint, place, device, dtype)
    head = new_head(read_size(backbone, inside), len(categories), HEAD_SEED).to(device)
    head.load_state_dict(load_file(str(args.run / HEAD_FILE)))
    views = [view for path in args.views for view in load_views(path) if view.split == args.split]
    held = (
        [view for view in load_views(args.calibration_views) if view.mode == NO_THINKING]
        if args.calibration_views
        else []
    )
    capture = decision_capture(backbone, place, inside)
    try:
        odds = module_odds(backbone, capture, head, views)
        held_odds = module_odds(backbone, capture, head, held)
    finally:
        capture.close()
    records = load_all_records(args.data)
    mapping = json.loads(args.categories.read_text(encoding="utf-8"))
    temperature = args.temperature or fitted_temperature(held_odds, view_labels(held, records, mapping, categories))
    fire = chances(odds, temperature)
    truths = rule_in_play(views, records, mapping, categories)
    teacher = load_teacher(args.teacher, categories)
    out = args.out or args.run
    out.mkdir(parents=True, exist_ok=True)
    with (out / FIRE_FILE).open("w", encoding="utf-8") as handle:
        for view_id, values in fire.items():
            handle.write(json.dumps({"view_id": view_id, "fire": dict(zip(categories, values, strict=True))}) + "\n")
    found = {
        "threshold": PLAY_THRESHOLD,
        "read_inside": inside,
        "temperature": temperature,
        "calibration_views": len(held),
        "sets": scores_by_set(fire, teacher, truths),
        "sets_without_temperature": scores_by_set(chances(odds, 1.0), teacher, truths),
    }
    (out / DECISIONS_FILE).write_text(json.dumps(found, indent=2), encoding="utf-8")
    print(json.dumps(found), flush=True)


if __name__ == "__main__":
    main()
