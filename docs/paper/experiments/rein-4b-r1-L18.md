# rein-4b-r1-L18

- **Purpose:** first Rein-4B run, readout at block 18 of 32 (chosen by `sweep-4b.jsonl`).
- **Recipe:**
  - Qwen3.5-4B-Base truncated after block 18.
  - LoRA r16/a32 on attention, DeltaNet and MLP, with the set-aware head.
  - CE + 0.25 Brier, 2 epochs, 2,862 optimizer steps, seed 0. Calibrated temperature 0.955.
- **Hardware:** Modal H100, fla 0.6.0 + causal-conv1d 1.7.0, torch 2.13, batch shape bucketing 64/16/32.
  Training took 6,822 s (about 1 h 54 min), plus dev and test evaluation.
- **Labels:** mix-v1-audited (27B auditor, label-blind, two thinking votes to correct).

## Test, rules never seen (audited labels)

| Slice | n | Accuracy | ECE |
|---|---|---|---|
| All questions | 28,433 | 0.968 | 0.003 |
| Target rule | 5,179 | 0.973 | 0.008 |
| hard_negative | 324 | 0.944 | 0.037 |
| narrated_action (watcher) | 525 | 0.964 | 0.025 |
| Held-out domain | 1,491 | 0.969 | 0.014 |
| Held-out category (mandatory_phrase) | 308 | 0.938 | 0.026 |
| Long conversations | 1,276 | 0.973 | 0.014 |
| tone_drift (score) | 94 | 0.777 | 0.086 |

Dev: accuracy 0.964, ECE 0.005, coverage at gate 0.918 with 0.990 accuracy at gate.

## Against the 0.8B on the same items and the same audited labels

The 28,102 test questions all three runs share, each scored against its audited label (argmax accuracy). The
middle column is `rein-0.8b-r1-L24-audited`: same recipe as the original L24 (block 24, no batch padding, seed 0),
trained on mix-v1-audited. It separates the effect of cleaner labels from the effect of size.

| Slice | n | 0.8B L24 (original labels) | 0.8B L24 (audited labels) | 4B L18 (audited labels) |
|---|---|---|---|---|
| All questions | 28,102 | 0.956 | 0.959 | 0.971 |
| Target rule | 5,078 | 0.962 | 0.962 | 0.979 |
| hard_negative | 309 | 0.890 | 0.916 | 0.961 |
| Held-out category | 305 | 0.912 | 0.921 | 0.941 |
| narrated_action (watcher) | 505 | 0.937 | 0.945 | 0.968 |
| npc_secret | 287 | 0.906 | 0.902 | 0.958 |
| tone_drift (score) | 83 | 0.819 | 0.783 | 0.795 |
| Labels the audit corrected | 567 | 0.469 | 0.563 | 0.635 |

Decomposition of the gain from the original 0.8B to the 4B:
- All questions, +1.5 points: +0.25 from the labels and +1.2 from size.
- Hard negatives, +7.1 points: +2.6 from the labels and +4.5 from size.

Most of the gain is the larger backbone, but cleaner labels alone already lift the hard negatives. Dev for the
audited 0.8B: accuracy 0.954, ECE 0.008, temperature 1.023.
