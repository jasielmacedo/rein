# rein-0.8b-r1-L11

- **Purpose:** readout-block ablation, arm B (block 11 of 24), on the rule-disjoint mix-v1.
- **Recipe:**
  - Qwen3.5-0.8B-Base truncated after block 11.
  - LoRA r16/a32 on attention, DeltaNet and MLP, with the set-aware head.
  - CE + 0.25 Brier, 2 epochs, 2,862 optimizer steps, seed 0. Calibrated temperature 0.955.
- **Hardware:** Modal H100, fla 0.6.0 kernels, torch 2.13. Training took about 48 min.
- **Labels:** teacher labels (two-framing judge + tiebreak). Human gold set pending.

| Slice (test, rules never seen) | n | Accuracy | ECE |
|---|---|---|---|
| All questions | 28,102 | 0.951 | 0.002 |
| Target rule | 5,078 | 0.956 | 0.011 |
| hard_negative | 309 | 0.854 | 0.051 |
| stale_trigger | 452 | 0.969 | 0.027 |
| narrated_action (watcher) | 505 | 0.935 | 0.043 |
| Held-out domain (noir) | 1,464 | 0.945 | 0.012 |
| Held-out category (mandatory_phrase) | 305 | 0.889 | 0.051 |
| Long conversations (15+ turns) | 1,254 | 0.970 | 0.018 |

- **Gate:** at the 0.85 gate the model decides 89% of test questions, with 98.5% accuracy on those.
- **Latency:** single-request p50 127 ms, about 24 ms per question (H100, encode included, unoptimized).
- **Weakest slice:** hard_negative, which is the target for directed data.
- Full slice tables: `rein-0.8b-r1-L11-dev.json` and `rein-0.8b-r1-L11-test.json`.
