# Experiment cards

Every training run writes `provenance.json` (run id, git commit and dirty flag, config and its hash, data
manifest hashes, base model and revision, seeds, library versions, hardware, timing) and `metrics.json` (dev
metrics per epoch, the fitted temperature, the training history) into its run folder. Evaluation writes
`eval-<split>/report.md`, `report.json` and per-question rows (`rows_rein.jsonl`). Runs are pulled to
`runs/checkpoints/<run>` and versioned with Git LFS.

The cards here are written by hand from those files: purpose, what changed against the previous run, headline
metrics, the slices that moved, surprises, cost, and the decision the run led to.

| Card | What it covers |
|---|---|
| `rein-0.8b-r1-L11.md` | 0.8B readout ablation, block 11 |
| `rein-4b-r1-L18.md` | Rein-4B, and the labels-versus-size comparison with both 0.8B runs |
| `compute-0.8b.md` | Training speed with and without causal-conv1d, parallel runs |
| `sweep-0.8b.jsonl`, `sweep-4b.jsonl` | Frozen-probe accuracy per readout block |
| `e2e-0.8b.md` | End to end with the 0.8B chat model: tools, character locks, the tavern session |
| `e2e-4b.md` | Control-layer guards, the PII session, the chat model x decider matrix |
| `soft-4b.md` | The decider as 64 soft tokens in the 4B Instruct's spare rows: test table against every decider, the tied-head leak, the untied export, the GGUF set |
| `cmp-4b-instruct-soft64-quiet-{dev,test,gold}.json` | Eval reports of that run (test, development, the user's hand-written gold set) |
| `e2e-soft-4b.md` | End to end in Rein's format on one GGUF (none vs rein), the human play material replayed through the decider, the chat in Rein mode |
| `baselines-4b.md` | Generic decision models (Kev, TinyJev) and the untrained generator on the same test items, with paired deltas against Rein-r1 4B |
| `baseline-kev-4b-test.json`, `baseline-kev-0.8b-test.json`, `baseline-tinyjev-0.6b-test.json`, `zero-4b-instruct-labels-test.json` | Those reports (the baseline ones hold the Rein-r1 4B and Rein-NPC rows too) |

All accuracies are argmax against the label, but two question sets appear. The evaluation reports score all
28,430 test questions. The labels-versus-size table in `rein-4b-r1-L18.md` scores only the 28,102 questions that
the original 0.8B evaluation also scored. The other 331 are questions the audit corrected and the original
evaluation did not score; they are the hardest items (0.66 accuracy for the audited 0.8B), so the shared-set
numbers sit about 0.3 points higher.
