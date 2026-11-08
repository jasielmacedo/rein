# Compute log: Rein-0.8B ablation runs

All runs: Modal, **1x H100 80GB per run**, one container per run. Runs launched together trained **in parallel on
separate GPUs** (no sharing). Same data (mix-v1, 22,864 train records, 1,431 batches per epoch, 2 epochs,
2,862 optimizer steps), same seed and hyperparameters; only the readout block and the kernel set differ.
Timings come from the trainer's `elapsed_s`, measured from the start of the training loop, so they leave out
the gate tests (about 2 min per launch) and model loading. Raw logs: `runs/logs/`.

## Runs

| Run | Readout block | Blocks executed | DeltaNet short conv | Parallel with | Outcome |
|---|---|---|---|---|---|
| L11 r1 | 11 | 11 | PyTorch fallback | L14 r1, L24 r1 | finished, 2,860 steps in 2,907 s |
| L14 r1 | 14 | 14 | PyTorch fallback | L11 r1, L24 r1 | container lost at step 2,340 (2,828 s) |
| L24 r1 | 24 | 24 | PyTorch fallback | L11 r1, L14 r1 | container lost at step 1,260 (2,637 s), restarted from 0 |
| L14 r4 | 14 | 14 | `causal-conv1d` 1.7.0 | L24 r4 | running |
| L24 r4 | 24 | 24 | `causal-conv1d` 1.7.0 | L14 r4 | running |

r2 and r3 were stopped by hand during warmup and are not reported.

## Elapsed seconds at each step

| Step | L11 r1 | L14 r1 | L14 r4 | L24 r1 | L24 r4 |
|---|---|---|---|---|---|
| 20 | 377 | 358 | 372 | 427 | 387 |
| 200 | 732 | 717 | 733 | 903 | 869 |
| 600 | 1,149 | 1,182 | 1,192 | 1,598 | 1,541 |
| 1,000 | 1,507 | 1,599 | 1,601 | 2,247 | |
| 1,400 | 1,829 | 1,976 | 1,973 | | |
| 2,000 | 2,319 | 2,547 | 2,536 | | |
| 2,800 | 2,866 | | | | |

## Readings

- **Warmup.** The first 20 steps take 6 to 7 minutes in every run. This is Triton autotuning in the fla kernels
  and has no relation to the readout block. From the next runs on, the Triton cache lives on the volume, so this
  should drop.
- **Steady state (steps 1,000 to 2,000).**
  - L11: about 0.81 s/step.
  - L14: about 0.94 s/step.
  - L24 (steps 200 to 600): about 1.7 s/step.
  - Cost grows with the number of blocks executed. A fixed per-step part (data, head, option rows, optimizer)
    keeps it below proportional.
- **causal-conv1d versus the PyTorch fallback** (same block, same data):
  - L14, steps 200 to 2,000: 1,803 s against 1,830 s, about 1.5% faster.
  - L24, steps 200 to 600: 672 s against 695 s, about 3% faster.
  - The gain is small and grows with depth: the short convolution is not the bottleneck at 0.8B.
  - The kernel stays in the image because it costs nothing at run time.
  - Before the 4B runs, profile one step to find where the time actually goes.
- **Parallel runs.** Each run has its own GPU, so parallel launches do not slow each other. Blocks 11 and 14 in
  r1 ran side by side, and their curves differ only by the depth gap.

## GPU-hours

| Run | GPU-hours (training loop plus about 4 min of gates and loading) |
|---|---|
| L11 r1 | about 0.88 |
| L14 r1 (lost) | about 0.85 |
| L24 r1 (lost, then restarted and stopped) | about 0.85 plus the restart |
| L14 r4 | to fill in at the end |
| L24 r4 | to fill in at the end |
