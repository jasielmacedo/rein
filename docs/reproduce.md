# Reproducing Rein

Everything here ran on rented single GPUs through [Modal](https://modal.com). The whole project, the two closed lines
included, cost US$370 of cloud compute; each step below gives its own time and cost.

## Setup

```bash
uv sync --all-extras          # Python 3.12, the package and its tools
cp .env.example .env          # HF_TOKEN, and the teacher and auditor endpoints (docs/endpoints.md)
uv run pytest                 # CPU or GPU; the suite picks CUDA when it is there
```

Training and the large evaluations run on Modal (`pip install modal`, `modal setup`). The apps keep data and runs on
two volumes, `rein-data` and `rein-hf-cache`, created on first use.

## Data

| part | how | where |
|---|---|---|
| synthetic conversations, rule sheets, labels | `python -m rein.datagen.run` with the teacher (`configs/datagen/`) | released as [`jmc0/rein-decisions`](https://huggingface.co/datasets/jmc0/rein-decisions) |
| label audit | `python -m rein.datagen.audit` with the auditor | in the released records (`meta.audit`) |
| converted public sets | `python -m rein.public.build`: Banking77, CLINC150, MASSIVE (as typed choice questions) and Kev's decision-v7 | not redistributed; rebuilt from their sources (see `NOTICE`) |

The training mix (`mix-v1-audited`) is 22,864 records: 16,924 synthetic, 3,440 converted from Banking77, CLINC150 and
MASSIVE, and 2,500 from Kev's decision-v7. Splits are by rule: scenarios are grouped, near-duplicate rules are blocked
across splits, and one domain and one category are held out of training entirely. Upload a built mix to the volume
with `modal run modal_apps/train_app.py::main --action upload --data-dir data/mix-v1-audited`.

## Models

| model | config | train | export | time on one H100 |
|---|---|---|---|---|
| [Rein-r1-0.8B](https://huggingface.co/jmc0/rein-r1-0.8b) | `configs/train/rein-0.8b-L24-audited.yaml` | `modal run modal_apps/train_app.py::main --action train --config <config>` | `modal run modal_apps/train_app.py::export --run <run> --repo <repo> --card configs/export/rein-0.8b-card.md` | 1 h 19 |
| [Rein-r1-4B](https://huggingface.co/jmc0/rein-r1-4b) | `configs/train/rein-4b-L18.yaml` | same | same, with `rein-4b-card.md` | 1 h 54 |
| [Rein-NPC-4B](https://huggingface.co/jmc0/Rein-NPC-4B) | `configs/train/cmp-4b-instruct-soft64-quiet.yaml` | same | `modal run modal_apps/soft_app.py::build --run <run> --untie`, then `python -m rein.export.publish_soft` | 3 h 07 (about US$15) |

`soft_app.py::build` writes the trained rows into a plain copy of Qwen3.5-4B (`rein.export.soft_merge --untie`: the
base table also becomes the output head, so the chat stays exactly Qwen's), converts it with llama.cpp's
`convert_hf_to_gguf.py` and quantizes it with `llama-quantize` (bf16, Q8_0, Q4_K_M). The browser shards come from
`llama-gguf-split --split-max-size 1G`.

## Evaluation

| what | command | output |
|---|---|---|
| decisions on the test (unseen rules) | `modal run modal_apps/train_app.py::evaluate --target rein --run runs/<run> --data mix-v1-audited/test.jsonl --out <dir>` | `report.json`, `report.md`, per-question rows |
| generic baselines on the same items | `... ::evaluate --target kev-4b` (or `kev-0.8b`, `tinyjev-0.6b`) `--rows rein=<rows>,soft4b=<rows>` | the same report with paired bootstrap deltas |
| the untrained generator in the lettered layout | `python -m rein.eval.zero_shot --base Qwen/Qwen3.5-4B --out <run>`, then `--target rein` | |
| soft tokens in the chat | `modal run modal_apps/soft_app.py::check_silence --run <run> [--untie]` | soft-token mass over held-out chat positions |
| GGUF decisions against PyTorch | `modal run modal_apps/soft_app.py::check_agreement --run <run> [--untie]` | same option, probability gaps |
| end to end on test-v1 | `modal run modal_apps/soft_app.py::play_sheets --run <run> --untie`, then `python -m rein.sheets.e2e_judge --sheets configs/sheets/test-v1/*.yaml --sessions <dir> --out <file>` | judged sessions, rates per agent |

The numbers of every run are in `docs/paper/experiments/` (one card per experiment, with the JSON reports beside
them). The cards cite the project's lab notebook and decision records (DR-xxx), which are not part of this release;
every number they quote comes from the reports next to them.

## The playground

`tools/playground/` is a static page: `python -m http.server 8790` inside it serves it locally, and
`python -m rein.export.publish_space --public` publishes it as the Space
[`jmc0/rein-npc-playground`](https://huggingface.co/spaces/jmc0/rein-npc-playground). It loads wllama and the Jinja
renderer from jsDelivr and the model shards from the GGUF repository.
