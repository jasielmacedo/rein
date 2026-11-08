# Related work notes

Facts gathered on 2026-11-04 from the sources linked. Re-check versions before citing.

## Typed decision models

**Jev / System One (TypeSafe AI).** Closed hosted decision model, announced 2026-09-15. API
`POST /v1/systemone` with a state and typed questions (`choice`, `noul` yes/no, `score`). Claims an
"RLCD" training method. No weights or paper. Docs: https://docs.typesafe.ai/api

**Kev (Jared Palmer).** Open reimplementation, Apache-2.0: https://github.com/jaredpalmer/kev
- It is based on Archer Hume's probing write-up "Jev's Architecture Unmasked": https://archerhume.com/posts/jevs-architecture-unmasked
- Kev 1.0 family: 0.8B, 4B and 9B on Qwen3.5-Base, plus 27B on Qwen3.8-27B (full fine-tune).
  Models: https://huggingface.co/jaredpalmer/kev-4b
- Pointer head: `q = Linear(d, 256)`, `k = Linear(d, 256)`, `z_i = k(h_opt_i) . q(h_decide) / sqrt(256)`,
  softmax per question. Questions share the state and cannot read each other.
- Input delimiters reuse Qwen special tokens: `<|fim_prefix|>` (state), `<|fim_middle|>` (question),
  `<|box_start|>`/`<|box_end|>` (options), `<|fim_suffix|>` (decide). User text that imitates them is rewritten.
- On hybrid Qwen3.5 backbones, option isolation is refused because DeltaNet cannot honor the block mask,
  and every question runs as its own row over a shared state prefix (`kev/shared_prefix.py`).
- Kev-4B recipe:
  - Stage 1: decision-v7, 2 epochs, LoRA r16/a32, lr 5e-5 one-cycle, effective batch 8, bf16, about 56 min on one H100.
  - Then three more stages (dates, documents, skills) at lr 2e-5 with replay.
  - Single temperature 2.41. Validated state length 8,192 tokens.
- README results (development / test): Kev-4B new sources 0.817 / 0.838, trained sources 0.873 / 0.865.
- Serving, checked 2026-11-08: install from source (`uv sync --extra serve`; the PyPI `kev` is unrelated), then
  `python -m kev.serve --run jaredpalmer/kev-4b --port 8009` (`--host`, `--device`); `POST /v1/systemone`, no
  `/health` route (404) but the server answers; eight requests in flight were fine on an H100. The state is computed
  once and its cache reused for every question row. A second model and a second Python server next to the game's
  chat model; 0.8B, 4B and 9B are LoRA adapters on Qwen3.5-Base, the 27B is 51 GB of full weights.

**TinyJev-4B (AnkitAI).** https://huggingface.co/AnkitAI/TinyJev-4B. Code (MIT): https://github.com/ankit-aglawe/tinyjev
- v1 (HF branch `v1`):
  - Recipe: Qwen3-4B-Base plus Kev's pointer head (fp32 head, 5.25 MB), LoRA r16/a32 merged into the
    weights, trained with Kev's runner (commit 2855ba2): lr 5e-5, 2 epochs on decision-v7 train
    (12,576 records), one H100, 45 min. No temperature scaling (T = 1.0).
  - OD-500 results: 474/500 (0.948), ECE 0.022, Brier 0.071; the 0.85 gate handles 87% of cases at 99.1%
    accuracy.
  - decision-v7 dev: 0.859 accuracy, ECE 0.064.
  - Known weakness: statement-form yes/no checks.
- v2 (main, 2026-10-01): Qwen3.5-4B without a pointer head (softmax over option-letter logits), LoRA r8,
  1 epoch, about 2 h on an A100, 37,840 Tev1 records + 1,269 ContractNLI records.
- OD-500 benchmark: https://github.com/ankit-aglawe/tinyjev/tree/main/benchmarks/opendecision
- Serving, checked 2026-11-08 (pip `tinyjev[torch]`, `tinyjev/serve.py` on main): `tinyjev serve` is Python's
  `http.server.HTTPServer` with a `BaseHTTPRequestHandler`, one request at a time, HTTP/1.1 keep-alive, the default
  listen backlog of 5; eight clients in flight got "all connection attempts failed" on our first run, one in flight
  works. Body limit 2 MB, any `Origin` header refused (no browser calls), no authentication, loopback by default.
  The README documents `tinyjev serve` for TinyJev-0.6B only; TinyJev-4B v2 (4.2B, 4.5 GB) runs through Ollama
  0.35.1 or newer (`ollama pull parable/tinyjev`, about 2 s a case on an M1), and v1 (4.0B, 8 GB) in-process on MLX
  (Mac). A game shipping it on Linux or Windows with a GPU has a second server process to run, either Python
  single-threaded or Ollama, beside the chat model's own server.

**Other open decision models (2025-2026), to position against:**
- Strands Decider 2B (Qwen3.5-2B-Base, LoRA + head): https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v21
- Tev1-4B (Together, generates one option letter): https://huggingface.co/togethercomputer/Tev1-4B-experimental
- H2O-Lightning-4B (label-token probability): https://huggingface.co/h2oai/h2o-lightning-4b
- Laya (ModernBERT-large, a per-option `[MASK]` head): https://huggingface.co/convaiinnovations/laya
- Clef (Cloudflare 27B, a joint schema head): https://huggingface.co/Cloudflare/clef
- pplx-decider v1.1 27B (a noncausal readout head, gated): https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b

## Policy and rule adherence over dialogues (closest to our domain)

- DynaGuard 1.7B/4B/8B: PASS/FAIL against user-written policies over dialogues, arXiv 2509.02563.
  https://huggingface.co/tomg-group-umd/DynaGuard-8B
- gpt-oss-safeguard 20B: a bring-your-own-policy reasoning classifier, Apache-2.0. https://huggingface.co/openai/gpt-oss-safeguard-20b

## Benchmarks

- JevBench (MIT, `/v1/systemone` adapter): https://github.com/fstandhartinger/jevbench
- OD-500 (above), Kev transfer-v4 locked test, breadth-v1.
- Survey with counter-evidence. It reports option-order flips of 7 to 11.5% and small trained classifiers
  matching decision models on safety tasks: https://redreamality.com/blog/open-decision-models-2026-survey-benchmarks/

## Data sources

- `jaredpalmer/kev-suites`: 295 MB, **no license stated**.
  - decision-v7 has train 12,576 / development 1,204 / calibration 968 / test 1,176 records.
  - Train composition: 10,000 public records from 10 datasets, 896 policy minimal pairs, and 1,680
    compositional rule records.
  - Record shape: `{state, questions: {id: {type, instructions, criteria, label}}}`.
  - https://huggingface.co/datasets/jaredpalmer/kev-suites
- Banking77 (`PolyAI/banking77`): CC BY 4.0, 77 intents, train 10,003 / test 3,080.
- CLINC150 (`clinc/clinc_oos`): CC BY 3.0, 150 intents + out-of-scope.
- MASSIVE v1.1 (`AmazonScience/massive`): CC BY 4.0, 52 locales, 60 intents; Portuguese is pt-PT only.
  The HF loader script fails on `datasets>=4`; use the tarball at
  https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz

## Base model facts (Qwen3.5)

- Repos `Qwen/Qwen3.5-0.8B(-Base)` and `Qwen/Qwen3.5-4B(-Base)`, Apache-2.0, released 2026-03-02.
- The checkpoints are vision-language models (`Qwen3_5ForConditionalGeneration`). `Qwen3_5ForCausalLM`
  loads the text model only.
- 0.8B: 24 layers, hidden 1,024. 4B: 32 layers, hidden 2,560. Both use 3 DeltaNet layers per full-attention
  layer, vocabulary 248,320 and tied embeddings.
- DeltaNet modules: `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`, `out_proj` (plus `conv1d`, which LoRA cannot target).
- Fast kernels: `flash-linear-attention` and `causal-conv1d`. Without them transformers silently falls
  back to slow torch. Support arrived in transformers v5.2.0.
- The chat template defaults to thinking on the 4B and to non-thinking on the 0.8B.
- SGLang notes that on hybrid models, probabilities shifted by about 0.07 between cold and prefix-cached
  requests. This is relevant to Rein's session cache, so we test cache equivalence explicitly.

## Composing two models in one network (gathered 2026-11-07)

- **FACADE** (Wallace et al., arXiv 2010.05419): merges two transformers layer by layer into one wider model whose
  output is the concatenation of both; linear layers become block-diagonal, the layer norm is applied per block.
  Built to hide one model inside another. https://arxiv.org/abs/2010.05419
- **CALM, LLM Augmented LLMs** (Bansal et al., ICLR 2024, arXiv 2401.02412): a frozen anchor model and a frozen
  augmenting model joined by learned cross-attention over chosen layers; neither model's weights change.
  https://arxiv.org/abs/2401.02412
- **Ladder Side-Tuning** (Sung, Cho, Bansal, NeurIPS 2022, arXiv 2206.06522): a small side network fed by the frozen
  backbone's intermediate activations; only the side network trains. https://arxiv.org/abs/2206.06522
- **Dragon Hatchling** (arXiv 2509.26507): two fine-tuned copies merged into one wider model without training, on its
  own sparse architecture; a little training restores the loss. https://arxiv.org/abs/2509.26507
- **mergekit MoE ("frankenMoE")**: dense models become the experts of a Mixtral-format MoE, attention from one donor,
  the router initialized from hidden states of prompts. https://huggingface.co/blog/mlabonne/frankenmoe
- **Chat Vector** (Huang et al., ACL 2024, arXiv 2310.04799): the weight difference between a chat model and its base,
  added to another model of the same base, transfers the ability without training. https://arxiv.org/abs/2310.04799
