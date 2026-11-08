# Rein-r1 without PyTorch, and inside the forward pass (research, 2026-11-06)

The question (user, 2026-11-06): Rein-r1 is a separate model and it is ready. Can it run inside the forward pass, or
in some form that works next to the Qwen that talks, in llama.cpp and the other runtimes, given that its current
implementation needs PyTorch?

A research agent read the source of llama.cpp (b8967, the version the project pins, and b12111), vLLM (v0.27.1 and
v0.29.0) and Ollama (v0.40.15), plus their docs, issues and release notes. Tags: [V] read in source or docs,
[I] inferred, [U] not checked. The checkouts sat in the session's scratchpad (`src-llamacpp`, `src-vllm`,
`src-ollama`) and may be gone; the versions above are enough to fetch them again.

Checked again by hand (2026-11-06): llama.cpp b12111 has `tools/server/server-decision.cpp` and registers
`/v1/systemone` (`tools/server/server.cpp:199`); its converter maps `KevModel` to the `lev` architecture
(`conversion/__init__.py:166`, `conversion/lev.py:151`); Ollama's client and benchmark call `/v1/systemone`
(`api/client.go:449`, `cmd/bench/decision.go:129`).

## What Rein-r1 is

Qwen3.5-4B Base, a LoRA, and a set-pointer head that reads the final norm of block 18 (`backbone.norm(h18)`,
`src/rein/model/prefix.py:197`). The head has 2,242,816 float32 parameters [V]. Each option is encoded in its own row
after the question, and the head attends over the set of options (`src/rein/model/head.py`), which gives exact
option-order invariance (DR-002) and lets options interact (DR-003). It reaches 0.97 to 0.98 on unseen rules.

## Short answer

It cannot run inside the generator's forward pass: different weights (Base plus LoRA against the post-trained
generator), its own input layout, and a head no runtime computes. It can run without PyTorch today as a second model,
and llama.cpp and Ollama now ship a native decision API.

## Options, most feasible first

### 1. Rein-r1 unchanged: GGUF backbone, embeddings out, head in numpy

- `rein.export.merge` already writes the merged 18-block `Qwen3_5TextModel`. llama.cpp b12111 converts that class
  [V `conversion/qwen.py:693`]; with b8967 the architecture likely needs renaming to `Qwen3_5ForCausalLM` [I].
- llama.cpp's qwen35 graph returns `output_norm` of the last block as its embeddings [V `src/models/qwen35.cpp:76-79`],
  the same tensor the head reads. Serve with `llama-server --embeddings --pooling last` and `embd_normalize: -1` [V],
  send token ids from Rein's own encoder (the tokenizer needs no torch), and run the head in numpy.
- Runs in llama.cpp (CPU, CUDA, Metal; bf16, Q8_0, Q4_K_M) and vLLM (`--convert embed`,
  `--pooler-config.task token_embed`, `use_activation=false` [V v0.27.1 docs]; vLLM already runs ColQwen3_5, a pooling
  model on the hybrid architecture [V]; the head could live in a vLLM IO-processor plugin [V]). Not Ollama: its
  `/api/embed` pools and L2-normalizes [V `server/routes.go:1352`]. MLX would need a port; mlx-lm has `qwen3_5` [V].
- Latency: hybrid models get no prefix reuse on embedding requests (llama-server makes checkpoints only for
  completions [V]; vLLM turns prefix caching off for hybrid pooling models [V `vllm/config/model.py`]), so each
  question reads the state once per option plus one. A small program on libllama (about 150 lines) keeps "state once,
  one row per option": `llama_memory_seq_cp` copies the recurrent state when a sequence forks
  [V `llama-memory-recurrent.cpp`]. One llama-server in router mode can host it next to the generator [V].
- Reuse 100%, no training. Risk: drift at Q8 and Q4. Effort: one to two days.

### 2. Through the System One decision API (native in llama.cpp and Ollama)

- llama.cpp added `/v1/systemone` on 2026-10-02 (https://github.com/ggml-org/llama.cpp/pull/29818); b12111 serves 16
  decision-model types [V `tools/server/server-decision.cpp`]. Ollama 0.40.15 bundles b12111 and passes `/v1/systemone`
  for GGUF models to llama-server [V `routes.go`]; its MLX engine also runs Kev [V `mlxrunner/model/kev`]. The project
  pins b8967, which predates the API.
- Kev (https://huggingface.co/jaredpalmer/kev-4b): Qwen3.5-4B-Base plus a LoRA (r16, alpha 32, on attention, MLP and
  DeltaNet) plus a pointer head, using Rein's delimiter tokens [V].
- Rein-r1 does not fit Kev's format as is: Kev puts all options in one row, reads q at the final `<|fim_suffix|>` and
  k at each `<|box_end|>` through affine maps (`classifier.out_proj`), scored as dot / sqrt(d) / T
  [V `conversion/lev.py`, `server-context.cpp:2573`], and receives the state as flattened text with special tokens
  escaped [V]. Rein isolates each option in its own row, reads the question before the options, and uses LayerNorm
  plus set attention.
- 2a. Retrain Rein in Kev's layout and head: same data, LoRA and recipe (r1 trained in 1 h 54 min on an H100 [V]),
  converted with llama.cpp's KevModel converter; cutting at block 18 needs a custom conversion step [I]. It gives up
  DR-002's exact order invariance and DR-003's interaction between options. No vLLM (neither System One nor Kev in
  v0.27.1 or v0.29.0 [V]).
- 2b. Contribute a "rein" decision type to llama.cpp: keeps Rein-r1 exact; the server already shares the state prefix
  across tasks and reads embeddings at marker tokens [V]. An upstream code change; Ollama would inherit it.

### 3. Inside the forward pass: option A with Rein-r1 as the teacher

- Use Rein-r1's soft labels for the module's decision read (A2), and let it label many more and longer conversations
  (the late failures on Ilsa's script were put down to short training conversations).
- Optionally, a decision turn whose answer is the option letter, read through logprobs plus a grammar, which
  llama.cpp, vLLM and Ollama all expose [V]; this is how openjev, lev and granite-switch work [V `common/common.h`].
- Already runs in every runtime (A1, group 4). Cost about one A2 rerun (US$3.60) plus the labeling [I]. Four inserted
  layers may still fall short of 0.979 [I].

### 4. A decision aLoRA on the generator

- llama-server applies an aLoRA per request and keeps the base model's cache up to the invocation tokens
  [V b8967 `server-common.cpp`]; the reply is untouched. Precedent: Granite Switch, served as text and decisions
  (https://github.com/ggml-org/llama.cpp/pull/30676), through a Granite-only architecture.
- vLLM has per-request LoRA but no aLoRA [V], so every decision is a full prefill; Ollama has no per-request adapters
  [V]. `convert_lora_to_gguf` will likely fail on a LoRA on DeltaNet `out_proj` because of the V-head reorder (16 K
  heads against 32 V heads) [I]. Needs a new verbalizer training; accuracy unknown.

### 5. Not possible without changing the runtimes

- Two models cannot share one forward pass: each architecture has one residual stream, and RMSNorm couples any
  widened stream.
- Extra tensors in a GGUF fail to load ("wrong number of tensors") [V]; the optional qwen35 `cls_out` in b12111 only
  replaces the embedding output [V].
- MTP and draft-model slots only propose tokens that the main model then checks. llama-server sets control vectors
  only at startup [V]. In b8967 `--pooling rank` has no qwen35 head [V].

## Recommendation and the cheapest check

Option 1 now: it is Rein-r1 itself, without PyTorch, next to the talking Qwen in llama.cpp and vLLM. Option 3 stays the
forward-pass track. Options 2a or 2b if Ollama support and a standard API matter more than Rein's exact behavior.

Cheapest experiment, no training, about US$1 to 2 [I]: export r1 with `rein.export.merge`, convert with b12111 to bf16,
Q8_0 and Q4_K_M, serve on an L4 with `--embeddings --pooling last`, run the numpy head on the 912 development records,
and compare with the PyTorch decider by the export's own thresholds (largest gap at most 0.02, agreement at least
0.995), plus accuracy and time per decision.
