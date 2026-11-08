---
title: Rein-NPC playground
emoji: 🛒
colorFrom: yellow
colorTo: green
sdk: static
app_file: index.html
pinned: false
license: apache-2.0
short_description: Three game NPCs driven by Rein-NPC-4B, in your browser
models:
  - jmc0/Rein-NPC-4B
  - jmc0/Rein-NPC-4B-GGUF
---

# Rein-NPC playground

A small town with three characters: a shop keeper, a field medic and the captain of the gate. Walk up to one and
talk. The character is [Rein-NPC-4B](https://huggingface.co/jmc0/Rein-NPC-4B), a Qwen3.5-4B with Rein's decision
model inside its own file, running in your browser through [wllama](https://github.com/ngxson/wllama) (llama.cpp in
WebAssembly and WebGPU). The same model writes the lines and, before each line, answers the character's rules: does
this turn call for a tool, and which one? Is someone rewriting the character's name? Is a fact of the world under
attack? The page shows the decisions under every reply and applies them the way a game would: the tool call is
prefilled, tool calls are vetoed when none is wanted, the rule's base is reminded when a rule fires, and the tool's
result changes the town (gold, stock, health, the gate, the guards).

The model (3.07 GB at Q4_K_M, four files) is downloaded once from
[`jmc0/Rein-NPC-4B-GGUF`](https://huggingface.co/jmc0/Rein-NPC-4B-GGUF) into the browser's cache and runs on your
machine; nothing you type leaves it. WebGPU (Chrome, Edge) is the fast path; without it the CPU build runs, slowly.
The page lets you delete the cached files and download them again.

Source: `tools/playground/` in the Rein repository. The decision protocol and the acting rule are documented on the
model card.

## Development

- Run locally: `python -m http.server 8790` inside `tools/playground/`, then open http://127.0.0.1:8790/. The page
  loads wllama and the Jinja renderer from jsDelivr and the model shards from the GGUF repository, so it needs the
  network; nothing is served from a backend of ours.
- The model files are `web/Rein-NPC-4B-Q4_K_M-0000N-of-00004.gguf` in `jmc0/Rein-NPC-4B-GGUF`, the Q4_K_M file split
  into four pieces under 1 GB with `llama-gguf-split --split-max-size 1G` (wllama needs files under 2 GB). The first
  shard is named in `js/config.js`; wllama fetches the rest.
- Publish: `python -m rein.export.publish_space --public` from the repository root (defaults to this folder and to
  the Space `jmc0/rein-npc-playground`).
- Layout: `js/config.js` holds every constant; `js/rein.js` the decision protocol; `js/chat.js` the chat prompt,
  generation and tool-call parsing; `js/npcs.js` the three sheets and their game effects; `js/game.js` the town;
  `js/model.js` download, cache and loading; `js/ui.js` the panels; `js/main.js` the wiring.
- Art: `assets/` (CC0, see `assets/LICENSE-assets.txt`).
