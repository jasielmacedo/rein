// The page's panels: the model manager, the conversation, and Rein's decisions under every reply.

import { GGUF_CARD_URL, MODEL_BYTES, MODEL_CARD_URL, MODEL_LABEL } from "./config.js";
import { PHASES } from "./model.js";

const GIGABYTE = 1024 ** 3;
const PERCENT = 100;

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function gigabytes(bytes) {
  return `${(bytes / GIGABYTE).toFixed(2)} GB`;
}

export class ModelPanel {
  constructor(root, actions) {
    this.root = root;
    this.actions = actions;
    this.status = element("p", "status");
    this.bar = element("div", "bar");
    this.fill = element("div", "fill");
    this.bar.append(this.fill);
    this.buttons = element("div", "buttons");
    const title = element("h2", null, MODEL_LABEL);
    const links = element("p", "links");
    const card = element("a", null, "model card");
    card.href = MODEL_CARD_URL;
    card.target = "_blank";
    const gguf = element("a", null, "GGUF files");
    gguf.href = GGUF_CARD_URL;
    gguf.target = "_blank";
    links.append(card, " · ", gguf);
    root.append(title, this.status, this.bar, this.buttons, links);
  }

  button(label, handler, kind = "") {
    const node = element("button", kind, label);
    node.addEventListener("click", handler);
    return node;
  }

  render(store) {
    const { phase, progress, cachedBytes } = store;
    const total = progress.total || MODEL_BYTES;
    const share = Math.min(1, progress.loaded / total);
    this.fill.style.width = `${Math.round(share * PERCENT)}%`;
    this.bar.classList.toggle("hidden", phase !== PHASES.downloading);
    this.buttons.replaceChildren();
    if (phase === PHASES.idle) {
      const cached = store.isCached;
      this.status.textContent = cached
        ? `Model in this browser's cache (${gigabytes(cachedBytes)}). Load it to play. ${store.backendLabel()}.`
        : `The model is downloaded once into this browser's cache (${gigabytes(MODEL_BYTES)}, four files) and ` +
          `runs on your machine. ${store.backendLabel()}.`;
      this.buttons.append(this.button(cached ? "Load model" : "Download and load", this.actions.load, "primary"));
      if (cached) this.buttons.append(this.button("Delete from cache", this.actions.remove));
      if (cached) this.buttons.append(this.button("Download again", this.actions.redownload));
    } else if (phase === PHASES.downloading) {
      this.status.textContent = `Downloading ${gigabytes(progress.loaded)} of ${gigabytes(total)} (${Math.round(share * PERCENT)}%)...`;
      this.buttons.append(this.button("Cancel", this.actions.cancel));
    } else if (phase === PHASES.loading) {
      this.status.textContent = `Loading the model into ${store.webgpu ? "the GPU" : "memory"}... (a minute on most machines)`;
    } else if (phase === PHASES.ready) {
      this.status.textContent = `Ready, ${store.backendLabel()}. Click a character and talk.`;
      this.buttons.append(this.button("Delete from cache", this.actions.remove));
      this.buttons.append(this.button("Download again", this.actions.redownload));
    } else if (phase === PHASES.deleting) {
      this.status.textContent = "Deleting the cached files...";
    } else if (phase === PHASES.error) {
      this.status.textContent = `Something failed: ${store.error?.message ?? store.error}. Try again, or delete the cache and download again.`;
      this.buttons.append(this.button("Try again", this.actions.load, "primary"));
      this.buttons.append(this.button("Delete from cache", this.actions.remove));
    }
  }
}

export class ChatPanel {
  constructor(root, onSend) {
    this.root = root;
    this.title = element("h2", null, "Pick a character");
    this.subtitle = element("p", "subtitle", "");
    this.log = element("div", "log");
    this.form = element("form", "composer");
    this.input = element("input");
    this.input.type = "text";
    this.input.placeholder = "Say something...";
    this.input.maxLength = 500;
    this.input.autocomplete = "off";
    this.send = element("button", "primary", "Send");
    this.send.type = "submit";
    this.form.append(this.input, this.send);
    this.form.addEventListener("submit", (event) => {
      event.preventDefault();
      const text = this.input.value.trim();
      if (!text) return;
      this.input.value = "";
      onSend(text);
    });
    root.append(this.title, this.subtitle, this.log, this.form);
    this.setEnabled(false);
  }

  setEnabled(enabled) {
    this.input.disabled = !enabled;
    this.send.disabled = !enabled;
  }

  open(npc) {
    this.title.textContent = npc.npc;
    this.subtitle.textContent = `${npc.title} · ${npc.language}`;
    this.log.replaceChildren();
  }

  bubble(kind, text) {
    const node = element("div", `bubble ${kind}`, text);
    this.log.append(node);
    this.log.scrollTop = this.log.scrollHeight;
    return node;
  }

  player(text) {
    return this.bubble("player", text);
  }

  npc(text) {
    return this.bubble("npc", text);
  }

  note(text) {
    return this.bubble("note", text);
  }

  decisions(plan) {
    const node = element("div", "decisions");
    for (const [rule, probabilities] of Object.entries(plan.answers)) {
      const [best, probability] = Object.entries(probabilities).reduce((top, entry) => (entry[1] > top[1] ? entry : top));
      node.append(element("span", "chip", `${rule}: ${best} ${Math.round(probability * PERCENT)}%`));
    }
    const acted = [];
    if (plan.tool) acted.push(`opened ${plan.tool}`);
    if (plan.veto) acted.push("tool calls vetoed");
    if (plan.reminders.length) acted.push(`reminded: ${plan.reminders.join(" / ")}`);
    if (acted.length) node.append(element("span", "acted", acted.join(" · ")));
    this.log.append(node);
    this.log.scrollTop = this.log.scrollHeight;
  }
}

export class StatePanel {
  constructor(root) {
    this.root = root;
  }

  render(state) {
    const inventory = state.player.inventory.length ? state.player.inventory.join(", ") : "nothing";
    this.root.replaceChildren(
      element("span", "chip", `gold ${state.player.gold}`),
      element("span", "chip", `health ${state.player.health}`),
      element("span", "chip", `armor ${state.player.armor}`),
      element("span", "chip", `pack: ${inventory}`),
      element("span", "chip", `gate ${state.gate}`),
      element("span", "chip", `guards ${state.guards}`),
    );
  }
}
