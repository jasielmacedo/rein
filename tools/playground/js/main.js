// Wires the town, the model and the panels: click a character, talk, decide, act, reply.

import { chatPrompt, generate, parseToolCall, visibleText } from "./chat.js";
import { JINJA_ESM, MAX_CALLS_PER_TURN } from "./config.js";
import { Game } from "./game.js";
import { ModelStore } from "./model.js";
import { NPCS, initialState, sheetFor } from "./npcs.js";
import { AFTER_TOOL_RESULT, decide, plan } from "./rein.js";
import { ChatPanel, ModelPanel, StatePanel } from "./ui.js";

const canvas = document.getElementById("world");
const state = initialState();
const game = new Game(canvas, NPCS);
const statePanel = new StatePanel(document.getElementById("state"));
const histories = Object.fromEntries(NPCS.map((npc) => [npc.id, []]));
let current = null;
let busy = false;
let template = null;

const store = new ModelStore((model) => {
  modelPanel.render(model);
  chat.setEnabled(model.isReady && current !== null && !busy);
});
const modelPanel = new ModelPanel(document.getElementById("model"), {
  load: () => store.load(),
  cancel: () => store.cancel(),
  remove: () => store.deleteCache(),
  redownload: () => store.redownload(),
});
const chat = new ChatPanel(document.getElementById("chat"), (text) => talk(text));

async function loadTemplate() {
  if (template) return template;
  const { Template } = await import(JINJA_ESM);
  template = new Template(store.wllama.getChatTemplate());
  return template;
}

function openConversation(npc) {
  current = npc;
  game.select(npc);
  chat.open(npc);
  for (const message of histories[npc.id]) {
    if (message.role === "user") chat.player(message.text);
    else if (message.role === "assistant" && message.text) chat.npc(message.text);
    else if (message.role === "tool") chat.note(`${message.name} → ${message.text}`);
  }
  if (!histories[npc.id].length) chat.note(`You approach ${npc.npc}.`);
  chat.setEnabled(store.isReady && !busy);
  chat.input.focus();
}

async function reply(npc, sheet, history, actions) {
  const prompt = chatPrompt(await loadTemplate(), sheet, history, actions.reminders);
  const bubble = chat.npc("...");
  const raw = await generate(store.wllama, prompt, actions, (partial) => {
    bubble.textContent = visibleText(partial) || "...";
  });
  const call = parseToolCall(raw);
  const text = visibleText(raw);
  bubble.textContent = text || (call ? `*${sheet.npc} acts*` : "...");
  if (text) game.say(npc.id, text);
  history.push({ role: "assistant", text, toolCalls: call ? [call] : [] });
  return call;
}

async function talk(text) {
  if (!current || busy || !store.isReady) return;
  busy = true;
  chat.setEnabled(false);
  const npc = current;
  const history = histories[npc.id];
  try {
    history.push({ role: "user", text });
    chat.player(text);
    const sheet = sheetFor(npc, state);
    const actions = plan(sheet, await decide(store.wllama, sheet, history));
    chat.decisions(actions);
    let call = await reply(npc, sheet, history, actions);
    for (let calls = 0; call && calls < MAX_CALLS_PER_TURN; calls += 1) {
      const effect = npc.effects[call.name];
      const result = effect ? effect(call.arguments, state) : JSON.stringify({ status: "unknown_tool", tool: call.name });
      statePanel.render(state);
      chat.note(`${call.name}(${JSON.stringify(call.arguments)}) → ${result}`);
      history.push({ role: "tool", name: call.name, text: result });
      call = await reply(npc, sheetFor(npc, state), history, AFTER_TOOL_RESULT);
    }
  } catch (error) {
    chat.note(`The model did not answer: ${error?.message ?? error}`);
  } finally {
    busy = false;
    chat.setEnabled(store.isReady && current !== null);
  }
}

function frame() {
  game.update();
  game.draw(state);
  requestAnimationFrame(frame);
}

function canvasPoint(event) {
  const rect = canvas.getBoundingClientRect();
  const scale = canvas.width / rect.width;
  return [(event.clientX - rect.left) * scale, (event.clientY - rect.top) * scale];
}

canvas.addEventListener("click", (event) => {
  const npc = game.npcAt(...canvasPoint(event));
  if (npc) openConversation(npc);
});
canvas.addEventListener("pointermove", (event) => {
  const npc = game.npcAt(...canvasPoint(event));
  game.hover(npc);
  canvas.style.cursor = npc ? "pointer" : "default";
});

const characters = document.getElementById("characters");
for (const npc of NPCS) {
  const button = document.createElement("button");
  button.textContent = `${npc.emoji} ${npc.npc}`;
  button.addEventListener("click", () => openConversation(npc));
  characters.append(button);
}

// The CPU build wants SharedArrayBuffer for its threads, which a static host cannot grant through headers; the
// service worker adds them and the page reloads once under it. WebGPU needs none of this.
const RELOADED_FLAG = "coi-reloaded";
async function isolateForThreads() {
  if (store.webgpu || window.crossOriginIsolated || !("serviceWorker" in navigator)) return;
  if (sessionStorage.getItem(RELOADED_FLAG)) return;
  await navigator.serviceWorker.register("coi-serviceworker.js");
  await navigator.serviceWorker.ready;
  sessionStorage.setItem(RELOADED_FLAG, "1");
  if (!navigator.serviceWorker.controller) window.location.reload();
}

statePanel.render(state);
game.load().catch(() => {});
frame();
store
  .init()
  .then(() => isolateForThreads().catch(() => {}))
  .catch((error) => {
    store.setPhase("error", error);
  });
window.reinPlayground = { store, game, state, histories, loadTemplate };
