import { api } from "./api.js";
import { avatar } from "./avatar.js";
import { sheetContent, subtitle } from "./card.js";
import { emptyChat, messageNode, typingNode } from "./chat.js";
import { clear, el, show } from "./dom.js";
import { icon } from "./icons.js";
import { characterRow, conversationRow } from "./lists.js";
import { renderTools } from "./tools.js";

const $ = (id) => document.getElementById(id);
const WIDE = window.matchMedia("(min-width: 900px)");
const state = { characters: [], character: null, conversation: null, sending: false, skipPop: false };

function showError(text) {
  $("error").textContent = text;
  show($("error"), Boolean(text));
}

function setView(view) {
  document.body.dataset.view = view;
  if (view !== "chat") document.body.dataset.left = view;
}

function scrollToEnd() {
  $("messages").scrollTo({ top: $("messages").scrollHeight });
}

function who(target, character) {
  clear(target);
  target.append(
    avatar(character.npc, "small"),
    el("span", { className: "who-text" }, [
      el("span", { className: "who-name", text: character.npc }),
      el("span", { className: "who-sub", text: `${subtitle(character)} · toque para ver a ficha` }),
    ]),
  );
}

function renderCharacters() {
  clear($("characters"));
  for (const character of state.characters) {
    const active = WIDE.matches && state.character?.id === character.id;
    $("characters").append(characterRow(character, active, () => go({ view: "threads", character: character.id })));
  }
}

async function renderConversations() {
  const list = await api.conversations(state.character.id);
  clear($("conversations"));
  if (!list.length) {
    $("conversations").append(
      el("li", { className: "list-empty" }, [
        el("strong", { text: "Nenhuma conversa ainda" }),
        el("span", { text: "Toque no + para começar." }),
      ]),
    );
  }
  for (const conversation of list) {
    const open = () => go({ view: "chat", character: state.character.id, conversation: conversation.id });
    $("conversations").append(
      conversationRow(conversation, state.conversation === conversation.id, open, () => removeConversation(conversation.id)),
    );
  }
  const card = state.characters.find((character) => character.id === state.character.id);
  if (card) card.conversations = list.length;
}

function showEmptyChat(text) {
  state.conversation = null;
  clear($("messages"));
  $("messages").append(emptyChat(state.character, text));
  show($("composer"), false);
}

async function showThreads(character) {
  state.character = character;
  who($("threads-who"), character);
  who($("chat-who"), character);
  show($("chat-who"), true);
  show($("chat-info"), true);
  renderTools($("tools-strip"), $("tool-detail"), character);
  show($("tools"), true);
  setView("threads");
  renderCharacters();
  showEmptyChat("Abra uma conversa ou comece uma nova.");
  await renderConversations();
}

async function showChat(character, conversationId) {
  if (state.character?.id !== character.id) await showThreads(character);
  const conversation = await api.conversation(conversationId);
  state.conversation = conversationId;
  clear($("messages"));
  if (!conversation.messages.length) $("messages").append(emptyChat(character, "Toque no nome lá em cima para ver a ficha."));
  for (const message of conversation.messages) $("messages").append(messageNode(message, showError));
  show($("composer"), true);
  setView("chat");
  scrollToEnd();
  await renderConversations();
  if (WIDE.matches) $("input").focus();
}

function showHome() {
  state.character = null;
  show($("chat-who"), false);
  show($("chat-info"), false);
  show($("tools"), false);
  setView("home");
  renderCharacters();
  clear($("messages"));
  $("messages").append(emptyChat(null, "Escolha um personagem para conversar."));
  show($("composer"), false);
}

async function restore(saved) {
  showError("");
  closeSheet(false);
  const character = state.characters.find((item) => item.id === saved?.character);
  try {
    if (!character) return showHome();
    if (saved.view === "chat") return await showChat(character, saved.conversation);
    return await showThreads(character);
  } catch (error) {
    showError(error.message);
  }
}

function go(next) {
  history.pushState(next, "");
  return restore(next);
}

async function newConversation() {
  try {
    const { id } = await api.start(state.character.id);
    await go({ view: "chat", character: state.character.id, conversation: id });
  } catch (error) {
    showError(error.message);
  }
}

async function removeConversation(id) {
  if (!window.confirm("Apagar esta conversa? Não dá para desfazer.")) return;
  try {
    await api.remove(id);
    if (state.conversation === id) showEmptyChat("Abra uma conversa ou comece uma nova.");
    await renderConversations();
  } catch (error) {
    showError(error.message);
  }
}

function openSheet() {
  if (!state.character) return;
  clear($("sheet-content"));
  $("sheet-content").append(...sheetContent(state.character, () => closeSheet(true)));
  show($("sheet"), true);
  history.pushState({ ...history.state, sheet: true }, "");
}

function closeSheet(viaHistory) {
  if ($("sheet").hidden) return;
  show($("sheet"), false);
  if (viaHistory && history.state?.sheet) {
    state.skipPop = true;
    history.back();
  }
}

function autoGrow() {
  const input = $("input");
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, window.innerHeight / 3)}px`;
}

async function send(event) {
  event.preventDefault();
  const text = $("input").value.trim();
  if (!text || state.sending || state.conversation === null) return;
  state.sending = true;
  showError("");
  $("send").disabled = true;
  $("input").value = "";
  autoGrow();
  $("messages").querySelector(".empty-state")?.remove();
  const pending = messageNode({ role: "user", text }, showError);
  const typing = typingNode();
  $("messages").append(pending, typing);
  scrollToEnd();
  try {
    const stored = await api.send(state.conversation, text);
    pending.remove();
    typing.remove();
    for (const message of stored) $("messages").append(messageNode(message, showError));
    renderConversations().catch((error) => showError(error.message));
  } catch (error) {
    pending.remove();
    typing.remove();
    $("input").value = text;
    autoGrow();
    showError(error.message);
  } finally {
    state.sending = false;
    $("send").disabled = false;
    scrollToEnd();
  }
}

function onKey(event) {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    $("composer").requestSubmit();
  }
}

function decorate() {
  $("threads-back").append(icon("back"));
  $("chat-back").append(icon("back"));
  $("threads-info").append(icon("info"));
  $("chat-info").append(icon("info"));
  $("new-conversation").append(icon("plus"));
  $("send").append(icon("send", "icon filled"));
}

function onPop(event) {
  if (state.skipPop) {
    state.skipPop = false;
    return;
  }
  if (!$("sheet").hidden) {
    closeSheet(false);
    return;
  }
  restore(event.state);
}

async function start() {
  decorate();
  $("threads-back").addEventListener("click", () => history.back());
  $("chat-back").addEventListener("click", () => history.back());
  for (const id of ["threads-who", "threads-info", "chat-who", "chat-info"]) $(id).addEventListener("click", openSheet);
  $("sheet-backdrop").addEventListener("click", () => closeSheet(true));
  document.addEventListener("keydown", (event) => event.key === "Escape" && closeSheet(true));
  $("new-conversation").addEventListener("click", newConversation);
  $("composer").addEventListener("submit", send);
  $("input").addEventListener("input", autoGrow);
  $("input").addEventListener("keydown", onKey);
  window.addEventListener("popstate", onPop);
  try {
    state.characters = await api.characters();
    history.replaceState({ view: "home" }, "");
    showHome();
  } catch (error) {
    showError(error.message);
  }
}

start();
