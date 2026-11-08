import { api } from "./api.js";
import { avatar } from "./avatar.js";
import { el } from "./dom.js";
import { icon } from "./icons.js";

const LABEL_LIMIT = 500;

function time(iso) {
  const date = iso ? new Date(iso) : new Date();
  return date.toLocaleTimeString("pt-BR", { hour: "2-digit", minute: "2-digit" });
}

function callText(call) {
  const args = Object.entries(call.arguments || {}).map(([name, value]) => `${name}: ${value}`);
  return `${call.name}(${args.join(", ")})`;
}

function withIcon(name, className, text) {
  const node = el("div", { className });
  node.append(icon(name), el("span", { text }));
  return node;
}

function flagLine(label) {
  return withIcon("flag", "flag-line", label);
}

function labelEditor(message, holder, onError) {
  const input = el("input", {
    attrs: { type: "text", maxlength: String(LABEL_LIMIT), placeholder: "O que está errado nessa resposta?" },
  });
  input.value = message.label || "";
  const editor = el("form", { className: "label-editor" });
  const close = () => editor.remove();
  const save = async (value) => {
    try {
      const saved = await api.label(message.id, value);
      message.label = saved.label;
      holder.querySelector(".flag-line")?.remove();
      if (message.label) holder.append(flagLine(message.label));
      close();
    } catch (error) {
      onError(error.message);
    }
  };
  editor.addEventListener("submit", (event) => {
    event.preventDefault();
    save(input.value.trim());
  });
  editor.append(
    el("span", { className: "editor-title", text: "Rotular esta resposta" }),
    input,
    el("div", { className: "label-actions" }, [
      message.label
        ? el("button", { className: "chip-button danger", text: "Remover", attrs: { type: "button" }, on: { click: () => save("") } })
        : null,
      el("button", { className: "chip-button", text: "Cancelar", attrs: { type: "button" }, on: { click: close } }),
      el("button", { className: "chip-button primary", text: "Salvar", attrs: { type: "submit" } }),
    ]),
  );
  return { editor, input };
}

function decisionsText(decisions) {
  const parts = [];
  for (const [rule, options] of Object.entries(decisions.decisions || {})) {
    const [best, probability] = Object.entries(options).sort((a, b) => b[1] - a[1])[0];
    parts.push(`${rule}: ${best} ${Math.round(probability * 100)}%`);
  }
  if (decisions.veto_tools) parts.push("tools vetoed");
  if (decisions.reminders?.length) parts.push(`reminded: ${decisions.reminders.join(" / ")}`);
  return parts.join(" · ");
}

function npcMessage(message, onError) {
  const bubble = el("div", { className: "bubble" });
  if (message.text) bubble.append(el("div", { className: "text", text: message.text }));
  if (message.tool_call) bubble.append(withIcon("tool", "tool-chip", callText(message.tool_call)));
  if (message.decisions) bubble.append(el("div", { className: "rein-line", text: decisionsText(message.decisions) }));
  bubble.append(el("div", { className: "meta", text: time(message.created_at) }));
  const holder = el("div", { className: "msg npc" });
  const more = el("button", { className: "more", attrs: { type: "button", "aria-label": "Rotular resposta" } });
  more.append(icon("more"));
  more.addEventListener("click", () => {
    const open = holder.querySelector(".label-editor");
    if (open) {
      open.remove();
      return;
    }
    const { editor, input } = labelEditor(message, holder, onError);
    holder.append(editor);
    input.focus();
    editor.scrollIntoView({ block: "nearest", behavior: "smooth" });
  });
  holder.append(el("div", { className: "msg-row" }, [bubble, more]));
  if (message.label) holder.append(flagLine(message.label));
  return holder;
}

export function messageNode(message, onError) {
  if (message.role === "tool") return withIcon("check", "system-note", `${message.tool_name}: ${message.text}`);
  if (message.role === "assistant") return npcMessage(message, onError);
  return el("div", { className: "msg user" }, [
    el("div", { className: "bubble" }, [
      el("div", { className: "text", text: message.text }),
      el("div", { className: "meta", text: time(message.created_at) }),
    ]),
  ]);
}

export function typingNode() {
  return el("div", { className: "msg npc" }, [
    el("div", { className: "bubble typing" }, [el("span", { className: "dot" }), el("span", { className: "dot" }), el("span", { className: "dot" })]),
  ]);
}

export function emptyChat(character, text) {
  return el("div", { className: "empty-state" }, [
    character ? avatar(character.npc, "large") : null,
    el("strong", { text: character ? `Diga oi para ${character.npc}` : "Rein Play" }),
    el("span", { text }),
  ]);
}
