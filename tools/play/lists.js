import { avatar } from "./avatar.js";
import { characterTags } from "./card.js";
import { el } from "./dom.js";
import { icon } from "./icons.js";

const DAY_MS = 86_400_000;

export function when(iso) {
  const date = new Date(iso);
  const today = new Date();
  const days = Math.floor((today.setHours(0, 0, 0, 0) - new Date(date).setHours(0, 0, 0, 0)) / DAY_MS);
  if (days <= 0) return date.toLocaleTimeString("pt-BR", { hour: "2-digit", minute: "2-digit" });
  if (days === 1) return "ontem";
  return date.toLocaleDateString("pt-BR", { day: "2-digit", month: "2-digit" });
}

export function characterRow(character, active, onOpen) {
  return el("li", {}, [
    el("button", { className: `row${active ? " active" : ""}`, on: { click: onOpen } }, [
      avatar(character.npc),
      el("span", { className: "row-main" }, [
        el("span", { className: "row-top" }, [
          el("span", { className: "row-name", text: character.npc }),
          character.conversations ? el("span", { className: "count", text: String(character.conversations) }) : null,
        ]),
        el("span", { className: "row-sub", text: character.persona }),
        el("span", { className: "row-tags" }, characterTags(character)),
      ]),
    ]),
  ]);
}

export function conversationRow(conversation, active, onOpen, onDelete) {
  const remove = el("button", { className: "icon-button subtle", attrs: { "aria-label": "Apagar conversa" }, on: { click: onDelete } });
  remove.append(icon("trash"));
  return el("li", { className: "with-action" }, [
    el("button", { className: `row${active ? " active" : ""}`, on: { click: onOpen } }, [
      el("span", { className: "row-main" }, [
        el("span", { className: "row-top" }, [
          el("span", { className: "row-name", text: conversation.first_text || "Conversa nova" }),
          el("span", { className: "row-time", text: when(conversation.updated_at) }),
        ]),
        el("span", { className: "row-sub", text: `${conversation.messages} mensagens` }),
      ]),
    ]),
    remove,
  ]);
}
