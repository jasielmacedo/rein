import { avatar } from "./avatar.js";
import { el } from "./dom.js";
import { icon } from "./icons.js";

const SHAPES = { sparse: "prompt enxuto", medium: "prompt médio", dense: "prompt denso" };
const LANGUAGES = { en: "inglês", "pt-BR": "português" };

export function shapeName(shape) {
  return SHAPES[shape] || shape;
}

export function tag(text) {
  return el("span", { className: "tag", text });
}

export function characterTags(character) {
  return [
    tag(shapeName(character.shape)),
    tag(`${character.tools.length} ferramentas`),
    tag(LANGUAGES[character.language] || character.language),
  ];
}

export function subtitle(character) {
  return `${shapeName(character.shape)} · ${character.tools.length} ferramentas`;
}

function section(title, body) {
  return el("section", { className: "sheet-section" }, [el("h3", { text: title }), body]);
}

function toolCard(tool) {
  return el("div", { className: "tool-card" }, [el("code", { text: tool.name }), el("span", { text: tool.description })]);
}

export function sheetContent(character, onClose) {
  const close = el("button", { className: "icon-button", attrs: { "aria-label": "Fechar" }, on: { click: onClose } });
  close.append(icon("plus", "icon rotate"));
  return [
    el("header", { className: "sheet-head" }, [
      avatar(character.npc, "large"),
      el("div", { className: "sheet-title" }, [
        el("h2", { text: character.npc, attrs: { id: "sheet-name" } }),
        el("div", { className: "tags" }, characterTags(character)),
      ]),
      close,
    ]),
    section("Ferramentas", el("div", { className: "tools" }, character.tools.map(toolCard))),
    section("System prompt: persona", el("p", { className: "prose", text: character.persona })),
    character.scene ? section("Cena", el("p", { className: "prose muted", text: character.scene })) : null,
    section("Regras", el("ol", { className: "rules" }, character.rules.map((rule) => el("li", { text: rule })))),
  ].filter(Boolean);
}
