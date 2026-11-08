import { el } from "./dom.js";

const HUES = 360;
const HASH_STEP = 31;
const SKIP = new Set(["the", "of", "do", "da", "de", "dos", "das", "mrs.", "mr.", "seu", "tio"]);

function hue(text) {
  let value = 0;
  for (const char of text) value = (value * HASH_STEP + char.codePointAt(0)) % HUES;
  return value;
}

function initials(name) {
  const words = name.replace(/['"]/g, "").split(/\s+/).filter((word) => word && !SKIP.has(word.toLowerCase()));
  return words.slice(0, 2).map((word) => word[0].toUpperCase()).join("") || "?";
}

export function avatar(name, size = "") {
  const node = el("span", { className: `avatar ${size}`.trim(), text: initials(name) });
  node.style.setProperty("--hue", String(hue(name)));
  return node;
}
