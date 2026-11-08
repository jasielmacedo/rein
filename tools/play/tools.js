import { clear, el, show } from "./dom.js";
import { icon } from "./icons.js";

export function renderTools(strip, detail, character) {
  clear(strip);
  show(detail, false);
  let selected = null;
  for (const tool of character.tools) {
    const pill = el("button", { className: "tool-pill", attrs: { type: "button" } });
    pill.append(icon("tool"), el("span", { text: tool.name }));
    pill.addEventListener("click", () => {
      selected?.classList.remove("active");
      if (selected === pill) {
        selected = null;
        show(detail, false);
        return;
      }
      selected = pill;
      pill.classList.add("active");
      detail.textContent = `${tool.name}: ${tool.description}`;
      show(detail, true);
    });
    strip.append(pill);
  }
}
