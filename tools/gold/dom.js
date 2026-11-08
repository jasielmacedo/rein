"use strict";
(function (Gold) {
  const TOAST_MS = 2200;

  Gold.el = function el(tag, props = {}, ...children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(props)) {
      if (value === undefined || value === null) continue;
      if (key === "on") for (const [event, fn] of Object.entries(value)) node.addEventListener(event, fn);
      else if (key === "dataset") Object.assign(node.dataset, value);
      else if (key in node) node[key] = value;
      else node.setAttribute(key, value);
    }
    for (const child of children.flat(Infinity)) if (child !== null && child !== undefined && child !== false) node.append(child);
    return node;
  };

  Gold.$ = (selector) => document.querySelector(selector);

  Gold.fill = function fill(node, ...children) {
    node.replaceChildren(...children.flat(Infinity).filter((child) => child !== null && child !== undefined && child !== false));
  };

  Gold.field = function field(label, hint, control) {
    const el = Gold.el;
    return el("div", { className: "field" },
      el("label", { textContent: label }),
      hint ? el("p", { className: "hint", textContent: hint }) : null,
      control);
  };

  Gold.textInput = function textInput(value, onInput, placeholder = "") {
    return Gold.el("input", { type: "text", value: value || "", placeholder, on: { input: (e) => onInput(e.target.value) } });
  };

  Gold.textArea = function textArea(value, onInput, placeholder = "", rows = 3) {
    return Gold.el("textarea", { value: value || "", placeholder, rows, on: { input: (e) => onInput(e.target.value) } });
  };

  Gold.choiceSelect = function choiceSelect(pairs, value, onChange) {
    return Gold.el("select", { on: { change: (e) => onChange(e.target.value) } },
      pairs.map(([key, label]) => Gold.el("option", { value: key, selected: key === value, textContent: label })));
  };

  Gold.toast = function toast(text) {
    const node = Gold.$("#toast");
    node.textContent = text;
    node.classList.add("show");
    clearTimeout(node._timer);
    node._timer = setTimeout(() => node.classList.remove("show"), TOAST_MS);
  };
})(window.Gold = window.Gold || {});
