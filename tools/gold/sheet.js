"use strict";
(function (Gold) {
  const el = (...args) => Gold.el(...args);

  function toolRows(sheet, app) {
    return el("div", { className: "stack" },
      sheet.tools.map((tool, i) => el("div", { className: "row-inline" },
        Gold.textInput(tool.name, (v) => { tool.name = v.trim(); app.changed(); }, "nome, ex.: ring_signal_bell"),
        Gold.textInput(tool.description, (v) => { tool.description = v; app.changed(); }, "o que ela faz, ex.: toca o sino de aviso do farol"),
        el("button", { className: "small danger", textContent: "remover", on: { click: () => { sheet.tools.splice(i, 1); app.changed(true); } } }))),
      el("button", { className: "small", textContent: "+ ferramenta", on: { click: () => { sheet.tools.push({ name: "", description: "" }); app.changed(true); } } }));
  }

  function kindChooser(rule, app) {
    return el("div", { className: "cards" }, Gold.KINDS.map(([key, label, hint]) =>
      el("button", {
        className: "card-choice" + (rule.kind === key ? " picked" : ""),
        on: { click: () => { rule.kind = key; app.changed(true); } },
      }, el("strong", { textContent: label }), el("span", { textContent: hint }))));
  }

  function looksAtChooser(rule, app) {
    return el("div", { className: "cards" }, Gold.LOOKS_AT.map(([key, label, hint]) =>
      el("button", {
        className: "card-choice" + (rule.looks_at === key ? " picked" : ""),
        on: { click: () => { rule.looks_at = key; app.changed(true); } },
      }, el("strong", { textContent: label }), el("span", { textContent: hint }))));
  }

  function choiceOptions(sheet, rule, app) {
    const fromTools = el("button", {
      className: "small", textContent: "usar as ferramentas da ficha",
      title: "Cria uma opção para cada ferramenta e mantém a opção 'nenhuma'",
      on: { click: () => {
        const kept = rule.options.filter((o) => Gold.C.NULL_KEYS.includes(o.key));
        rule.options = sheet.tools.filter((t) => t.name).map((t) => ({ key: t.name, when: t.description }));
        rule.options.push(...(kept.length ? kept : [{ key: "none", when: "nada a fazer agora" }]));
        app.changed(true);
      } },
    });
    return el("div", { className: "stack" },
      el("p", { className: "hint", textContent: "Cada opção tem um nome curto e quando ela vale. Mantenha uma opção 'none' para quando nada deve acontecer: ela é a resposta padrão." }),
      rule.options.map((option, j) => el("div", { className: "row-inline" },
        Gold.textInput(option.key, (v) => { option.key = v.trim(); app.changed(); }, "nome, ex.: ring_signal_bell"),
        Gold.textInput(option.when, (v) => { option.when = v; app.changed(); }, "quando escolher esta opção"),
        el("button", { className: "small danger", textContent: "×", title: "remover opção", on: { click: () => { rule.options.splice(j, 1); app.changed(true); } } }))),
      el("div", { className: "row-inline" },
        el("button", { className: "small", textContent: "+ opção", on: { click: () => { rule.options.push({ key: "", when: "" }); app.changed(true); } } }),
        sheet.tools.length ? fromTools : null));
  }

  function scaleLevels(rule, app) {
    return el("div", { className: "stack" },
      el("p", { className: "hint", textContent: "Do menor para o maior. O nível 0 costuma ser 'nada de errado'." }),
      rule.levels.map((level, j) => el("div", { className: "row-inline" },
        el("span", { className: "level", textContent: String(j) }),
        Gold.textInput(level, (v) => { rule.levels[j] = v; app.changed(); }, "o que esse nível significa"),
        el("button", { className: "small danger", textContent: "×", on: { click: () => { rule.levels.splice(j, 1); app.changed(true); } } }))),
      el("button", { className: "small", textContent: "+ nível", on: { click: () => { rule.levels.push(""); app.changed(true); } } }));
  }

  function ruleCard(sheet, rule, i, app) {
    const body = rule.kind === "yes_no"
      ? el("div", { className: "two" },
        Gold.field("Conta como SIM quando... (opcional)", null, Gold.textInput(rule.yes, (v) => { rule.yes = v; app.changed(); }, "ex.: pede, insiste ou narra que o NPC tem outro nome")),
        Gold.field("Conta como NÃO quando... (opcional)", null, Gold.textInput(rule.no, (v) => { rule.no = v; app.changed(); }, "ex.: só pergunta quem ele é")))
      : rule.kind === "choice" ? choiceOptions(sheet, rule, app) : scaleLevels(rule, app);
    return el("div", { className: "rule-card" },
      el("div", { className: "rule-head" },
        el("strong", { textContent: `Regra ${i + 1}` }),
        el("span", { className: "muted", textContent: rule.id ? `id: ${rule.id}` : "" }),
        el("span", { className: "spacer" }),
        el("button", { className: "small danger", textContent: "remover regra", on: { click: () => { if (confirm("Remover esta regra? As respostas dela nas conversas somem.")) { sheet.rules.splice(i, 1); app.changed(true); } } } })),
      Gold.field("O que a regra pergunta?", "Escreva como você explicaria para uma pessoa. O sistema faz essa pergunta a cada mensagem.",
        el("textarea", {
          value: rule.question, rows: 2, placeholder: "ex.: O jogador está tentando fazer o NPC aceitar outro nome?",
          on: { input: (e) => { rule.question = e.target.value; app.changed(); }, blur: () => { Gold.ensureRuleId(sheet, rule); app.changed(true); } },
        })),
      Gold.field("Que tipo de resposta?", null, kindChooser(rule, app)),
      body,
      Gold.field("Ela olha para", null, looksAtChooser(rule, app)),
      Gold.field("Que tipo de regra é?", Gold.hintOf(Gold.CATEGORIES, rule.category),
        Gold.choiceSelect(Gold.CATEGORIES.map(([k, l]) => [k, l]), rule.category, (v) => { rule.category = v; app.changed(true); })),
      Gold.field("Frase curta que o NPC sabe sobre a regra (opcional)",
        "Vai no system prompt para o modelo saber que a regra existe. Ex.: 'Você só atende pelo nome Maren Vell.' Não escreva a ação nem palavras proibidas. Se deixar vazio, o pipeline escreve uma.",
        Gold.textInput(rule.base, (v) => { rule.base = v; app.changed(); }, "uma frase curta")));
  }

  Gold.renderSheet = function renderSheet(container, sheet, app) {
    const problems = Gold.sheetProblems(sheet);
    container.replaceChildren(
      el("div", { className: "sheet" },
        el("div", { className: "sheet-top" },
          el("h2", { textContent: sheet.npc ? `Ficha: ${sheet.npc}` : "Ficha nova" }),
          el("span", { className: "spacer" }),
          el("button", { className: "danger", textContent: "apagar ficha", on: { click: () => app.deleteSheet(sheet.id) } }),
          el("button", { className: "primary", textContent: "Ir para as conversas", on: { click: () => app.showChat() } })),
        el("p", { className: "hint big", textContent: "A ficha descreve o NPC, as ferramentas e as regras. Você escreve uma vez e usa em várias conversas." }),
        problems.length ? el("ul", { className: "problems" }, problems.map((p) => el("li", { textContent: p }))) : null,
        el("section", {},
          el("h3", { textContent: "1. Quem é o NPC" }),
          Gold.field("Nome do NPC", null, Gold.textInput(sheet.npc, (v) => { sheet.npc = v; app.changed(); }, "ex.: Maren Vell")),
          Gold.field("Personalidade, história e jeito de falar", "É o começo do system prompt: o que o modelo sabe sobre quem ele é.",
            Gold.textArea(sheet.persona, (v) => { sheet.persona = v; app.changed(); }, "ex.: Você é Maren Vell, a guardiã do farol...", 8)),
          Gold.field("Cena e estado do jogo", "Onde a conversa acontece e o estado atual: hora, flags, inventário. Vai logo depois da personalidade.",
            Gold.textArea(sheet.scene, (v) => { sheet.scene = v; app.changed(); }, "ex.: Anoitecer, neblina subindo, um navio à vista.", 3)),
          Gold.field("Idioma em que as regras estão escritas", null,
            Gold.choiceSelect(Gold.LANGUAGES, sheet.language, (v) => { sheet.language = v; app.changed(); }))),
        el("section", {},
          el("h3", { textContent: "2. Ferramentas que o NPC pode chamar" }),
          el("p", { className: "hint", textContent: "O nome é como o modelo chama a ferramenta (só letras, números e _). A descrição diz o que ela faz. Pode deixar vazio se o NPC não usa ferramentas." }),
          toolRows(sheet, app)),
        el("section", {},
          el("h3", { textContent: "3. Regras" }),
          el("p", { className: "hint", textContent: "Cada regra é uma pergunta que o sistema faz a cada mensagem. Nas conversas, você vai marcar a resposta certa de cada uma." }),
          sheet.rules.map((rule, i) => ruleCard(sheet, rule, i, app)),
          el("button", { textContent: "+ regra", on: { click: () => { sheet.rules.push(Gold.blankRule()); app.changed(true); } } }))));
  };
})(window.Gold = window.Gold || {});
