"use strict";
(function (Gold) {
  const C = {
    FORMAT: "rein-gold-project",
    VERSION: 1,
    STORAGE_KEY: "rein-gold-project-v1",
    ID_PATTERN: /^[A-Za-z0-9_.:\-]{1,64}$/,
    NULL_KEYS: ["none", "general", "no"],
    MAX_OPTION_TEXT: 600,
    MAX_QUESTION: 2000,
    SLUG_WORDS: 4,
    MIN_LEVELS: 2,
    MAX_LEVELS: 11,
    MIN_OPTIONS: 2,
  };
  Gold.C = C;

  Gold.KINDS = [
    ["yes_no", "Sim ou não", "A regra dispara ou não. Ex.: o jogador está tentando mudar o nome do NPC?"],
    ["choice", "Uma opção de uma lista", "Escolhe uma entre várias respostas. Ex.: qual ferramenta chamar agora, ou nenhuma."],
    ["scale", "Nota numa escala", "Uma nota de 0 até N. Ex.: o quanto a mensagem foge do tom do jogo."],
  ];

  Gold.LOOKS_AT = [
    ["player", "A mensagem do jogador", "Decide assim que a mensagem chega, antes de o NPC responder. Quase todas as regras são assim."],
    ["npc_reply", "A resposta do NPC", "Decide olhando o que o NPC escreveu. Ex.: o NPC contou que tocou o sino, mas não chamou a ferramenta?"],
  ];

  Gold.CATEGORIES = [
    ["tool_trigger", "Chamar ferramenta", "o NPC deve chamar uma ferramenta (tocar o sino, entregar um item...)"],
    ["narrated_action", "Ação narrada sem ferramenta", "o NPC descreve que fez algo, mas não chamou a ferramenta que faz"],
    ["locked_attribute", "Algo que não muda", "nome, idade, cargo: o jogador tenta mudar e o NPC não aceita"],
    ["mutable_attribute", "Algo que muda com condição", "pode mudar, mas só se uma condição do jogo for cumprida"],
    ["npc_secret", "Segredo", "informação que o NPC só revela em certas condições"],
    ["lore_consistency", "Coerência com a lore", "o jogador afirma algo que contradiz o mundo ou a história"],
    ["break_character", "Sair do personagem", "pedem para o NPC admitir que é uma IA ou sair do papel"],
    ["prompt_injection", "Injeção de prompt", "tentam dar ordens ao sistema ou trocar as regras"],
    ["business_limit", "Limite de negócio", "desconto, reembolso, preço ou crédito além do permitido"],
    ["age_rating", "Classificação etária", "conteúdo impróprio para a classificação do jogo"],
    ["format_rule", "Formato da resposta", "a resposta precisa seguir um formato"],
    ["mandatory_phrase", "Frase obrigatória", "o NPC precisa dizer uma frase específica em certo momento"],
    ["routing", "Encaminhamento", "para qual fluxo ou setor a conversa deve ir"],
    ["escalation", "Passar para um humano", "quando chamar um atendente ou um responsável"],
    ["tone_drift", "Tom", "o NPC está saindo do tom combinado"],
    ["pii", "Dados pessoais", "pedir ou expor dados pessoais"],
    ["other", "Outra", "nenhuma das anteriores"],
  ];

  Gold.FORMS = [
    ["auto", "Deixar o sistema decidir", "pedido direto quando alguma regra dispara, conversa normal quando nada dispara"],
    ["neutral", "Conversa normal", "nada de especial; serve para ver se as regras ficam quietas"],
    ["direct", "Pedido direto", "o jogador pede com todas as letras"],
    ["paraphrase", "Com outras palavras", "o mesmo pedido, sem as palavras óbvias"],
    ["slang", "Gíria ou informal", "gíria, abreviações, erros de digitação"],
    ["indirect", "Indireto", "insinua sem pedir com todas as letras"],
    ["narration_attack", "Ataque por narração", "usa *asteriscos* ou voz de narrador para forçar a mudança"],
    ["spread_over_turns", "Espalhado em vários turnos", "o pedido só fica completo somando as últimas mensagens"],
    ["early_reference", "Lembra algo do começo", "só faz sentido com algo dito bem antes (ex.: faz o que eu pedi no início)"],
    ["adversarial", "Tentando enganar", "hipóteses, troca de papéis, pressão ou falsa autoridade para contornar a regra"],
    ["hard_negative", "Parece, mas não é", "usa as palavras do gatilho com outra intenção; a regra não deve disparar"],
    ["stale_trigger", "Gatilho que já passou", "o gatilho aconteceu antes e já foi tratado; agora não deve disparar de novo"],
    ["co_trigger", "Duas regras juntas", "a mensagem dispara esta regra e outra ao mesmo tempo"],
  ];

  Gold.LANGUAGES = [
    ["pt-BR", "Português"], ["en", "Inglês"], ["pt-BR+en", "Português misturado com inglês"], ["es", "Espanhol"],
    ["fr", "Francês"], ["de", "Alemão"], ["it", "Italiano"], ["ja", "Japonês"], ["zh", "Chinês"], ["ko", "Coreano"],
    ["ru", "Russo"], ["ar", "Árabe"], ["hi", "Hindi"], ["other", "Outro"],
  ];

  Gold.SPEAKERS = { player: "Jogador", npc: "NPC", narration: "Narração" };

  Gold.labelOf = (pairs, key) => (pairs.find((pair) => pair[0] === key) || [key, key])[1];
  Gold.hintOf = (pairs, key) => (pairs.find((pair) => pair[0] === key) || [key, key, ""])[2];

  Gold.newProject = () => ({ format: C.FORMAT, version: C.VERSION, sheets: [], conversations: [] });

  Gold.uniqueId = function uniqueId(prefix, taken) {
    let n = 1;
    while (taken.has(`${prefix}-${n}`)) n++;
    return `${prefix}-${n}`;
  };

  Gold.slug = function slug(text, taken, fallback) {
    const words = (text || "").toLowerCase().normalize("NFD").replace(/[̀-ͯ]/g, "")
      .split(/[^a-z0-9]+/).filter(Boolean).slice(0, C.SLUG_WORDS);
    const base = (words.join("_") || fallback).slice(0, 48);
    if (!taken.has(base)) return base;
    return Gold.uniqueId(base, taken);
  };

  Gold.blankSheet = function blankSheet(project) {
    const id = Gold.uniqueId("ficha", new Set(project.sheets.map((s) => s.id)));
    return { id, npc: "", persona: "", scene: "", language: "pt-BR", tools: [], rules: [] };
  };

  Gold.blankRule = function blankRule() {
    return {
      id: "", question: "", kind: "yes_no", yes: "", no: "",
      options: [{ key: "none", when: "nada a fazer agora" }],
      levels: ["", ""], looks_at: "player", category: "other", base: "",
    };
  };

  Gold.blankConversation = function blankConversation(project, sheet) {
    const id = Gold.uniqueId("conversa", new Set(project.conversations.map((c) => c.id)));
    return { id, sheet: sheet.id, title: "", language: sheet.language || "pt-BR", messages: [] };
  };

  Gold.blankMessage = (from, text) => ({ from, name: "", text, tool_calls: [], answers: {}, tests: "", form: "auto", note: "", skip: false });

  Gold.optionKeys = function optionKeys(rule) {
    if (rule.kind === "yes_no") return ["no", "yes"];
    if (rule.kind === "choice") return rule.options.map((o) => o.key.trim());
    return rule.levels.map((_, i) => String(i));
  };

  Gold.defaultAnswer = function defaultAnswer(rule) {
    if (rule.kind === "yes_no") return "no";
    if (rule.kind === "choice") return Gold.optionKeys(rule).find((key) => C.NULL_KEYS.includes(key)) || null;
    return null;
  };

  Gold.answerOf = function answerOf(message, rule) {
    const stored = message.answers[rule.id];
    return Gold.optionKeys(rule).includes(stored) ? stored : Gold.defaultAnswer(rule);
  };

  Gold.fires = (rule, answer) => answer !== null && answer !== Gold.defaultAnswer(rule) && rule.kind !== "scale";

  Gold.sheetOf = (project, conversation) => project.sheets.find((s) => s.id === conversation.sheet) || null;

  Gold.rulesFor = function rulesFor(sheet, message) {
    if (!sheet) return [];
    const scope = message.from === "npc" ? "npc_reply" : "player";
    return sheet.rules.filter((rule) => rule.looks_at === scope && rule.id);
  };

  Gold.ruleTitle = (rule) => rule.question.trim() || "(regra sem pergunta)";

  Gold.answerLabel = function answerLabel(rule, key) {
    if (rule.kind === "yes_no") return key === "yes" ? "Dispara" : "Não dispara";
    if (rule.kind === "scale") return `${key}${rule.levels[Number(key)] ? ": " + rule.levels[Number(key)] : ""}`;
    if (C.NULL_KEYS.includes(key)) return "Nenhuma (nada a fazer)";
    return key;
  };

  Gold.sheetProblems = function sheetProblems(sheet) {
    const problems = [];
    if (!sheet.npc.trim()) problems.push("Dê um nome ao NPC.");
    if (!sheet.persona.trim()) problems.push("Escreva quem é o NPC.");
    const toolNames = new Set();
    sheet.tools.forEach((tool, i) => {
      const name = tool.name.trim();
      if (!C.ID_PATTERN.test(name)) problems.push(`Ferramenta ${i + 1}: use só letras, números e _ no nome (ex.: ring_signal_bell).`);
      if (toolNames.has(name)) problems.push(`Ferramenta ${name}: nome repetido.`);
      toolNames.add(name);
    });
    if (!sheet.rules.length) problems.push("Adicione pelo menos uma regra.");
    sheet.rules.forEach((rule, i) => {
      const name = `Regra ${i + 1}`;
      if (!rule.question.trim()) problems.push(`${name}: escreva o que a regra pergunta.`);
      if (rule.question.length > C.MAX_QUESTION) problems.push(`${name}: a pergunta passou de ${C.MAX_QUESTION} caracteres.`);
      if (rule.kind === "choice") {
        const keys = Gold.optionKeys(rule);
        if (keys.length < C.MIN_OPTIONS) problems.push(`${name}: uma lista precisa de pelo menos ${C.MIN_OPTIONS} opções.`);
        if (keys.some((key) => !key)) problems.push(`${name}: toda opção precisa de um nome.`);
        if (new Set(keys).size !== keys.length) problems.push(`${name}: há opções com o mesmo nome.`);
        if (rule.options.some((o) => o.when.length > C.MAX_OPTION_TEXT)) problems.push(`${name}: a descrição de uma opção passou de ${C.MAX_OPTION_TEXT} caracteres.`);
      }
      if (rule.kind === "scale" && (rule.levels.length < C.MIN_LEVELS || rule.levels.length > C.MAX_LEVELS || rule.levels.some((l) => !l.trim()))) {
        problems.push(`${name}: a escala precisa de ${C.MIN_LEVELS} a ${C.MAX_LEVELS} níveis, todos descritos.`);
      }
    });
    return problems;
  };

  Gold.messageProblems = function messageProblems(sheet, message) {
    if (message.skip) return [];
    return Gold.rulesFor(sheet, message)
      .filter((rule) => Gold.answerOf(message, rule) === null)
      .map((rule) => `Falta responder: ${Gold.ruleTitle(rule)}`);
  };

  Gold.isDecision = (sheet, message) => !message.skip && Gold.rulesFor(sheet, message).length > 0 && message.text.trim() !== "";

  Gold.summary = function summary(project) {
    let decisions = 0;
    let firing = 0;
    for (const conversation of project.conversations) {
      const sheet = Gold.sheetOf(project, conversation);
      for (const message of conversation.messages) {
        if (!Gold.isDecision(sheet, message)) continue;
        decisions++;
        if (Gold.rulesFor(sheet, message).some((rule) => Gold.fires(rule, Gold.answerOf(message, rule)))) firing++;
      }
    }
    return { decisions, firing };
  };

  Gold.ensureRuleId = function ensureRuleId(sheet, rule) {
    if (rule.id) return;
    const taken = new Set(sheet.rules.filter((r) => r !== rule).map((r) => r.id));
    rule.id = Gold.slug(rule.question, taken, "regra");
  };

  function asText(value) {
    return typeof value === "string" ? value : "";
  }

  function cleanSheet(raw) {
    return {
      id: asText(raw.id), npc: asText(raw.npc), persona: asText(raw.persona), scene: asText(raw.scene),
      language: asText(raw.language) || "pt-BR",
      tools: (Array.isArray(raw.tools) ? raw.tools : []).map((t) => ({ name: asText(t.name), description: asText(t.description) })),
      rules: (Array.isArray(raw.rules) ? raw.rules : []).map((r) => ({
        id: asText(r.id), question: asText(r.question),
        kind: ["yes_no", "choice", "scale"].includes(r.kind) ? r.kind : "yes_no",
        yes: asText(r.yes), no: asText(r.no),
        options: (Array.isArray(r.options) ? r.options : []).map((o) => ({ key: asText(o.key), when: asText(o.when) })),
        levels: (Array.isArray(r.levels) ? r.levels : ["", ""]).map(asText),
        looks_at: r.looks_at === "npc_reply" ? "npc_reply" : "player",
        category: asText(r.category) || "other", base: asText(r.base),
      })),
    };
  }

  function cleanConversation(raw) {
    return {
      id: asText(raw.id), sheet: asText(raw.sheet), title: asText(raw.title), language: asText(raw.language) || "pt-BR",
      messages: (Array.isArray(raw.messages) ? raw.messages : []).map((m) => ({
        from: ["player", "npc", "narration"].includes(m.from) ? m.from : "player",
        name: asText(m.name), text: asText(m.text),
        tool_calls: (Array.isArray(m.tool_calls) ? m.tool_calls : []).map(asText).filter(Boolean),
        answers: Object.fromEntries(Object.entries(m.answers && typeof m.answers === "object" ? m.answers : {}).map(([k, v]) => [asText(k), asText(v)])),
        tests: asText(m.tests), form: asText(m.form) || "auto", note: asText(m.note), skip: m.skip === true,
      })),
    };
  }

  Gold.parseProject = function parseProject(text) {
    const raw = JSON.parse(text);
    if (!raw || raw.format !== C.FORMAT) throw new Error("este arquivo não é um projeto de gold set");
    if (raw.version !== C.VERSION) throw new Error(`versão ${raw.version} não suportada`);
    return {
      format: C.FORMAT, version: C.VERSION,
      sheets: (Array.isArray(raw.sheets) ? raw.sheets : []).map(cleanSheet),
      conversations: (Array.isArray(raw.conversations) ? raw.conversations : []).map(cleanConversation),
    };
  };

  Gold.serialize = (project) => JSON.stringify(project, null, 2) + "\n";

  Gold.persist = function persist(state) {
    try {
      localStorage.setItem(C.STORAGE_KEY, JSON.stringify({ project: state.project, sheet: state.sheet, conversation: state.conversation }));
    } catch (_) { /* storage may be unavailable; the file save still works */ }
  };

  Gold.restore = function restore() {
    try {
      const saved = JSON.parse(localStorage.getItem(C.STORAGE_KEY) || "null");
      if (saved && saved.project) {
        return { project: Gold.parseProject(JSON.stringify(saved.project)), sheet: saved.sheet || null, conversation: saved.conversation || null };
      }
    } catch (_) { /* a broken draft starts a new project */ }
    return { project: Gold.newProject(), sheet: null, conversation: null };
  };
})(window.Gold = window.Gold || {});
