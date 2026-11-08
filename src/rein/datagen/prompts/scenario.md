You design rule sheets for a typed decision model. The model watches a conversation and decides which behavior rules apply to the latest turn, so an application can act on the generator (force a tool call, lock an attribute, reinject a rule, escalate).

Create one scenario for this domain: $domain.
Write the title, the context and every rule in $rule_language. Word every rule as $rule_style, like a real creator of this kind of product would.

The context is what the application knows: a character card or product policy, plus any game-state flags or business limits the rules depend on. Keep it under 1200 characters and make it specific (names, numbers, flags).

Write exactly one rule for each of these categories, in this order:
$categories

Each rule is a question about the latest turn of the conversation, never about the whole history. Types:
- "noul": a yes/no question; "criteria" may describe the "true" and "false" answers, or be null.
- "choice": named options; "criteria" maps each snake_case option key to a short description or null, and it must include a "none" option.
- "score": exactly 5 ordered levels; "criteria" is the list of 5 level descriptions, lowest first, each a concrete anchor (for tone: the first is fully on tone, the last is a completely different character), so different annotators give the same level to the same situation.

"fires_when" and "does_not_fire_when" describe, in one sentence each, what makes the rule apply and a close case where it does not.
List in "tools" every tool name (snake_case) that a tool or action rule refers to, or an empty list.

Reply with a JSON object only:
{"title": str, "context": str, "character_name": str or null, "tools": [str], "rules": [{"id": snake_case str, "category": str, "type": "noul" | "choice" | "score", "instructions": str, "criteria": object | list | null, "fires_when": str, "does_not_fire_when": str}]}
