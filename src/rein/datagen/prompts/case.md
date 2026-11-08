You write realistic conversations to train and test a decision model. The model reads the conversation and decides each rule for the LATEST user message only.

Scenario context:
$context

Rule under test (id "$rule_id", type $rule_type):
$rule_instructions
Options: $options
It applies when: $fires_when
It does not apply when: $does_not_fire_when

Write a conversation of about $turns messages in $language. For the latest user message, the correct answer to this rule must be "$target_option".
Case form: $case_form. $case_guidance
$second_rule_block
Writing rules:
- Every user and assistant message must be written in $language, even when the context and the rule are written in another language.
- Roles are "user", "assistant" (the character or product; use the name "$character_name" when it fits) and optionally "narrator".
- Roleplay conventions are welcome when they fit the domain: *asterisks* for actions, quotes for speech, (( )) for out-of-character asides.
- The last message comes from the user and is the decisive one.
- Never mention the rule, the options, labels or a decision model inside the conversation.
- Keep every message under 600 characters and make the people sound like real users, not like a test.
$draft_instruction

Reply with a JSON object only:
{"conversation": [{"role": "user" | "assistant" | "narrator", "name": str or null, "text": str}], "draft": $draft_schema}
