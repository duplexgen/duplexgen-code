SINGLE_STEP_CONVERSION_PROMPT = """### Role
Expert in refining static, polished text transcripts into lifelike, multi-turn spoken interactions.

### Mission
Convert formal text dialogues into spoken versions that emulate real-time, natural conversations between a user and an AI assistant.

Begin with a concise checklist (3–7 bullets) of what you will do; keep items conceptual, not implementation-level.

### Instructions
- Rewrite the provided text dialogue with spoken, conversational flow.
- Inject the following stylistic elements based on speaker role:
  - **User:** Use contractions, natural disfluencies (e.g., "um", "uh", "you know"), some self-repair or repetition (e.g., "I, I"), and informal phrasing.
  - **AI Assistant:** Maintain a professional yet approachable tone; use contractions but avoid disfluencies such as "um", "uh".
- Ensure the conversation unfolds reactively, not as monologues.
- Safeguard all original facts, goals, and speaker personas.
- Omit or replace direct name references. If a speaker addresses someone by name, delete it or substitute with a generic or natural alternative (e.g., no greeting, "hey", or "you").
- Do not use em dashes (—) anywhere in the output. Use commas or periods instead.

### Output Format
Produce ONLY a valid JSON array of objects. Each object must have `role` and `content` fields.

Example Output:
[
  {"role": "user", "content": "Hi there, um, I was wondering if you could help me?"},
  {"role": "assistant", "content": "I'd be happy to help! What's on your mind?"}
]
"""

SINGLE_STEP_CONVERSION_PROMPT_CONCISE = """Developer: ### Role
Expert in refining static, polished text transcripts into lifelike, concise, multi-turn spoken interactions.

### Mission
Convert lengthy formal text dialogues into brief spoken versions that emulate real-time, natural conversations between a user and an AI assistant, while significantly reducing both overall length and turn count.

### Instructions
- Rewrite the provided text dialogue with spoken, conversational flow.
- Inject the following stylistic elements based on speaker role:
  - **User:** Use contractions, natural disfluencies (e.g., "um", "uh", "you know"), some self-repair or repetition (e.g., "I, I"), and informal phrasing.
  - **AI Assistant:** Maintain a professional yet approachable tone; use contractions but avoid disfluencies such as "um", "uh".
- Ensure the conversation unfolds reactively, not as monologues.
- Safeguard all original facts, goals, and speaker personas.
- Omit or replace direct name references. If a speaker addresses someone by name, delete it or substitute with a generic or natural alternative (e.g., no greeting, "hey", or "you").
- Do not use em dashes (—) anywhere in the output. Use commas or periods instead.

### Output Format
Produce ONLY a valid JSON array of objects. Each object must have `role` and `content` fields.

Example Output:
[
  {"role": "user", "content": "Hi there, um, I was wondering if you could help me?"},
  {"role": "assistant", "content": "I'd be happy to help! What's on your mind?"}
]
"""


def build_single_step_prompt(source_turns, context_description):
    source_block = ""
    for i, (role, text) in enumerate(source_turns):
        source_block += f"{i}) {role}: {text}\n"

    prompt = f"Context: {context_description}\n\nOriginal Text Dialogue:\n{source_block}\n\n"
    prompt += "Please convert this entire dialogue into a spoken-style transcript following the guidelines."
    return prompt
