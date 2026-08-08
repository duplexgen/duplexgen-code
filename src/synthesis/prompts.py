from typing import List, Tuple
import re

# ----------------------------
# Utilities
# ----------------------------


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def dialog_to_block(turns: List[Tuple[str, ...]]) -> str:
    lines = []
    for i, turn in enumerate(turns, 1):
        # Unpack first two, ignore rest if any
        spk = turn[0] if turn else ""
        txt = turn[1] if len(turn) > 1 else ""
        lines.append(
            "{idx}) {spk}: {txt}".format(
                idx=i,
                spk=spk,
                txt=_normalize_ws(txt),
            )
        )
    return "\n".join(lines)


# ----------------------------
# Prompt templates (ALL here)
# ----------------------------

TOKEN_FT = "[TAKE_FLOOR]"
TOKEN_BC = "[BACKCHANNEL]"

ROLE_USER_TURN_PROMPT = f"""### System Prompt: Spoken Dialogue Simulator

**Role:**
You are an expert dialogue simulator specialized in converting static text conversations into realistic, multi-turn **spoken interactions**. Your goal is to mirror the nuances of verbal communication, including reactivity, interruptions, and context-dependent responses. Specifically, you should simulate the user's behavior as a human.

**Role Lock (critical):** You speak ONLY as the **user** for this turn. Never produce the assistant's lines, and never adopt the assistant's side, goals, or job — e.g. do not start giving help, answers, or offers if in the Original Full Text the user is the one seeking them. Mirror the SAME side the **user** takes in the Original Full Text.

**Input Data:**

1. **Original Full Text Conversation:** A complete, polished transcript of a text-based interaction. This serves as the "ground truth" for the facts, goals, and personalities of the speakers.
2. **Partial Transcribed Conversation:** The current history of the spoken simulation. This may contain unfinished sentences or fragmented thoughts.

**The `{TOKEN_FT}` Mechanism:**
The token `{TOKEN_FT}` represents a mid-sentence interruption or a rapid turn-take.

* **Trigger:** When the `{TOKEN_FT}` token appears at the end of a speaker's turn, it indicates that the next speaker has started talking **immediately**.
* **Constraint:** You must ignore any content from the "Original Full Text" that would have occurred after the `{TOKEN_FT}` token.
* **Response Logic:** Your response must be based **only** on the information provided up to that token. If the speaker was cut off mid-thought, you must react to that partial thought naturally (e.g., asking for clarification, finished their sentence, or responding to the fragment).

**Guidelines for Spoken Realism:**

* **Brief & Reactive:** Spoken turns are generally shorter than text.
* **Contextual Awareness:** If the user cuts you off with `{TOKEN_FT}`, acknowledge the shift in topic or the immediate needs of the user.
* **Prosodic Adaptation:** While you output text, write in a way that sounds "spoken"-include natural hesitations, occasional fillers, or more casual syntax compared to the source text.
    - Use contractions (I'm, don't, can't)
    - Include light disfluencies when natural (um, uh, you know)
    - Occasional short self-repairs are okay
    - Do NOT use em-dashes (—); use commas or natural pauses instead
* **Faithfulness to Scenario:** Maintain the personas and objectives established in the Original Full Text, but adapt the *delivery* to the real-time flow of the partial transcript.

**Output Constraint (STRICT — violating any of these invalidates the response):**

* **Single Turn Only:** Generate **only one single-turn utterance** as plain text. Do not simulate the rest of the conversation or provide multiple turns of dialogue. Your output should end as soon as your specific turn is complete.
* **Utterance Text ONLY.** Output raw spoken text. Do NOT prepend any of the following:
    - Turn indices like `1)`, `2.`, `- `, `* `, `# `, `10)`.
    - Role labels like `user:`, `assistant:`, `system:`, `human:`, `AI:`, `Speaker 1:`, `Speaker 2:`, `User —`, `USER:`.
    - Combined forms like `10) user:` or `user said:`.
    - Surrounding quotes (`"..."`, `'...'`, backticks), JSON, or Markdown formatting.
* First character of your response must be the first character of the utterance itself.

**Task:**
Analyze the provided **Transcribed Conversation**. Identify if the last turn ended with `{TOKEN_FT}`. If it did, generate the next logical response based on that specific cutoff point, ensuring your tone aligns with the **Original Full Text Conversation** while respecting the immediacy of the spoken format."""

ROLE_AI_TURN_PROMPT = f"""### System Prompt: Spoken Dialogue Simulator

**Role:**
You are an expert dialogue simulator specialized in converting static text conversations into realistic, multi-turn **spoken interactions**. Your goal is to mirror the nuances of verbal communication, including reactivity, interruptions, and context-dependent responses. Specifically, you should simulate the AI assistant's behavior.

**Role Lock (critical):** You speak ONLY as the **assistant** for this turn. Never produce the user's lines, and never adopt the user's side, goals, or requests — e.g. do not start asking for help, buying, or being tutored if in the Original Full Text the assistant is the one providing those. Mirror the SAME side the **assistant** takes in the Original Full Text.

**Input Data:**

1. **Original Full Text Conversation:** A complete, polished transcript of a text-based interaction. This serves as the "ground truth" for the facts, goals, and personalities of the speakers.
2. **Partial Transcribed Conversation:** The current history of the spoken simulation. This may contain unfinished sentences or fragmented thoughts.

**The `{TOKEN_FT}` Mechanism:**
The token `{TOKEN_FT}` represents a mid-sentence interruption or a rapid turn-take.

* **Trigger:** When the `{TOKEN_FT}` token appears at the end of a speaker's turn, it indicates that the next speaker has started talking **immediately**.
* **Constraint:** You must ignore any content from the "Original Full Text" that would have occurred after the `{TOKEN_FT}` token.
* **Response Logic:** Your response must be based **only** on the information provided up to that token. If the speaker was cut off mid-thought, you must react to that partial thought naturally (e.g., asking for clarification, finished their sentence, or responding to the fragment).

**Guidelines for Spoken Realism:**

* **Brief & Reactive:** Spoken turns are generally shorter than text.
* **Contextual Awareness:** If the user cuts you off with `{TOKEN_FT}`, acknowledge the shift in topic or the immediate needs of the user.
* **Prosodic Adaptation:** While you output text, write in a way that sounds "spoken" but clear and professional.
    - Use contractions (I'm, don't, can't)
    - Keep responses concise and direct
    - Avoid disfluencies (no um, uh, you know)
    - Do NOT use em-dashes (—); use commas or natural pauses instead
* **Faithfulness to Scenario:** Maintain the personas and objectives established in the Original Full Text, but adapt the *delivery* to the real-time flow of the partial transcript.

**Output Constraint (STRICT — violating any of these invalidates the response):**

* **Single Turn Only:** Generate **only one single-turn utterance** as plain text. Do not simulate the rest of the conversation or provide multiple turns of dialogue. Your output should end as soon as your specific turn is complete.
* **Utterance Text ONLY.** Output raw spoken text. Do NOT prepend any of the following:
    - Turn indices like `1)`, `2.`, `- `, `* `, `# `, `10)`.
    - Role labels like `user:`, `assistant:`, `system:`, `human:`, `AI:`, `Speaker 1:`, `Speaker 2:`, `User —`, `USER:`.
    - Combined forms like `10) user:` or `user said:`.
    - Surrounding quotes (`"..."`, `'...'`, backticks), JSON, or Markdown formatting.
* First character of your response must be the first character of the utterance itself.

**Task:**
Analyze the provided **Transcribed Conversation**. Identify if the last turn ended with `{TOKEN_FT}`. If it did, generate the next logical response based on that specific cutoff point, ensuring your tone aligns with the **Original Full Text Conversation** while respecting the immediacy of the spoken format."""

# ROLE_USER_TURN_PROMPT = """Your task: Convert one dialogue turn into natural spoken language.

# You will receive:
# - The full original text dialogue (for reference)
# - The dialogue history generated so far

# Your role: You are the user (human)

# Instructions:
# 1. Given the original full dialogue as reference and a partial spoken version, generate the next turn in the spoken version so that it maintains a flow consistent with the original dialogue.
# 2. Keep it natural and conversational:
#     - Use contractions (I'm, don't, can't)
#     - Include light disfluencies when natural (um, uh, you know)
#     - Occasional short self-repairs are okay
# 3. Keep responses brief

# Output format: Plain text only (no JSON, quotes, or markdown)
# """

# ROLE_AI_TURN_PROMPT = """Your task: Convert one dialogue turn into natural spoken language.

# You will receive:
# - The full original text dialogue (for reference)
# - The dialogue history generated so far

# Your role: You are the AI assistant

# Instructions:
# 1. Given the original full dialogue as reference and a partial spoken version, generate the next turn in the spoken version so that it maintains a flow consistent with the original dialogue.
# 2. Be clear and concise
# 3. Avoid disfluencies (no um, uh, etc.)

# Output format: Plain text only (no JSON, quotes, or markdown)
# """

REWRITE_WITH_HISTORY_TEMPLATE = """Original full text conversation (for reference):
{full_source}

Transcribed conversation history:
{history}

Task: Generate ONLY the next turn as {role}.
Output ONLY the raw utterance text. Do NOT prepend a turn index, the role name, "user:", "assistant:", quotes, or any other label. Start with the first spoken word.
"""

REWRITE_WITH_HISTORY_STREAMING_TEMPLATE = """Original full text conversation (for reference):
{full_source}

Transcribed conversation history:
{history}

Task: Continue the current turn as {role}.

Important:
- Do NOT start a new turn or switch speakers
- Generate ONLY the continuation that comes AFTER what has already been said
- Maintain the same tone and context
- Do NOT repeat or include any part of the existing content
- Output ONLY the raw continuation text — no turn indices, role labels, quotes, or markdown
"""

REWRITE_FIRST_TURN_TEMPLATE = """Original full text conversation (for reference):
{full_source}

Transcribed conversation history:
(empty)

Next source turn to rewrite:
{src_spk}: {src_txt}

Task: Rewrite ONLY the first source turn as {role}.
Output ONLY the raw utterance text. Do NOT prepend a turn index, the role name, "user:", "assistant:", quotes, or any other label. Start with the first spoken word.
"""

# ROLE_USER_TURN_PROMPT = """You rewrite ONE turn as a spoken utterance.

# Context you will receive
# - The full original text dialogue (for context only).
# - The full dialogue history so far.

# Rules
# - You are the user (human).
# - Generate the next natural response that follows the dialogue history.
# - Use the full original dialogue only to resolve references and keep consistency.
# - Keep it natural and brief: contractions, light disfluencies (um/uh/you know), and occasional short self-repairs are allowed.
# - Output ONLY the utterance text (no JSON, no quotes, no markdown).
# """

# ROLE_AI_TURN_PROMPT = """You rewrite ONE turn as a spoken utterance.

# Context you will receive
# - The full original text dialogue (for context only).
# - The full dialogue history so far.

# Rules
# - You are the AI assistant.
# - Generate the next natural response that follows the dialogue history.
# - Use the full original dialogue only to resolve references and keep consistency.
# - Be clear and concise. No disfluencies.
# - Output ONLY the utterance text (no JSON, no quotes, no markdown).
# """

# REWRITE_WITH_HISTORY_TEMPLATE = """Full original text dialogue:
# {full_source}

# Conversation so far (spoken, optional style context):
# {history}

# Rewrite ONLY the next turn as {role}.
# """

# REWRITE_WITH_HISTORY_STREAMING_TEMPLATE = """Full original text dialogue:
# {full_source}

# Conversation so far (spoken, optional style context):
# {history}

# Continue generating the same turn as {role}. Do NOT start a new turn or switch speakers. Generate ONLY the continuation that comes AFTER what has already been said, maintaining the same tone and context. Do NOT repeat or include any part of the existing content.
# """

# REWRITE_FIRST_TURN_TEMPLATE = """Full original text dialogue:
# {full_source}

# Conversation so far (spoken, optional style context):
# (empty)

# Next source turn to rewrite:
# {src_spk}: {src_txt}

# Rewrite ONLY the next source turn as {role}.
# """

DONE_JUDGE_PROMPT = """You judge (1) coverage and (2) whether further generation is no longer meaningful.

Goal
- Prefer continuing generation until max_turns.
- Stop early only when continuing is very unlikely to add value.

Definitions
- "Covers" means: every source turn's intent or content appears somewhere in the generated dialogue. Minor paraphrasing is fine.
- "No longer meaningful" means: recent turns repeatedly add little or no new information, intent, or interactional progress, and mainly consist of politeness, closure, or repetition.

Rules
1) Coverage
- If important source content is missing, return done=false and list missing items.

2) Early shutdown
- If the dialogue appears to have stalled and additional turns are unlikely to introduce new content or intent, set stop=true.
- This should be a high-confidence judgment based on recent turns as a whole, not a single utterance.
- Avoid stopping if there is any reasonable chance that meaningful content could still emerge.

Output
Return ONLY valid JSON:
{
  "done": true/false,
  "missing": ["...","..."],
  "stop": true/false,
  "stop_reason": "coverage" | "stalled_dialogue" | "other",
  "notes": "brief explanation"
}
"""

JUDGE_USER_TEMPLATE = """SOURCE:
{source}

GENERATED:
{generated}
"""


# --- Prompts from snippets ---

PROMPT_BOUNDARY_DETECTION = """# Role and Objective
Transform transcript text by inserting a '|' symbol after every word that signals the end of a logical clause, sentence, or complete thought.

# Instructions
- Analyze the spoken dialogue text and identify the conclusion of each logical clause.
- Insert a '|' symbol immediately after the final word (and punctuation, if present) marking the end of a clause, sentence, or complete thought.

## Detailed Rules
- If a clause ends with punctuation (e.g., period, comma, exclamation mark, or question mark), place the '|' directly after the punctuation (do not remove or replace the original punctuation).
- Do not remove or alter existing text or punctuation.
- Only insert the '|' marker as specified; do not add extra output or commentary.

## Error Handling
- If the input text is missing or malformed, return an empty string.

# Context
Input variable: {original_text} (string containing the spoken dialogue transcript).

# Output Format
Return **only** the marked text string, with '|' symbols marking the ends of logical clauses as described.

### Example
**Input:** Well, I was thinking maybe we can talk later if you have time.
**Output:** Well,| I was thinking| maybe we can talk later| if you have time.|
"""

PROMPT_VERBALIZED_SCORING = """You are asked to annotate the specific action for potential AI turn-taking when user is speaking, based on **what turn-taking behavior users would prefer at that moment**.

# Task

You will be given:
1) A brief scenario description.
2) A dialogue context (four previous turns).
3) The user's current streaming utterance.

For the next token generation step, estimate a probability distribution over the AI assistant's next action at that moment:
- **floor_taking**: The assistant interrupts and takes the conversational floor.
- **backchannel**: The assistant produces a brief continuer/acknowledgement WITHOUT taking the floor (e.g., "mm-hm", "I see", "right"), and the user is expected to keep speaking.
- **silence**: The assistant does nothing and keeps listening.

# Output requirements

- Return valid JSON only.
- Output a probability distribution over {floor_taking, backchannel, silence}.
- Each probability must be a number between 0 and 1.
- The three probabilities must sum to exactly 1.

# Input format

Scenario description:
<scenario description here>

Dialogue context:
<dialogue context here>

User turn:
<user turn here>

# Output JSON format
{
    "floor_taking": PROBABILITY_OF_FLOOR_TAKING,
    "backchannel": PROBABILITY_OF_BACKCHANNEL,
    "silence": PROBABILITY_OF_SILENCE
}"""

# ----------------------------
# Prompt builders (logic only)
# ----------------------------


def build_rewrite_user_prompt(
    role: str,
    full_source_turns: List[Tuple[str, str]],
    history_turns: List[Tuple[str, str]],
    source_turn: Tuple[str, str] | None,
    streaming: bool = False,
) -> str:
    full_source_block = dialog_to_block(full_source_turns)

    if history_turns:
        history_block = dialog_to_block(history_turns)

        if streaming:
            return REWRITE_WITH_HISTORY_STREAMING_TEMPLATE.format(
                full_source=full_source_block,
                history=history_block,
                role=role,
            )
        else:
            return REWRITE_WITH_HISTORY_TEMPLATE.format(
                full_source=full_source_block,
                history=history_block,
                role=role,
            )

    src_spk, src_txt = source_turn
    return REWRITE_FIRST_TURN_TEMPLATE.format(
        full_source=full_source_block,
        src_spk=src_spk,
        src_txt=_normalize_ws(src_txt),
        role=role,
    )


def build_judge_user_prompt(
    source_turns: List[Tuple[str, str]],
    spoken_turns: List[Tuple[str, str]],
) -> str:
    return JUDGE_USER_TEMPLATE.format(
        source=dialog_to_block(source_turns),
        generated=dialog_to_block(spoken_turns),
    )
