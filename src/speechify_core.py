# speechify_core.py
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

from openai import OpenAI
from src.speechify_prompts import (
    SINGLE_STEP_CONVERSION_PROMPT,
    SINGLE_STEP_CONVERSION_PROMPT_CONCISE,
    build_single_step_prompt,
)
from pydantic import BaseModel


class DialogueTurn(BaseModel):
    role: str
    content: str


class DialogueResponse(BaseModel):
    turns: List[DialogueTurn]


CONFIG = {}


def speechify_full_dialogue(
    llm_model_name: str,
    client: OpenAI,
    source_turns: List[Tuple[str, str]],
    context: str,
    temperature: float = 0.7,
    concise: bool = False,
) -> List[Dict[str, Any]]:
    if concise:
        # Halve the source before conversion; the concise prompt shortens what
        # remains, so feeding the full dialogue would fight against it.
        source_turns = source_turns[: len(source_turns) // 2]

    user_prompt = build_single_step_prompt(source_turns, context)

    system_prompt = (
        SINGLE_STEP_CONVERSION_PROMPT_CONCISE if concise else SINGLE_STEP_CONVERSION_PROMPT
    )

    completion = client.beta.chat.completions.parse(
        model=llm_model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        response_format=DialogueResponse,
    )

    # The content is already parsed into the Pydantic model
    parsed_response = completion.choices[0].message.parsed

    if parsed_response:
        # Convert the Pydantic objects back to a list of dicts for your return type
        return [turn.model_dump() for turn in parsed_response.turns]

    return []
