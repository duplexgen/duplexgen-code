import json
import re
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union, Dict
from tqdm import tqdm

Turn = Tuple[str, str]
Context = Dict[str, str]


def get_sampled_indices(total_count: int, max_samples: Optional[int]) -> List[int]:
    """Helper to calculate indices for proportional sampling."""
    if max_samples is None or max_samples >= total_count or max_samples <= 0:
        return list(range(total_count))

    # Calculate step size to spread samples across the whole dataset
    step = total_count / max_samples
    return [int(i * step) for i in range(max_samples)]


# SocraTeach dataset parsing utilities
def flatten_socrateach_dialog(dialog_list) -> List[Turn]:
    turns: List[Turn] = []
    for obj in dialog_list:
        if "END" in obj:
            if "system" in obj and obj["system"].strip():
                turns.append(("assistant", obj["system"].strip()))
            continue
        if "system" in obj and obj["system"].strip():
            turns.append(("assistant", obj["system"].strip()))
        if "user" in obj and obj["user"].strip():
            turns.append(("user", obj["user"].strip()))
    return turns


def build_socrateach_context(top_item: dict) -> Context:
    q = (top_item.get("question") or "").strip()
    context = f"User is solving a word problem. Assistant acts as a tutor, using the problem statement to guide step by step.\nQuestion:\n{q}"
    return context


def infer_split_from_key(key: str) -> str:
    k = key.lower()
    if "_train_" in k:
        return "train"
    if "_test_" in k:
        return "test"
    return "unknown"


def iter_socrateach(
    input_path: Union[str, Path],
    max_samples: Optional[int] = None,
    max_variants: Optional[int] = None,
) -> Iterator[Tuple[str, List[Turn], Context, str]]:
    """
    Yields (example_id, turns, context, split_label).
    example_id is "{top_k}__{vkey}"
    """
    input_path = Path(input_path)
    socra = json.loads(input_path.read_text(encoding="utf-8"))

    all_keys = list(socra.keys())
    indices = get_sampled_indices(len(all_keys), max_samples)
    top_keys = [all_keys[i] for i in indices]

    for top_k in top_keys:
        item = socra[top_k]
        split_label = infer_split_from_key(top_k)

        context = build_socrateach_context(item)
        dialogues = item.get("dialogues", {})
        variant_keys = sorted(dialogues.keys())
        if isinstance(max_variants, int):
            variant_keys = variant_keys[:max_variants]

        for vkey in variant_keys:
            ex_id = f"{top_k}__{vkey}"
            dialog_list = dialogues[vkey]
            turns = flatten_socrateach_dialog(dialog_list)
            yield ex_id, turns, context, split_label


def iter_multiwoz(
    input_path: Union[str, Path],
    max_samples: Optional[int] = None,
) -> Iterator[Tuple[str, List[Turn], Context]]:
    """
    Yields (example_id, turns, context).
    example_id is dialogue_id from the JSON.
    """
    input_path = Path(input_path)
    all_data = []
    for input_file in input_path.glob("*.json"):
        with open(input_file, "r", encoding="utf-8") as f:
            all_data.extend(json.load(f))

    indices = get_sampled_indices(len(all_data), max_samples)
    sampled_data = [all_data[i] for i in indices]

    context = "User is interacting with assistant to get information and make reservations for various service."

    for entry in sampled_data:
        ex_id = entry.get("dialogue_id", "unknown").strip(".json")
        turns: List[Turn] = []
        for turn in entry.get("turns", []):
            speaker = turn.get("speaker", "").upper()
            role = "user" if speaker == "USER" else "assistant"
            utterance = turn.get("utterance", "").strip()
            if utterance:
                turns.append((role, utterance))
        yield ex_id, turns, context


def split_interviewer_turns(text: str) -> List[Turn]:
    """
    Parses the raw text transcript into a list of (role, content) tuples.
    Handles variations in speaker labels (Assistant, Claude, AI, User).
    """
    # Regex to find "Speaker: Content" while handling multi-line content
    # Lookahead ensures we don't consume the next speaker label
    pattern = r"(Assistant|Claude|AI|User):\s*(.*?)(?=\n(?:Assistant|Claude|AI|User):|$)"
    matches = re.findall(pattern, text, flags=re.S | re.I)

    turns: List[Turn] = []
    for speaker, content in matches:
        speaker = speaker.strip().upper()
        content = content.strip()

        if not content:
            continue

        # Map labels to standard roles
        role = "assistant" if speaker in ["ASSISTANT", "CLAUDE", "AI"] else "user"
        turns.append((role, content))

    return turns


def iter_interviewer(
    dataset: List[Dict[str, str]],
    max_samples: Optional[int] = None,
    split: str = "workforce",
) -> Iterator[Tuple[str, List[Turn], Context]]:
    """
    Yields (example_id, turns, context).
    Since this dataset is typically loaded via HuggingFace 'datasets' library,
    this iterator accepts a list of dictionaries (the dataset slice).
    """
    samples = dataset[split]
    total_count = len(samples)

    indices = get_sampled_indices(total_count, max_samples)

    for i in indices:
        item = samples[i]
        ex_id = item.get("transcript_id", "unknown_id")
        raw_text = item.get("text", "")

        if not raw_text:
            yield ex_id, [], ""
            continue

        turns = split_interviewer_turns(raw_text)
        context = (
            f"An AI user researcher at Anthropic is interviewing a human participant "
            f"regarding their experiences with AI in the context of {split}."
        )

        yield ex_id, turns, context


def iter_persuader(
    input_path: Union[str, Path],
    max_samples: Optional[int] = None,
    max_variants: Optional[int] = None,
) -> Iterator[Tuple[str, List[Turn], str]]:
    """
    Yields (example_id, turns, context).
    example_id is formatted as {id}__{session_id}
    """
    input_path = Path(input_path)
    data = json.loads(input_path.read_text(encoding="utf-8"))

    indices = get_sampled_indices(len(data), max_samples)
    sampled_data = [data[i] for i in indices]

    for item in sampled_data:
        main_id = item.get("id", "unknown")

        scenario = item.get("scenario", {})
        background = scenario.get("background", "").strip()
        goal = scenario.get("goal", "").strip()
        context = f"Context: {background}\nGoal: {goal}"

        dialog_sessions = item.get("dialog", {})

        session_items = dialog_sessions.items()
        if isinstance(max_variants, int):
            session_items = list(session_items)[:max_variants]

        for session_id, session_turns in session_items:
            ex_id = f"{main_id}__{session_id}"
            turns: List[Turn] = []

            for t in session_turns:
                role = "assistant" if t.get("role") == "persuader" else "user"
                content = t.get("response", "").strip()
                if content:
                    turns.append((role, content))

            yield ex_id, turns, context


def iter_negotiator(
    input_path: Union[str, Path],
    max_samples: Optional[int] = None,
) -> Iterator[Tuple[str, List[Turn], Context]]:
    """
    Yields (example_id, turns, context).
    The context includes details about the item and the specific goals (targets)
    of the buyer and seller. agent 0 is treated as user, agent 1 as assistant.
    """
    input_path = Path(input_path)
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    indices = get_sampled_indices(len(data), max_samples)
    sampled_data = [data[i] for i in indices]

    for entry in sampled_data:
        ex_id = entry.get("uuid", "unknown_id")
        scenario = entry.get("scenario", {})
        kbs = scenario.get("kbs", [])

        # Build a descriptive context from the Knowledge Base (KB)
        # We'll identify the buyer and seller roles to explain the stakes
        context_parts = [f"Category: {scenario.get('category', 'N/A')}"]

        for kb in kbs:
            personal = kb.get("personal", {})
            role = {"buyer": "user (buyer)", "seller": "assistant (seller)"}.get(
                personal.get("Role", "unknown"),
                "unknown",
            )
            target = personal.get("Target", "N/A")
            item = kb.get("item", {})
            title = item.get("Title", "this item")

            context_parts.append(f"Role: {role} | Target Price: {target} | Item: {title}")

        context_str = "\n".join(context_parts)

        turns: List[Turn] = []
        events = entry.get("events", [])

        for event in events:
            action = event.get("action")
            agent = event.get("agent")

            # Map Agent 0 (Buyer) to User and Agent 1 to Assistant (Negotiator)
            role = "user" if agent == 0 else "assistant"

            if action == "message":
                content = event.get("data", "").strip()
            else:
                continue

            if content:
                turns.append((role, content))

        yield ex_id, turns, {"description": context_str}


def is_competitive_scenario(dialogue: str, client, model: str) -> bool:
    """Uses the LLM to classify if a scenario involves competition or conflict."""
    if not dialogue or client is None:
        return False

    formatted_dialogue = "\n".join([f"Speaker {i%2 + 1}: {msg}" for i, msg in enumerate(dialogue)])

    prompt = (
        "Analyze the following social scenario narrative. "
        "Determine if the interaction is COMPETITIVE (e.g., negotiation, debate, disagreement, "
        "conflicting goals, or if the participants are arguing/fighting or showing tension due to "
        "differences in opinions or mood) or NOT. "
        "Respond with only 'YES' if it is competitive, and 'NO' otherwise."
        "Also, the first speaker is the user and the second is the assistant. Answer YES only if this exchange would naturally occur in a user–AI interaction.\n\n"
        f"Dialogue: {formatted_dialogue}"
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0,
        )
        prediction = response.choices[0].message.content.strip().upper()
        return "YES" in prediction
    except Exception as e:
        print(f"Classification error: {e}")
        return False


def iter_soda(
    dataset,
    model: str,
    max_samples: Optional[int] = None,
    client: Optional[any] = None,
) -> Iterator[Tuple[str, List[Turn], Context]]:
    """
    Yields (example_id, turns, context) only for competitive scenarios.
    Replaces speaker names in the narrative with 'user' or 'assistant'.
    """
    samples = dataset
    count = 0

    for i, item in enumerate(tqdm(samples)):
        if max_samples is not None and count >= max_samples:
            break

        dialogue = item.get("dialogue", [])
        narrative = item.get("narrative", "").strip()
        speakers = item.get("speakers", [])

        # 1. Filter for competitive scenarios using the LLM
        # (Fixed: narrative must be defined before calling this)
        if not is_competitive_scenario(dialogue, client, model=model):
            continue

        # 2. Modify the narrative: Replace speaker names with roles
        # We use \b (word boundaries) to ensure we don't replace parts of other words
        processed_narrative = narrative
        if len(speakers) >= 2:
            user_name = speakers[0]
            assistant_name = speakers[1]

            # Replace Speaker 0 with 'user'
            processed_narrative = re.sub(
                rf"\b{re.escape(user_name)}\b", "User", processed_narrative, flags=re.IGNORECASE
            )
            # Replace Speaker 1 with 'assistant'
            processed_narrative = re.sub(
                rf"\b{re.escape(assistant_name)}\b",
                "Assistant",
                processed_narrative,
                flags=re.IGNORECASE,
            )

        ex_id = f"soda_{i}"

        # 3. Format turns
        turns: List[Turn] = []
        for idx, utterance in enumerate(dialogue):
            role = "user" if idx % 2 == 0 else "assistant"
            turns.append((role, utterance.strip()))

        context = {
            "description": f"Social interaction scenario: {processed_narrative}",
            "speakers": speakers,
            "original_narrative": narrative,  # Keeping original just in case
        }

        count += 1
        yield ex_id, turns, context
