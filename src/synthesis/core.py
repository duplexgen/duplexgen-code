import json
import re
import random
from typing import Any, Dict, List, Optional, Tuple
from openai import OpenAI
from tqdm import tqdm

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.llm_client import no_thinking_extra_body
from src.synthesis.prompts import (
    build_rewrite_user_prompt,
    build_judge_user_prompt,
    ROLE_USER_TURN_PROMPT,
    ROLE_AI_TURN_PROMPT,
    DONE_JUDGE_PROMPT,
    TOKEN_FT,
    TOKEN_BC,
    PROMPT_BOUNDARY_DETECTION,
    PROMPT_VERBALIZED_SCORING,
)


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from LLM responses (e.g. Qwen thinking mode)."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ── Speaker-prefix / formatting contamination sanitizer ────────────────────────
# Matches an unwanted leading prefix the model may emit, e.g.:
#   "user:", "Assistant:", "AI:", "human:", "system:",
#   "Speaker 1:", "speaker 2:",
#   "10)", "2.", "- ", "* ", "1) user:", "2. assistant —",
#   wrapping quotes.
_ROLE_WORD = r"(?:user|assistant|ai|human|system|speaker\s*\d+|bot|agent)"
_INDEX_PREFIX = r"(?:\d+\s*[\.\)\]:]|[-*#•]\s)"
_PREFIX_RX = re.compile(
    rf"^\s*(?:{_INDEX_PREFIX}\s*)?(?:{_ROLE_WORD}\s*[:\-—–]\s*)?",
    re.IGNORECASE,
)
_ROLE_ONLY_RX = re.compile(rf"^\s*{_ROLE_WORD}\s*[:\-—–]\s*", re.IGNORECASE)
_INDEX_ONLY_RX = re.compile(rf"^\s*{_INDEX_PREFIX}\s*", re.IGNORECASE)
# Hard-check: after sanitation, no line may still start with one of these patterns.
_VALIDATE_BAD_START_RX = re.compile(
    rf"^\s*(?:{_INDEX_PREFIX}\s*)?{_ROLE_WORD}\s*[:\-—–]",
    re.IGNORECASE,
)


def sanitize_utterance(raw: str) -> str:
    """Strip speaker-prefix / enumeration / markdown contamination from one utterance.

    Safe to run repeatedly. Returns normalized text (may be empty string)."""
    if not raw:
        return ""
    s = raw.strip()

    # Strip <think>…</think> (defense in depth; callers also strip).
    s = _strip_think_tags(s)

    # Strip code fences / markdown headings / leading bullets lines-first.
    s = re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", s).strip()
    s = re.sub(r"\s*```\s*$", "", s).strip()

    # Peel leading prefixes in a loop to catch nested cases like
    # "10) user: \"hi there\"" → "hi there"
    for _ in range(4):
        before = s
        # index + role (e.g. "10) user:")
        m = _PREFIX_RX.match(s)
        if m and m.end() > 0:
            s = s[m.end() :].lstrip()
        # bare role ("user:")
        m = _ROLE_ONLY_RX.match(s)
        if m:
            s = s[m.end() :].lstrip()
        # bare index ("10)" still present)
        m = _INDEX_ONLY_RX.match(s)
        if m:
            s = s[m.end() :].lstrip()
        # surrounding quotes
        if len(s) >= 2 and s[0] in "\"'“‘`" and s[-1] in "\"'”’`":
            s = s[1:-1].strip()
        if s == before:
            break

    # Strip em-dashes (legacy).
    s = s.replace("\u2014", ", ").replace("\u2013", ", ")
    # Normalize whitespace.
    s = re.sub(r"\s+", " ", s).strip()
    return s


def has_prefix_contamination(s: str) -> bool:
    """Hard validator: True iff the string still begins with a role/index prefix."""
    if not s:
        return False
    return bool(_VALIDATE_BAD_START_RX.match(s.strip()))


# --- Configuration & Constants ---
MARKS = ['"', "'", "`", "!", "?", ".", ","]
HESITATIONS = {"um", "uh", "umm", "uhh", "hm", "hmm", "huh"}
# Heuristics for boundary detection
STOP_PUNCTUATION = r"[.,?!;]+$"

CONFIG = {
    "length_guard_start": 0,
    "interruption_guard_start": 3,  # Don't allow floor-taking decisions in the first N words
    "length_guard_gap": 4,
    "sampling_seed": 42,
}

# --- Core Logic Functions ---


def clean_word(token: str) -> str:
    return re.sub(r"[^\w\s]", "", token).lower()


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def safe_json_parse(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except Exception:
        # Try to find JSON block
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        return None


def _sample_action(probs: Dict[str, float], rng: random.Random) -> str:
    r = rng.random()
    cum = 0.0
    for k in ("floor_taking", "backchannel", "silence"):
        cum += probs.get(k, 0.0)
        if r <= cum:
            return k
    return "silence"


def process_history(history: List[Tuple[str, ...]]) -> List[Tuple[str, ...]]:
    new_history = []
    for turn in history:
        copy_turn = list(turn)
        content = copy_turn[1]
        content = content.replace(TOKEN_BC, "")
        if TOKEN_FT in content:
            content = " ".join([content.split(TOKEN_FT)[0], TOKEN_FT])
        copy_turn[1] = normalize_ws(content)
        if len(copy_turn) > 2:
            copy_turn = copy_turn[:2]
        new_history.append(tuple(copy_turn))
    return new_history


# --- New Logic: Boundary Detection (Strategy 1) ---


def detect_turn_boundaries(client: OpenAI, model_name: str, text: str) -> List[int]:
    """
    Uses LLM to insert '|' markers, then maps them + heuristics to word indices.
    """
    text = normalize_ws(text)
    words = text.split()
    if not words:
        return []

    # 1. LLM Boundary Detection
    prompt = PROMPT_BOUNDARY_DETECTION.format(original_text=text)
    predicted_indices = set()

    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a linguistic expert. Insert '|' at clause boundaries.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            extra_body=no_thinking_extra_body(model_name),
        )
        marked_text = _strip_think_tags(resp.choices[0].message.content or "").strip()

        # Map markers back to indices
        marked_tokens = marked_text.split()
        current_original_idx = 0

        # Simple alignment: iterate marked tokens, if pipe exists, mark current index
        # Note: This alignment assumes 1:1 token mapping mostly.
        for token in marked_tokens:
            if "|" in token:
                predicted_indices.add(current_original_idx)
            current_original_idx += 1

    except Exception as e:
        print(f"Warning: Boundary detection LLM failed: {e}")

    # 2. Heuristic Check (Hesitations & Punctuation)
    final_indices = []
    for i, word in enumerate(words):
        is_boundary = False

        # Rule A: LLM said so
        if i in predicted_indices:
            is_boundary = True

        # Rule B: Explicit Punctuation
        if re.search(STOP_PUNCTUATION, word):
            is_boundary = True

        # Rule C: Hesitations
        clean = clean_word(word)
        if clean in HESITATIONS:
            is_boundary = True

        if is_boundary:
            final_indices.append(i)

    return sorted(list(set(final_indices)))[:-1]  # Exclude last boundary to avoid end-of-turn


# --- New Logic: Probability Estimation (Strategy 2a) --- LLM-based ---


def predict_turn_taking_probabilities(
    client: OpenAI,
    model_name: str,
    scenario_desc: str,
    history_turns: List[Tuple[str, str]],
    full_turn_text: str,
    boundary_indices: List[int],
) -> List[Dict[str, float]]:
    """
    For each boundary index, asks LLM for [floor_taking, backchannel, silence] probs.
    """
    words = normalize_ws(full_turn_text).split()
    results = []

    # Format history for prompt
    formatted_history = ""
    start_idx = max(0, len(history_turns) - 4)
    for r, c, h in history_turns[start_idx:]:
        formatted_history += f"**{r}**: {c}\n"

    # Cache prompt parts if possible, but here we loop
    for b_idx in boundary_indices:  # Exclude last boundary to avoid end-of-turn
        # Partial utterance up to boundary
        partial_turn = " ".join(words[: b_idx + 1])

        prompt = (
            f"Scenario description:\n{scenario_desc}\n\n"
            f"Dialogue context:\n{formatted_history}\n\n"
            f"User turn:\n{partial_turn}"
        )

        # Retry logic or safe parsing
        dist = {"floor_taking": 0.0, "backchannel": 0.0, "silence": 1.0}  # Default

        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": PROMPT_VERBALIZED_SCORING},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                extra_body=no_thinking_extra_body(model_name),
            )
            raw_json = _strip_think_tags(resp.choices[0].message.content or "")
            parsed = safe_json_parse(raw_json)

            if parsed and all(k in parsed for k in ["floor_taking", "backchannel", "silence"]):
                # Normalize just in case
                total = sum([parsed[k] for k in ["floor_taking", "backchannel", "silence"]])
                if total > 0:
                    dist = {
                        k: parsed[k] / total for k in ["floor_taking", "backchannel", "silence"]
                    }
        except Exception as e:
            print(f"Warning: Probability estimation failed at index {b_idx}: {e}")

        results.append(dist)

    return results


# --- New Logic: Probability Estimation (Strategy 2b) --- HF Classification ---


def predict_turn_taking_probabilities_hf(
    model,
    tokenizer,
    history_turns: List[Tuple],
    full_turn_text: str,
    boundary_indices: List[int],
    max_seq_length: int = 1024,
    use_last_n_history: int = 4,
    batch_size: int = 8,
) -> List[Dict[str, float]]:
    """
    Uses a HF token-classification model to predict turn-taking probabilities at
    each boundary position in full_turn_text.

    The model must return {"logits": Tensor[B, T, num_labels]} with labels ordered
    as in inference_turntaking_hf.py: ["backchannel"(0), "interruption"(1), "none"(2)].

    Returns a list of dicts with keys: floor_taking, backchannel, silence.
    """
    import torch

    words = normalize_ws(full_turn_text).split()
    default_dist = {"floor_taking": 0.0, "backchannel": 0.0, "silence": 1.0}

    if not words or not boundary_indices:
        return [default_dist.copy() for _ in boundary_indices]

    # Build context from the most recent spoken turns, stripping TT tokens
    start_idx = max(0, len(history_turns) - use_last_n_history)
    context = []
    for turn in history_turns[start_idx:]:
        role = turn[0]
        content = turn[1] if len(turn) > 1 else ""
        mapped_role = "user" if role == "user" else "assistant"
        if isinstance(content, str):
            clean = normalize_ws(content.replace(TOKEN_FT, "").replace(TOKEN_BC, ""))
            if clean:
                context.append({"role": mapped_role, "content": clean})

    tokenizer_kwargs = {}
    if hasattr(tokenizer, "name_or_path") and "Qwen/Qwen3" in tokenizer.name_or_path:
        tokenizer_kwargs["enable_thinking"] = False

    # Single tokenization of the full utterance with offset mapping
    utterance = normalize_ws(full_turn_text)
    messages = context + [{"role": "user", "content": utterance}]
    try:
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            **tokenizer_kwargs,
        )
        out = tokenizer(
            text=full_text,
            truncation=True,
            max_length=max_seq_length,
            return_offsets_mapping=True,
        )
    except Exception as e:
        print(f"Warning: HF tokenization failed: {e}")
        return [default_dist.copy() for _ in boundary_indices]

    input_ids = out["input_ids"]
    offsets = out["offset_mapping"]
    if not input_ids:
        return [default_dist.copy() for _ in boundary_indices]

    # Map each boundary word_idx → token position
    word_end_offsets = []
    curr_pos = 0
    for w in words:
        start = utterance.find(w, curr_pos)
        end = start + len(w)
        word_end_offsets.append(end)
        curr_pos = end

    utterance_start_char = full_text.find(utterance)
    if utterance_start_char == -1:
        return [default_dist.copy() for _ in boundary_indices]

    valid_boundary_tok_indices = []  # (list_idx, token_idx)
    for list_idx, b_idx in enumerate(boundary_indices):
        if b_idx < 0 or b_idx >= len(word_end_offsets):
            continue
        target_char_pos = utterance_start_char + word_end_offsets[b_idx]
        for t_idx, (s, e) in enumerate(offsets):
            if s < target_char_pos <= e:
                valid_boundary_tok_indices.append((list_idx, t_idx))
                break

    if not valid_boundary_tok_indices:
        return [default_dist.copy() for _ in boundary_indices]

    # Single forward pass
    results_map: Dict[int, Dict[str, float]] = {}

    model.eval()
    with torch.no_grad():
        input_ids_t = torch.tensor([input_ids], dtype=torch.long)
        attention_mask_t = torch.tensor([out["attention_mask"]], dtype=torch.long)

        if torch.cuda.is_available():
            input_ids_t = input_ids_t.cuda()
            attention_mask_t = attention_mask_t.cuda()

        logits = model(input_ids=input_ids_t, attention_mask=attention_mask_t)["logits"]
        logits = logits[0]  # (T, C), single example

        tok_indices = torch.tensor(
            [t_idx for _, t_idx in valid_boundary_tok_indices],
            dtype=torch.long,
            device=logits.device,
        )
        selected = logits[tok_indices]  # (num_boundaries, C)
        probs = torch.softmax(selected.float(), dim=-1).detach().cpu().numpy()

        for (list_idx, _), prob_arr in zip(valid_boundary_tok_indices, probs):
            # Label order: 0=backchannel, 1=interruption(->floor_taking), 2=none(->silence)
            results_map[list_idx] = {
                "floor_taking": float(prob_arr[1]),
                "backchannel": float(prob_arr[0]),
                "silence": float(prob_arr[2]),
            }

    return [results_map.get(i, default_dist.copy()) for i in range(len(boundary_indices))]


# --- Token Insertion (Unchanged logic, wrapped) ---


def insert_action_tokens_from_llm_annotations(
    text: str,
    boundary_word_indices: List[int],
    boundary_dists: List[Dict[str, float]],
    length_guard_start: int = 0,
    length_guard_gap: int = 0,
    interruption_guard_start: int = 5,
    rng: Optional[random.Random] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    rng = rng or random.Random()
    words = normalize_ws(text).split()

    if not words or not boundary_word_indices:
        return normalize_ws(text), [{"full_content": normalize_ws(text)}]

    # Defensive check
    if len(boundary_word_indices) != len(boundary_dists):
        min_len = min(len(boundary_word_indices), len(boundary_dists))
        boundary_word_indices = boundary_word_indices[:min_len]
        boundary_dists = boundary_dists[:min_len]

    paired = sorted(zip(boundary_word_indices, boundary_dists), key=lambda x: x[0])

    out_words: List[str] = []
    action_history: List[Dict[str, Any]] = [{"full_content": normalize_ws(text)}]

    last_bc_pos: Optional[int] = None
    floor_taking_triggered = False
    boundary_ptr = 0

    for i, w in enumerate(words):
        out_words.append(w)

        while boundary_ptr < len(paired) and paired[boundary_ptr][0] == i:
            probs = paired[boundary_ptr][1]
            decision = "silence"
            inserted = None

            if i < length_guard_start or floor_taking_triggered:
                decision = "silence"
            else:
                sampled = _sample_action(probs, rng)
                if sampled == "floor_taking":
                    if i < interruption_guard_start:
                        decision = "silence"
                    else:
                        decision = "floor_taking"
                        inserted = TOKEN_FT
                        floor_taking_triggered = True
                elif sampled == "backchannel":
                    if last_bc_pos is None or (i - last_bc_pos) >= length_guard_gap:
                        decision = "backchannel"
                        inserted = TOKEN_BC
                        last_bc_pos = i
                    else:
                        decision = "silence"
                else:
                    decision = "silence"

            if inserted:
                out_words.append(inserted)

            action_history.append(
                {"word_index": i, "probs": probs, "decision": decision, "inserted_token": inserted}
            )
            boundary_ptr += 1

    out = normalize_ws(" ".join(out_words))
    # Post-processing cleanup
    if TOKEN_FT in out:
        out = normalize_ws(" ".join([out.replace(TOKEN_BC, "").split(TOKEN_FT)[0], TOKEN_FT]))
    else:
        out = normalize_ws(out.replace(TOKEN_BC, ""))

    return out, action_history


def generate_raw_content_turn(
    llm_model_name: str,
    client: OpenAI,
    role: str,
    history_turns: List[Tuple[str, str]],
    full_source_turns: List[Tuple[str, str]],
    temperature: float = 0.7,
    source_turn: Optional[Tuple[str, str]] = None,
    max_format_retries: int = 3,
) -> str:
    """Generates the text content ONLY, without tokens.

    Hardening against speaker-prefix / enumeration contamination:
      1. Strict system prompt forbids role prefixes, indices, quotes, markdown.
      2. Raw output is passed through ``sanitize_utterance`` which peels leading
         role labels, enumerations, and wrapping quotes.
      3. After sanitation we hard-validate with ``has_prefix_contamination``.
         On failure we regenerate up to ``max_format_retries`` times with an
         explicit corrective reminder and slightly raised temperature.
    """
    sys_prompt = ROLE_USER_TURN_PROMPT if role == "user" else ROLE_AI_TURN_PROMPT
    user_prompt = build_rewrite_user_prompt(
        role=role,
        full_source_turns=full_source_turns,
        history_turns=process_history(history_turns),
        source_turn=source_turn,
    )

    attempts: List[str] = []
    for attempt in range(max_format_retries + 1):
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if attempt > 0:
            # Corrective reminder echoing the offending previous attempt.
            messages.append(
                {
                    "role": "assistant",
                    "content": attempts[-1],
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous response started with a disallowed prefix "
                        "(turn index, role name like 'user:' / 'assistant:', or "
                        "wrapping quotes). Regenerate the SAME utterance as raw "
                        "spoken text ONLY. No index, no role label, no quotes, no "
                        "markdown. Begin with the first spoken word."
                    ),
                }
            )

        resp = client.chat.completions.create(
            model=llm_model_name,
            messages=messages,
            temperature=min(1.0, temperature + 0.1 * attempt),
            extra_body=no_thinking_extra_body(llm_model_name),
        )
        raw_full = _strip_think_tags(resp.choices[0].message.content or "")
        attempts.append(raw_full)

        raw = raw_full.replace(TOKEN_FT, "").replace(TOKEN_BC, "")
        clean = sanitize_utterance(raw)

        if clean and not has_prefix_contamination(clean):
            return normalize_ws(clean)

        print(
            f"Warning: prefix contamination detected (attempt {attempt + 1}/"
            f"{max_format_retries + 1}); sample={raw[:80]!r}"
        )

    # Final fallback: return the last sanitized version (possibly still flawed).
    # Caller logs this; downstream validation in run_single() will reject the
    # whole dialogue if needed.
    last_clean = sanitize_utterance(attempts[-1].replace(TOKEN_FT, "").replace(TOKEN_BC, ""))
    return normalize_ws(last_clean)


def judge_done(
    client: OpenAI,
    llm_model_name: str,
    source_turns: List[Tuple[str, str]],
    spoken_turns: List[Tuple[str, str]],
) -> bool:
    prompt = build_judge_user_prompt(source_turns, spoken_turns)
    resp = client.chat.completions.create(
        model=llm_model_name,
        messages=[
            {"role": "system", "content": DONE_JUDGE_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        extra_body=no_thinking_extra_body(llm_model_name),
    )
    obj = safe_json_parse(_strip_think_tags(resp.choices[0].message.content or ""))
    return bool(obj and obj.get("done") is True)


# --- Main Pipeline ---


def speechify_turn_by_turn(
    llm_model_name: str,
    client: OpenAI,
    client_tt: Optional[OpenAI] = None,  # Used for LLM-based probability estimation
    client_boundary: Optional[OpenAI] = None,  # Used for boundary detection
    source_turns: List[Tuple[str, str]] = [],
    scenario_description: str = "A generic conversation.",
    max_turns: Optional[int] = None,
    stop_check_every: Optional[int] = 1,
    min_turns_before_stop: int = 2,
    temperature_user: float = 0.7,
    temperature_ai: float = 0.2,
    dataset: Optional[str] = None,
    tt_model_name: str = "Qwen/Qwen3-14B",
    boundary_model_name: str = "gpt-4.1-mini",
    # HF classification model (mutually exclusive with client_tt for TT scoring)
    hf_model=None,
    hf_tokenizer=None,
    hf_max_seq_length: int = 1024,
    hf_use_last_n_history: int = 4,
    hf_batch_size: int = 8,
) -> List[Dict[str, str]]:

    spoken_turns: List[Tuple[str, str]] = []
    steps = 0
    current_role = source_turns[0][0]  # "user" or "assistant"

    # Global RNG for token sampling
    GLOBAL_RNG = random.Random(CONFIG.get("sampling_seed", 42))

    if max_turns is not None:
        pbar = tqdm(total=max_turns, desc="Generating turns")
    else:
        pbar = tqdm(desc="Generating turns")

    while True:
        if max_turns is not None and steps >= max_turns:
            break

        source_turn = source_turns[0] if steps == 0 else None
        temp = temperature_user if current_role == "user" else temperature_ai

        # 1. Generate Raw Text Content
        raw_content = generate_raw_content_turn(
            llm_model_name=llm_model_name,
            client=client,
            role=current_role,
            history_turns=spoken_turns,
            full_source_turns=source_turns,
            temperature=temp,
            source_turn=source_turn,
        )

        final_content = raw_content
        history_meta = [{"full_content": raw_content}]

        # 2. If User, Apply Turn-Taking Logic
        if current_role == "user":
            # A. Detect Boundaries
            b_indices = detect_turn_boundaries(client_boundary, boundary_model_name, raw_content)

            if b_indices:
                # B. Predict Probabilities: HF model takes priority over LLM
                if hf_model is not None and hf_tokenizer is not None:
                    b_dists = predict_turn_taking_probabilities_hf(
                        model=hf_model,
                        tokenizer=hf_tokenizer,
                        history_turns=spoken_turns,
                        full_turn_text=raw_content,
                        boundary_indices=b_indices,
                        max_seq_length=hf_max_seq_length,
                        use_last_n_history=hf_use_last_n_history,
                        batch_size=hf_batch_size,
                    )
                else:
                    if client_tt is None:
                        raise ValueError(
                            "Either provide hf_model+hf_tokenizer or client_tt for TT scoring."
                        )
                    b_dists = predict_turn_taking_probabilities(
                        client=client_tt,
                        model_name=tt_model_name,
                        scenario_desc=scenario_description,
                        history_turns=spoken_turns,
                        full_turn_text=raw_content,
                        boundary_indices=b_indices,
                    )

                # C. Insert Tokens
                final_content, history_meta = insert_action_tokens_from_llm_annotations(
                    text=raw_content,
                    boundary_word_indices=b_indices,
                    boundary_dists=b_dists,
                    length_guard_start=CONFIG["length_guard_start"],
                    length_guard_gap=CONFIG["length_guard_gap"],
                    interruption_guard_start=CONFIG["interruption_guard_start"],
                    rng=GLOBAL_RNG,
                )

        spoken_turns.append((current_role, final_content, history_meta))
        current_role = "assistant" if current_role == "user" else "user"
        steps += 1

        pbar.update(1)
        pbar.set_postfix({"step": steps, "role": current_role})

        if (
            stop_check_every
            and stop_check_every > 0
            and steps >= min_turns_before_stop
            and (steps % stop_check_every == 0)
        ):
            if judge_done(client, llm_model_name, source_turns, spoken_turns):
                break

    return [{"role": s, "content": t, "history": h} for (s, t, h) in spoken_turns]
