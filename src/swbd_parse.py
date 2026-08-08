import re
from openai import OpenAI
from typing import List, Optional, Set, Dict, Union
from src.boundary_annotator import (
    HESITATIONS,
    is_clause_or_sentence_boundary,
    clean_word,
    get_llm_boundaries,
)

utt_re = re.compile(
    r"^\s*(?P<da>\S+)\s+(?P<spkturn>[@@]?[AB]\.\d+)\s+utt(?P<utt>\d+):\s*(?P<text>.*)$"
)

meta_re = re.compile(r"^\s*([A-Z0-9_#]+):\s*(.*)$")


def split_end_marker(text: str):
    text = text.rstrip()
    for end in ["-/", "/"]:
        if text.endswith(end):
            return text[: -len(end)].rstrip(), end
    return text, None


def clean_repairs_keep_both(s: str):
    # Remove brackets and plus entirely: "[ X + Y ]" -> "X Y"
    s = re.sub(r"\[|\]|\+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def clean_markup(s: str):
    s = clean_repairs_keep_both(s)
    # {F um, } -> um,
    s = re.sub(r"\{[A-Z]\s+([^}]+)\}", r"\1", s)

    # Convert <laughter> to [laughter], remove other <...> tags
    s = re.sub(r"<\s*laughter\s*>", " [laughter] ", s, flags=re.IGNORECASE)
    s = re.sub(r"<[^>]+>", " ", s)

    # remove # pause markers
    s = s.replace("#", " ")

    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip(" -")
    return s.strip()


def parse_file(lines):
    meta = {}
    utterances = []

    in_body = False
    for line in lines:
        if line.strip().startswith("===="):
            in_body = True
            continue

        if not in_body:
            m = meta_re.match(line)
            if m:
                meta[m.group(1)] = m.group(2).strip()
            continue

        # skip blank lines
        if not line.strip():
            continue

        m = utt_re.match(line)
        if not m:
            continue

        da = m.group("da")
        spkturn = m.group("spkturn")  # like A.9 or @@B.88
        utt_idx = int(m.group("utt"))
        text_raw = m.group("text")

        text_wo_end, end = split_end_marker(text_raw)
        speaker = spkturn.lstrip("@@")[0]
        turn = int(spkturn.lstrip("@@").split(".")[1])

        utterances.append(
            {
                "da": da,
                "speaker": speaker,
                "turn": turn,
                "utt": utt_idx,
                "text_raw": text_raw,
                "end": end,
                "text_clean": clean_markup(text_wo_end),
            }
        )

    return meta, utterances


TypedVal = Union[str, Dict[str, Optional[str]]]


def extract_typed_boundaries_from_raw(raw: str, bc_contents: Optional[List[str]] = None):
    """
    Returns:
      content_tokens: list[str] (raw without [BACKCHANNEL]/[TAKE_FLOOR])
      typed: dict[int, dict]  (idx -> {"type": "...", "content": optional str})
    Boundary idx means: boundary AFTER word idx.
    """
    toks = raw.split()
    content_tokens: List[str] = []
    typed: Dict[int, Dict[str, Optional[str]]] = {}

    word_count = 0  # counts content tokens only
    bc_ptr = 0
    bc_contents = bc_contents or []

    for t in toks:
        if t == "[BACKCHANNEL]":
            if word_count > 0:
                bc_text = bc_contents[bc_ptr] if bc_ptr < len(bc_contents) else None
                bc_ptr += 1
                typed[word_count - 1] = {"type": "BACKCHANNEL", "content": bc_text}
            continue

        if t == "[TAKE_FLOOR]":
            if word_count > 0:
                typed[word_count - 1] = {"type": "TAKE_FLOOR", "content": None}
            continue

        content_tokens.append(t)
        word_count += 1

    return content_tokens, typed


def merge_boundary_candidates_and_types(candidate_indices, typed_map):
    all_indices = sorted(set(candidate_indices) | set(typed_map.keys()))
    out = []
    for idx in all_indices:
        entry = typed_map.get(idx, None)
        if isinstance(entry, dict):
            obj = {"idx": idx, "type": entry.get("type")}
            if entry.get("content") is not None:
                obj["content"] = entry.get("content")
            out.append(obj)
        else:
            out.append({"idx": idx, "type": None})
    return out


def _map_llm_boundaries_to_original_indices(
    original_tokens: List[str], marked_tokens: List[str]
) -> Set[int]:
    """
    More robust than naive index matching.
    Greedy alignment using clean_word() and maps any '|' in marked token to the matched original token index.
    Boundary index means: boundary AFTER word idx (idx refers to original token position).
    """
    llm_boundary_idxs: Set[int] = set()
    i = 0  # original pointer
    j = 0  # marked pointer

    # If token counts match, fast path similar to boundary_annotator.py
    if len(marked_tokens) == len(original_tokens):
        for k in range(min(len(original_tokens), len(marked_tokens))):
            if "|" in marked_tokens[k] and k < len(original_tokens):
                llm_boundary_idxs.add(k)
        return llm_boundary_idxs

    # Otherwise, greedy align by (rough) token identity
    while i < len(original_tokens) and j < len(marked_tokens):
        m_tok = marked_tokens[j]
        m_has_bar = "|" in m_tok
        m_norm = m_tok.replace("|", "")
        m_clean = clean_word(m_norm)
        o_clean = clean_word(original_tokens[i])

        if m_clean == o_clean and (m_clean != "" or m_norm.strip() == original_tokens[i].strip()):
            if m_has_bar:
                llm_boundary_idxs.add(i)
            i += 1
            j += 1
            continue

        # Try to recover from mismatches: advance the side that looks "emptier" or less informative
        # If marked token is punctuation-only or empty after cleaning, skip it
        if m_clean == "" and m_norm.strip() != "":
            j += 1
            continue
        if o_clean == "" and original_tokens[i].strip() != "":
            i += 1
            continue

        # Default: advance marked to find a match, but avoid infinite loops by also advancing original sometimes
        j += 1

    return llm_boundary_idxs


def compute_candidates_with_gpt_and_heuristics(
    content: str,
    client: Optional[OpenAI] = None,
    max_gap: int = 8,
    fill_every: int = 4,
) -> List[int]:
    """
    Returns boundary candidate indices (word indices) for `content`.

    Logic mirrors boundary_annotator.py:
      1) candidates = (heuristic boundaries) OR (LLM-marked boundaries)
         - heuristic boundary if token ends with punctuation [.,?!;] OR is hesitation
      2) consecutive cluster filter: keep 2nd, 4th, ... items within each consecutive run
      3) long-gap fallback: if distance between boundaries > max_gap, insert boundaries every `fill_every`
      4) excludes last token from boundary placement

    Args:
      content: plain text (no [BACKCHANNEL]/[TAKE_FLOOR] markers)
      client: OpenAI client, if None -> heuristics only
      max_gap: boundary_annotator uses 8
      fill_every: boundary_annotator uses 4

    Returns:
      sorted list of unique indices (each means boundary after that word)
    """
    content = (content or "").strip()
    if not content:
        return []

    original_tokens = content.split()
    if len(original_tokens) <= 1:
        return []

    # 1) LLM boundaries
    llm_boundary_idxs: Set[int] = set()
    if client is not None:
        marked_text = get_llm_boundaries(content, client)
        marked_tokens = marked_text.split() if marked_text else []
        llm_boundary_idxs = _map_llm_boundaries_to_original_indices(original_tokens, marked_tokens)

    # 2) Combine heuristics + LLM
    boundary_indices: List[int] = []
    for i, token in enumerate(original_tokens[:-1]):  # exclude last token
        clean_t = clean_word(token)
        is_heuristic = is_clause_or_sentence_boundary(token) or (clean_t in HESITATIONS)
        is_llm = i in llm_boundary_idxs
        if is_heuristic or is_llm:
            boundary_indices.append(i)

    boundary_indices = sorted(set(boundary_indices))
    if not boundary_indices:
        boundary_indices = []

    # 3) Omit logic for consecutive clusters
    filtered_indices: List[int] = []
    if boundary_indices:
        clusters: List[List[int]] = []
        current = [boundary_indices[0]]
        for k in range(1, len(boundary_indices)):
            if boundary_indices[k] == boundary_indices[k - 1] + 1:
                current.append(boundary_indices[k])
            else:
                clusters.append(current)
                current = [boundary_indices[k]]
        clusters.append(current)

        for cluster in clusters:
            if len(cluster) == 1:
                filtered_indices.append(cluster[0])
            else:
                filtered_indices.extend(cluster[1::2])  # keep 2nd, 4th, ...

    filtered_indices = sorted(set(filtered_indices))

    # 4) Length-based fallback: fill long gaps
    final_indices: List[int] = []
    last_idx = -1

    for b in sorted(filtered_indices):
        gap = b - last_idx
        if gap > max_gap:
            for k in range(last_idx + fill_every, b, fill_every):
                # keep within valid range (exclude last token index)
                if 0 <= k < len(original_tokens) - 1:
                    final_indices.append(k)
        final_indices.append(b)
        last_idx = b

    # Tail gap to end (excluding last token)
    total_tokens = len(original_tokens)
    tail_gap = (total_tokens - 1) - last_idx
    if tail_gap > max_gap:
        for k in range(last_idx + fill_every, total_tokens - 1, fill_every):
            if 0 <= k < total_tokens - 1:
                final_indices.append(k)

    # Dedup + sort
    final_indices = sorted(set(i for i in final_indices if 0 <= i < total_tokens - 1))
    return final_indices


def convert_to_streaming_dialogue_json(
    utterances,
    example_id: str = "swbd_example",
    config: dict | None = None,
    context: str = "",
    meta: dict | None = None,
    client: Optional[OpenAI] = None,
    filename: str | None = None,
):
    """
    Convert parsed SWBD-style utterances into a streaming dialogue JSON.

    Implemented rules:
      1) Backchannel detection:
         - da starts with "b" ("b", "ba", "bh", ...)  OR  (da == "%" and end != "-/")
         Backchannels are not kept as their own turns.
         We try to "insert" a [BACKCHANNEL] token into surrounding speech:

           Preferred case (common in SWBD):
             X (speaker S) ... , BC (other speaker), X continues (speaker S)
             -> merge the two X parts with " [BACKCHANNEL] " between them.

           Fallback:
             - If we cannot find same-speaker surrounding parts, attach the
               token to the nearest non-backchannel utterance of the other speaker
               (usually the previous one). If that also fails, treat as normal.

      2) If end == "-/": append " [TAKE_FLOOR]" to that utterance text.
      3) Consecutive utterances of the same speaker are merged.
      4) Each final turn is:
           {"role": "user"/"assistant", "content": ..., "history": [{"full_content": ...}]}
         Per-token scoring entries are omitted.

    Speaker mapping:
      'A' -> "assistant"
      'B' -> "user"

    Uses utterance["text_clean"] as produced by parse_file().
    """
    if config is None:
        config = {}
    if meta is None:
        meta = {}

    def is_backchannel_utt(u: dict) -> bool:
        if u.get("da", "").startswith("b"):
            return True
        if u.get("da") == "%" and u.get("end") != "-/":
            return True
        return False

    def norm_space(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip()

    def speaker_to_role(spk: str) -> str:
        # Mapping to the requested "assistant" or "user" roles
        return "assistant" if spk == "A" else "user"

    uts = [dict(u) for u in utterances]

    # Pass 1: Mark potential backchannels
    for i, u in enumerate(uts):
        if i == 0 or uts[i - 1].get("speaker") == u.get("speaker"):
            u["_is_bc"] = False
            continue
        if i == len(uts) - 1 or uts[i + 1].get("speaker") == u.get("speaker"):
            u["_is_bc"] = False
            continue
        u["_is_bc"] = is_backchannel_utt(u)

    # Pass 2: Floor logic and base text
    for i, u in enumerate(uts):
        txt = (u.get("text_clean") or "").strip()
        if u.get("end") == "-/":
            is_interruption = False
            if i + 1 < len(uts):
                next_u = uts[i + 1]
                if next_u.get("speaker") != u.get("speaker") and is_backchannel_utt(next_u):
                    is_interruption = True

            if not is_interruption and i < len(uts) - 1:
                txt = f"{txt} [TAKE_FLOOR]"
        u["_base_text"] = norm_space(txt)
        u["_bc_data"] = []  # To store list of {role, content}

    # Pass 3: Backchannel Insertion
    remove_idx = set()

    def find_non_bc(start: int, step: int) -> int | None:
        idx = start
        while 0 <= idx < len(uts):
            if not uts[idx]["_is_bc"] and uts[idx]["_base_text"]:
                return idx
            idx += step
        return None

    for i, bc in enumerate(uts):
        if not bc["_is_bc"]:
            continue

        bc_role = speaker_to_role(bc.get("speaker"))
        bc_content = bc.get("text_clean", "")
        prev_i = find_non_bc(i - 1, -1)
        next_i = find_non_bc(i + 1, +1)
        inserted = False

        # Preferred: Sandwich
        if prev_i is not None and next_i is not None:
            prev_u = uts[prev_i]
            next_u = uts[next_i]
            if prev_u["speaker"] == next_u["speaker"] and prev_u["speaker"] != bc["speaker"]:
                prev_u["_base_text"] = norm_space(
                    prev_u["_base_text"] + " [BACKCHANNEL] " + next_u["_base_text"]
                )
                # Record BC data in the host utterance
                prev_u["_bc_data"].append({"role": bc_role, "content": bc_content})
                next_u["_base_text"] = ""
                remove_idx.add(next_i)
                inserted = True

        # Fallback: Attach to nearest
        if not inserted:
            attach_i = None
            if prev_i is not None and uts[prev_i]["speaker"] != bc["speaker"]:
                attach_i = prev_i
                uts[attach_i]["_base_text"] = norm_space(
                    uts[attach_i]["_base_text"] + " [BACKCHANNEL]"
                )
                uts[attach_i]["_bc_data"].append({"role": bc_role, "content": bc_content})
                inserted = True
            elif next_i is not None and uts[next_i]["speaker"] != bc["speaker"]:
                attach_i = next_i
                uts[attach_i]["_base_text"] = norm_space(
                    "[BACKCHANNEL] " + uts[attach_i]["_base_text"]
                )
                uts[attach_i]["_bc_data"].append({"role": bc_role, "content": bc_content})
                inserted = True

        if inserted:
            remove_idx.add(i)
        else:
            bc["_is_bc"] = False

    # Final Merge and Indexing
    history = []
    current_turn = None

    for i, u in enumerate(uts):
        if i in remove_idx or not u["_base_text"]:
            continue

        role = speaker_to_role(u["speaker"])
        full_content = u["_base_text"]
        bc_list = u["_bc_data"]

        if current_turn and current_turn["role"] == role:
            current_turn["_full_raw"] = norm_space(current_turn["_full_raw"] + " " + full_content)
            current_turn["_bc_data"].extend(bc_list)
        else:
            # Assign indices to first batch of backchannels
            for idx, bc_item in enumerate(bc_list):
                bc_item["idx"] = idx

            current_turn = {
                "role": role,
                "content": "",  # placeholder
                "_full_raw": full_content,
                "_bc_data": bc_list,  # keep internally only
                "history": [],
            }
            history.append(current_turn)

    # Cleanup internal tags for final output
    for turn in history:
        raw = turn.pop("_full_raw")
        bc_list = turn.pop("_bc_data", [])
        bc_texts = [b.get("content", "") for b in bc_list]  # in encounter order

        content_tokens, typed_map = extract_typed_boundaries_from_raw(raw, bc_contents=bc_texts)
        content = norm_space(" ".join(content_tokens))

        candidate_indices = compute_candidates_with_gpt_and_heuristics(content, client)

        turn["content"] = content
        turn["boundary_word_indices"] = merge_boundary_candidates_and_types(
            candidate_indices, typed_map
        )

    out = {
        "example_id": meta["FILENAME"] if "FILENAME" in meta else filename,
        "speakers": ["user", "assistant"],
        "config": config,
        "history": history,
    }
    if context:
        out["context"] = context
    if meta:
        out["meta"] = meta

    return out


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python swbd_parse.py <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    meta, utterances = parse_file(lines)

    print("Metadata:")
    for k, v in meta.items():
        print(f"{k}: {v}")

    print("\nUtterances:")
    for utt in utterances:
        print(utt)

    dialogue_json = convert_to_streaming_dialogue_json(
        utterances,
        example_id="swbd_example_001",
        context="This is a sample conversation from the Switchboard corpus.",
        meta=meta,
    )
    import json

    print("\nDialogue JSON:")
    print(json.dumps(dialogue_json, indent=2))
