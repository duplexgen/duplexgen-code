import json
import glob
import re
import os
import argparse
from openai import OpenAI
from tqdm import tqdm
from collections import defaultdict

HESITATIONS = {"um", "uh", "umm", "uhh", "hm", "hmm", "huh"}


def is_clause_or_sentence_boundary(token):
    return re.search(r"[.,?!;]+$", token) is not None


def clean_word(token):
    return re.sub(r"[^\w\s]", "", token).lower()


def get_llm_boundaries(text, client):
    """
    Prompts the LLM to mark boundaries with '|'.
    """
    if not text.strip():
        return ""

    prompt = f"""
    # Role and Objective
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
    Input variable: {text} (string containing the spoken dialogue transcript).

    # Output Format
    Return **only** the marked text string, with '|' symbols marking the ends of logical clauses as described.

    ### Example
    **Input:** Well, I was thinking maybe we can talk later if you have time.
    **Output:** Well,| I was thinking| maybe we can talk later| if you have time.|
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a linguistic expert. Insert '|' at clause boundaries.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"LLM Error: {e}")
        return text


def annotate_dialogue(file_path, client):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for turn in data.get("history", []):
        # Only process if the role is 'user'
        if turn.get("role") != "user":
            continue

        content = turn.get("content", "")
        if not content:
            continue

        # 1. Get LLM suggested boundaries
        marked_text = get_llm_boundaries(content, client)

        # Split both original and marked text to map indices
        original_tokens = content.split()
        marked_tokens = marked_text.split()

        boundary_indices = []

        # 2. Identify indices from LLM and Heuristics
        for i, token in enumerate(original_tokens[:-1]):  # Exclude last token
            clean_t = clean_word(token)

            # Check Heuristics
            is_heuristic_boundary = is_clause_or_sentence_boundary(token) or clean_t in HESITATIONS

            # Check LLM (matching index in marked_tokens)
            is_llm_boundary = False
            if i < len(marked_tokens):
                if "|" in marked_tokens[i]:
                    is_llm_boundary = True

            if is_heuristic_boundary or is_llm_boundary:
                boundary_indices.append(i)

        # 3. Omit Logic: Handle consecutive clusters
        filtered_indices = []
        if boundary_indices:
            clusters = []
            current_cluster = [boundary_indices[0]]
            for i in range(1, len(boundary_indices)):
                if boundary_indices[i] == boundary_indices[i - 1] + 1:
                    current_cluster.append(boundary_indices[i])
                else:
                    clusters.append(current_cluster)
                    current_cluster = [boundary_indices[i]]
            clusters.append(current_cluster)

            for cluster in clusters:
                if len(cluster) == 1:
                    filtered_indices.append(cluster[0])
                else:
                    # For consecutive clusters: keep the 2nd, 4th, etc.
                    filtered_indices.extend(cluster[1::2])

        # 4. Length-Based Fallback: Fill long gaps (>8 tokens)
        final_indices = []
        last_idx = -1  # Start tracking from before the first word

        # Iterate through currently detected boundaries
        for boundary in sorted(filtered_indices):
            # Calculate gap size (number of tokens between last boundary and this one)
            gap = boundary - last_idx

            # If span is longer than 8 consecutive tokens
            if gap > 8:
                # Insert slots approx every 4 tokens
                # range starts at last_idx + 4, stops before current boundary, steps by 4
                for k in range(last_idx + 4, boundary, 4):
                    final_indices.append(k)

            final_indices.append(boundary)
            last_idx = boundary

        # Handle the tail: span from last boundary to end of sentence
        total_tokens = len(original_tokens)
        tail_gap = (total_tokens - 1) - last_idx

        if tail_gap > 8:
            # We stop before the very last token to avoid boundary at EOF if not intended,
            # but we cover the span.
            for k in range(last_idx + 4, total_tokens - 1, 4):
                final_indices.append(k)

        # Sort and deduplicate just in case, though logic should produce sorted order
        turn["boundary_word_indices"] = sorted(list(set(final_indices)))

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        type=str,
        default="*",
        help="Scenario name (e.g. interviewer). Defaults to all scenarios.",
    )
    parser.add_argument(
        "--split", type=str, default="train", help="Dataset split (e.g. train, test)."
    )
    parser.add_argument(
        "--input_root",
        type=str,
        default="data/spoken_dialogues",
        help="Root holding text_dialogue_<scenario>/<split>/*.json spoken-style dialogues "
        "(e.g. a local checkout of the DuplexGen/duplexgen-corpus dialogues split).",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/annotated_dialogues",
        help="Where to write slot-annotated dialogues.",
    )
    args = parser.parse_args()

    input_pattern = f"{args.input_root}/text_dialogue_{args.scenario}/{args.split}/*.json"

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    all_files = glob.glob(input_pattern, recursive=True)

    # Collect files per dataset
    files_by_dataset = defaultdict(list)
    for file_path in all_files:
        dataset_dir = next(
            (p for p in file_path.split(os.sep) if p.startswith("text_dialogue_")), None
        )
        if dataset_dir:
            files_by_dataset[dataset_dir].append(file_path)

    selected_files = sorted(all_files)

    print(
        f"Found {len(files_by_dataset)} datasets. Processing all files -> total {len(selected_files)} files."
    )

    for file_path in tqdm(selected_files):
        dataset_dir = next(
            (p for p in file_path.split(os.sep) if p.startswith("text_dialogue_")), ""
        )
        output_dir = os.path.join(args.output_root, dataset_dir, args.split)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.basename(file_path))

        if os.path.exists(output_path):
            continue

        annotated_data = annotate_dialogue(file_path, client)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(annotated_data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
