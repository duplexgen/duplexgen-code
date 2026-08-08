"""
Prepare text dialogue corpus for the spoken dialogue generation pipeline.

Parses original corpora (MultiWoZ, CraigslistBargain, SocraTeach, PersuasionForGood,
AnthropicInterviewer, SODA) into the JSON format consumed by src/synthesis/run.py:

    {
        "example_id": "...",
        "speakers": ["user", "assistant"],
        "config": {},
        "history": [{"role": "user"|"assistant", "content": "..."}],
        "context": "...",
        "meta": {"dataset": "...", "split": "..."}
    }

Output structure:
    {save_dir}/text_dialogue_{scenario}/{split}/*.json

Splitting strategy:
    - Datasets with explicit train/test splits (multiwoz, negotiator, soda):
      follow the native split.
    - Datasets without explicit splits (interviewer, persuader, socraticlm):
      use example IDs from --test_ref_dir as the test set, everything else
      goes to train. If test exceeds --max_test, excess is reassigned to train.

Usage:
    # All datasets
    python src/prepare_corpus.py -d all --save_dir results/annotated_dialogues

    # Single dataset, custom sample counts
    python src/prepare_corpus.py -d soda --max_train 100 --max_test 50

    # Skip LLM filtering for SODA (use all dialogues)
    python src/prepare_corpus.py -d soda --no_soda_filter
"""

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple, Union

from tqdm import tqdm

Turn = Tuple[str, str]

# Datasets that provide their own train/test split
EXPLICIT_SPLIT_DATASETS = {"multiwoz", "negotiator", "soda"}


# ---------------------------------------------------------------------------
# Reference test set loader
# ---------------------------------------------------------------------------


def load_test_ref_ids(test_ref_dir: Path, dataset: str) -> Set[str]:
    """Load example IDs from reference test set at
    {test_ref_dir}/text_dialogue_{dataset}/test/*.json"""
    test_dir = test_ref_dir / f"text_dialogue_{dataset}" / "test"
    if not test_dir.is_dir():
        return set()

    ids = set()
    for f in test_dir.glob("*.json"):
        try:
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
            eid = data.get("example_id", f.stem)
            ids.add(str(eid))
        except Exception:
            ids.add(f.stem)
    return ids


# ---------------------------------------------------------------------------
# Dataset iterators
# ---------------------------------------------------------------------------
# Datasets WITHOUT explicit splits yield split_label=None.
# Datasets WITH explicit splits yield "train" or "test".
# ---------------------------------------------------------------------------


def iter_interviewer(
    split: str = "workforce",
) -> Iterator[Tuple[str, List[Turn], str, Optional[str]]]:
    """Yields (example_id, turns, context, None). No explicit split."""
    from datasets import load_dataset

    ds = load_dataset("Anthropic/AnthropicInterviewer")
    samples = ds[split]

    pattern = re.compile(
        r"(Assistant|Claude|AI|User):\s*(.*?)(?=\n(?:Assistant|Claude|AI|User):|$)",
        flags=re.S | re.I,
    )

    for i in range(len(samples)):
        item = samples[i]
        ex_id = item.get("transcript_id", f"work_{i:04d}")
        raw = item.get("text", "")
        if not raw:
            continue

        turns: List[Turn] = []
        for speaker, content in pattern.findall(raw):
            role = (
                "assistant" if speaker.strip().upper() in ("ASSISTANT", "CLAUDE", "AI") else "user"
            )
            content = content.strip()
            if content:
                turns.append((role, content))

        if not turns:
            continue

        context = (
            f"An AI user researcher at Anthropic is interviewing a human participant "
            f"regarding their experiences with AI in the context of {split}."
        )

        yield ex_id, turns, context, None


def iter_multiwoz(
    root: Union[str, Path] = "datasets/multiwoz/data/MultiWOZ_2.2",
) -> Iterator[Tuple[str, List[Turn], str, str]]:
    """Yields (example_id, turns, context, split_label). Explicit split."""
    root = Path(root)
    context = "User is interacting with assistant to get information and make reservations for various service."

    for split_label, subdir in [("train", "train"), ("test", "test")]:
        all_data = []
        for f in sorted((root / subdir).glob("*.json")):
            with open(f, encoding="utf-8") as fh:
                all_data.extend(json.load(fh))

        for i, entry in enumerate(all_data):
            ex_id = entry.get("dialogue_id", f"mwoz_{i}").rstrip(".json")
            turns: List[Turn] = []
            for t in entry.get("turns", []):
                role = "user" if t.get("speaker", "").upper() == "USER" else "assistant"
                utt = t.get("utterance", "").strip()
                if utt:
                    turns.append((role, utt))
            if turns:
                yield ex_id, turns, context, split_label


def iter_negotiator(
    root: Union[str, Path] = "datasets/CraigslistBargain",
) -> Iterator[Tuple[str, List[Turn], str, str]]:
    """Yields (example_id, turns, context, split_label). Explicit split."""
    root = Path(root)

    for split_label, fname in [("train", "train.json"), ("test", "test.json")]:
        fpath = root / fname
        if not fpath.exists():
            continue
        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)

        for i, entry in enumerate(data):
            ex_id = entry.get("uuid", f"neg_{i}")
            scenario = entry.get("scenario", {})
            kbs = scenario.get("kbs", [])

            ctx_parts = [f"Category: {scenario.get('category', 'N/A')}"]
            for kb in kbs:
                personal = kb.get("personal", {})
                role_label = {"buyer": "user (buyer)", "seller": "assistant (seller)"}.get(
                    personal.get("Role", ""), "unknown"
                )
                target = personal.get("Target", "N/A")
                title = kb.get("item", {}).get("Title", "this item")
                ctx_parts.append(f"Role: {role_label} | Target Price: {target} | Item: {title}")

            context = "\n".join(ctx_parts)

            turns: List[Turn] = []
            for event in entry.get("events", []):
                if event.get("action") != "message":
                    continue
                role = "user" if event.get("agent") == 0 else "assistant"
                content = event.get("data", "").strip()
                if content:
                    turns.append((role, content))

            if turns:
                yield ex_id, turns, context, split_label


def iter_socraticlm(
    path: Union[str, Path] = "datasets/SocraticLM/data/SocraTeach_multi.json",
    max_variants: int = 1,
) -> Iterator[Tuple[str, List[Turn], str, Optional[str]]]:
    """Yields (example_id, turns, context, None). No explicit split."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    for key, item in data.items():
        question = (item.get("question") or "").strip()
        context = (
            f"User is solving a word problem. Assistant acts as a tutor, "
            f"using the problem statement to guide step by step.\nQuestion:\n{question}"
        )

        dialogues = item.get("dialogues", {})
        variant_keys = sorted(dialogues.keys())[:max_variants]

        for vkey in variant_keys:
            ex_id = f"{key}__{vkey}"
            dialog_list = dialogues[vkey]
            turns: List[Turn] = []
            for obj in dialog_list:
                if "END" in obj:
                    if obj.get("system", "").strip():
                        turns.append(("assistant", obj["system"].strip()))
                    continue
                if obj.get("system", "").strip():
                    turns.append(("assistant", obj["system"].strip()))
                if obj.get("user", "").strip():
                    turns.append(("user", obj["user"].strip()))

            if turns:
                yield ex_id, turns, context, None


def iter_persuader(
    path: Union[str, Path] = "datasets/persuasionforgood/data/FullData/full_dialog.csv",
) -> Iterator[Tuple[str, List[Turn], str, Optional[str]]]:
    """Yields (example_id, turns, context, None). No explicit split."""
    path = Path(path)

    # Group rows by conversation ID (B2 column)
    conversations: Dict[str, List] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) < 5:
                continue
            conv_id = row[4]
            conversations[conv_id].append(row)

    context = (
        "A persuasion dialogue where one participant tries to convince "
        "the other to donate to a children's charity."
    )

    for conv_id in sorted(conversations.keys()):
        rows = conversations[conv_id]
        rows.sort(key=lambda r: int(r[0]))

        turns: List[Turn] = []
        for row in rows:
            utterance = row[1].strip()
            speaker_id = int(row[3])  # B4: 0 = persuader, 1 = persuadee
            role = "assistant" if speaker_id == 0 else "user"
            if utterance:
                turns.append((role, utterance))

        if not turns:
            continue

        yield conv_id, turns, context, None


def iter_soda(
    max_train: Optional[int] = None,
    max_test: Optional[int] = None,
    filter_competitive: bool = True,
    start_index: int = 0,
) -> Iterator[Tuple[str, List[Turn], str, str]]:
    """Yields (example_id, turns, context, split_label). Explicit split."""
    from datasets import load_dataset
    from openai import OpenAI

    ds = load_dataset("allenai/soda")

    client = None
    if filter_competitive:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    for split_label, ds_split, max_n in [
        ("train", ds["train"], max_train),
        ("test", ds["test"], max_test),
    ]:
        count = 0
        for i, item in enumerate(tqdm(ds_split, desc=f"SODA ({split_label})")):
            if i < start_index:
                continue
            if max_n is not None and count >= max_n:
                break

            dialogue = item.get("dialogue", [])
            narrative = item.get("narrative", "").strip()
            speakers = item.get("speakers", [])

            if not dialogue or len(dialogue) < 2:
                continue

            # Competitive scenario filter
            if filter_competitive and client:
                formatted = "\n".join(
                    f"Speaker {j % 2 + 1}: {msg}" for j, msg in enumerate(dialogue)
                )
                try:
                    resp = client.chat.completions.create(
                        model="gpt-4.1-mini",
                        messages=[
                            {
                                "role": "user",
                                "content": (
                                    "Analyze the following social scenario narrative. "
                                    "Determine if the interaction is COMPETITIVE (e.g., negotiation, debate, "
                                    "disagreement, conflicting goals, arguing, tension due to differences in "
                                    "opinions or mood) or NOT. "
                                    "Respond with only 'YES' if it is competitive, and 'NO' otherwise. "
                                    "Also, the first speaker is the user and the second is the assistant. "
                                    "Answer YES only if this exchange would naturally occur in a user-AI interaction.\n\n"
                                    f"Dialogue: {formatted}"
                                ),
                            }
                        ],
                        max_tokens=5,
                        temperature=0,
                    )
                    if "YES" not in resp.choices[0].message.content.strip().upper():
                        continue
                except Exception as e:
                    print(f"  SODA filter error at {i}: {e}")
                    continue

            # Replace speaker names with roles in narrative
            processed_narrative = narrative
            if len(speakers) >= 2:
                processed_narrative = re.sub(
                    rf"\b{re.escape(speakers[0])}\b", "User", processed_narrative, flags=re.I
                )
                processed_narrative = re.sub(
                    rf"\b{re.escape(speakers[1])}\b", "Assistant", processed_narrative, flags=re.I
                )

            turns: List[Turn] = []
            for j, utt in enumerate(dialogue):
                role = "user" if j % 2 == 0 else "assistant"
                turns.append((role, utt.strip()))

            context = f"Social interaction scenario: {processed_narrative}"
            ex_id = f"soda_{split_label}_{i}"
            count += 1
            yield ex_id, turns, context, split_label


# ---------------------------------------------------------------------------
# Write helper
# ---------------------------------------------------------------------------


def write_dialogue(
    save_dir: Path,
    dataset: str,
    split_label: str,
    example_id: str,
    turns: List[Turn],
    context: str,
) -> Path:
    out_dir = save_dir / f"text_dialogue_{dataset}" / split_label
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize filename
    safe_id = re.sub(r"[^\w\-.]", "_", str(example_id))
    out_path = out_dir / f"{safe_id}.json"

    obj = {
        "example_id": str(example_id),
        "speakers": ["user", "assistant"],
        "config": {},
        "history": [{"role": r, "content": c} for r, c in turns],
        "context": context if isinstance(context, str) else json.dumps(context),
        "meta": {
            "dataset": dataset,
            "split": split_label,
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DATASET_CHOICES = ["interviewer", "multiwoz", "negotiator", "persuader", "socraticlm", "soda"]


def main():
    parser = argparse.ArgumentParser(
        description="Prepare text dialogue corpus from original datasets."
    )
    parser.add_argument(
        "-d", "--dataset", type=str, choices=DATASET_CHOICES + ["all"], required=True
    )
    parser.add_argument("--save_dir", type=str, default="results/annotated_dialogues")
    parser.add_argument(
        "--max_train",
        type=int,
        default=None,
        help="Max train dialogues per dataset (default: no limit)",
    )
    parser.add_argument(
        "--max_test", type=int, default=50, help="Max test dialogues per dataset (default: 50)"
    )
    parser.add_argument(
        "--test_ref_dir",
        type=str,
        default="results/annotated_dialogues_260314",
        help="Reference directory for test set IDs (for datasets without explicit splits)",
    )
    parser.add_argument(
        "--no_soda_filter", action="store_true", help="Skip LLM competitive filter for SODA"
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Start index for SODA dataset iteration (to resume)",
    )
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    test_ref_dir = Path(args.test_ref_dir)
    datasets = DATASET_CHOICES if args.dataset == "all" else [args.dataset]

    summary = {}

    for ds_name in datasets:
        print(f"\n{'='*60}")
        print(f"  Processing: {ds_name}")
        print(f"{'='*60}")

        counts = {"train": 0, "test": 0}
        has_explicit_split = ds_name in EXPLICIT_SPLIT_DATASETS

        # For datasets without explicit splits, load reference test IDs
        test_ref_ids: Set[str] = set()
        if not has_explicit_split:
            test_ref_ids = load_test_ref_ids(test_ref_dir, ds_name)
            if test_ref_ids:
                print(f"  Loaded {len(test_ref_ids)} reference test IDs from {test_ref_dir}")
            else:
                print("  No reference test IDs found; all samples go to train")

        # Create iterator (no max_samples — we iterate all and assign splits)
        if ds_name == "interviewer":
            it = iter_interviewer()
        elif ds_name == "multiwoz":
            it = iter_multiwoz()
        elif ds_name == "negotiator":
            it = iter_negotiator()
        elif ds_name == "socraticlm":
            it = iter_socraticlm()
        elif ds_name == "persuader":
            it = iter_persuader()
        elif ds_name == "soda":
            it = iter_soda(
                max_train=args.max_train,
                max_test=args.max_test,
                filter_competitive=not args.no_soda_filter,
                start_index=args.start_index,
            )
        else:
            continue

        for ex_id, turns, context, split_label in it:
            # Determine split for datasets without explicit splits
            if split_label is None:
                if str(ex_id) in test_ref_ids and counts["test"] < (args.max_test or float("inf")):
                    split_label = "test"
                else:
                    split_label = "train"

            # Enforce limits
            if (
                split_label == "test"
                and args.max_test is not None
                and counts["test"] >= args.max_test
            ):
                # Excess test samples go to train
                split_label = "train"

            if (
                split_label == "train"
                and args.max_train is not None
                and counts["train"] >= args.max_train
            ):
                continue

            write_dialogue(save_dir, ds_name, split_label, ex_id, turns, context)
            counts[split_label] += 1

        summary[ds_name] = counts
        print(f"  train: {counts['train']}, test: {counts['test']}")

    # Print summary
    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    print(f"  {'Dataset':<15} {'Train':>6} {'Test':>6} {'Total':>7}")
    print(f"  {'-'*36}")
    total_train = total_test = 0
    for ds_name, c in summary.items():
        print(f"  {ds_name:<15} {c['train']:>6} {c['test']:>6} {c['train']+c['test']:>7}")
        total_train += c["train"]
        total_test += c["test"]
    print(f"  {'-'*36}")
    print(f"  {'TOTAL':<15} {total_train:>6} {total_test:>6} {total_train+total_test:>7}")
    print(f"\n  Output: {save_dir}/")


if __name__ == "__main__":
    main()
