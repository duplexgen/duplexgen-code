import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

from tqdm import tqdm
from openai import OpenAI

from src.llm_client import make_client
from src.speechify_datasets import (
    iter_negotiator,
    iter_multiwoz,
    iter_socrateach,
    iter_interviewer,
    iter_persuader,
    iter_soda,
)
from src.speechify_datasets import get_sampled_indices
from src.speechify_core import speechify_full_dialogue, CONFIG
from datasets import load_dataset

# The six scenarios of the released corpus, matching the downstream stages
# (src/synthesis/run.py, src/prepare_corpus.py, tools/unpack_corpus.py).
DATASET_CHOICES = [
    "socraticlm",
    "multiwoz",
    "interviewer",
    "negotiator",
    "persuader",
    "soda",
]

# Where each locally-held corpus sits under --data-root. Datasets absent from
# this table load from the Hub (interviewer, soda) or take --input_path
# directly (persuader), and so need no --data-root at all.
LOCAL_CORPUS_PATHS = {
    "multiwoz": {
        "train": "multiwoz/data/MultiWOZ_2.2/train/",
        "test": "multiwoz/data/MultiWOZ_2.2/test/",
    },
    "negotiator": {
        "train": "CraigslistBargain/train.json",
        "test": "CraigslistBargain/test.json",
    },
    "socraticlm": {"all": "SocraticLM/data/SocraTeach_multi.json"},
}
DATA_ROOT_DATASETS = set(LOCAL_CORPUS_PATHS)

# Corpora shipping one file per split, each read by its own loader.
SEPARATE_FILE_LOADERS = {
    "multiwoz": iter_multiwoz,
    "negotiator": iter_negotiator,
}

SPLITS = ("train", "test")
DEFAULT_SAMPLES_PER_SPLIT = 25


@dataclass(frozen=True)
class SampleBudget:
    """How many dialogues to convert per split; None means every one of them."""

    train: Optional[int]
    test: Optional[int]

    @classmethod
    def from_args(cls, args) -> "SampleBudget":
        return cls(train=_as_limit(args.max_train_samples), test=_as_limit(args.max_test_samples))

    def for_split(self, split: str) -> Optional[int]:
        return self.train if split == "train" else self.test

    @property
    def total(self) -> Optional[int]:
        if self.train is None or self.test is None:
            return None
        return self.train + self.test


def _as_limit(count: Optional[int]) -> Optional[int]:
    """Normalize a CLI count, where anything non-positive means 'no limit'."""
    return None if count is None or count <= 0 else count


def take(items: list, limit: Optional[int]) -> list:
    """Subsample at even spacing, so a limit still covers the whole source."""
    return [items[i] for i in get_sampled_indices(len(items), limit)]


def split_by_file(dataset_name: str, args, budget: SampleBudget) -> List[Tuple[str, list]]:
    """Strategy 1: the corpus ships train and test as separate files."""
    load = SEPARATE_FILE_LOADERS[dataset_name]
    return [
        (split, take(list(load(corpus_path(dataset_name, split, args))), budget.for_split(split)))
        for split in SPLITS
    ]


def split_by_label(labeled: list, budget: SampleBudget) -> List[Tuple[str, list]]:
    """Strategy 2: one file whose entries carry their own split label.

    Partition before sampling, so each split is drawn from its own pool
    rather than from whatever the source ordering happened to surface.
    """
    return [
        (
            split,
            take([item[:3] for item in labeled if item[3] == split], budget.for_split(split)),
        )
        for split in SPLITS
    ]


def split_in_halves(items: list, budget: SampleBudget) -> List[Tuple[str, list]]:
    """Strategy 3: one undifferentiated stream that we cut into splits ourselves.

    Sampling happens over dialogues rather than over source records, so a
    --max_variants above 1 still yields the requested per-split counts.
    """
    picked = take(items, budget.total)
    mid = len(picked) // 2 if budget.total is None else budget.train
    return [("train", picked[:mid]), ("test", picked[mid:])]


def corpus_path(dataset_name: str, key: str, args) -> str:
    """Resolve one local corpus location under --data-root.

    Single-file corpora honour an --input_path override; the split-per-file
    ones cannot, since one path can't stand in for both splits.
    """
    if key == "all" and args.input_path:
        return args.input_path
    if not args.data_root:
        raise ValueError(
            f"--data-root is required for --dataset {dataset_name}: pass the "
            "directory holding your local copies of the raw third-party "
            "datasets (see the Stage 1 table in the README)."
        )
    return os.path.join(args.data_root, LOCAL_CORPUS_PATHS[dataset_name][key])


def split_output_dir(save_dir: str, dataset_name: str, split_label: str, test_parse: bool) -> Path:
    """Where one split's output goes.

    --test-parse dumps unconverted source text, so it gets its own tree: at the
    real path, the resume skip below would mistake a dump for finished work and
    convert nothing. The tree deliberately avoids the `text_dialogue_` prefix,
    which Stages 2-4 glob for to discover scenarios.
    """
    if test_parse:
        return Path(save_dir) / "parsed_source" / dataset_name / split_label
    return Path(save_dir) / f"text_dialogue_{dataset_name}" / split_label


def is_converted(path: Path) -> bool:
    """Whether an existing output file is a finished spoken-style conversion.

    Guards the resume skip, so a dump left at the real path by an older version
    -- or a file truncated by an interrupted run -- gets redone rather than
    counted as done.
    """
    try:
        with path.open(encoding="utf-8") as rf:
            return json.load(rf).get("meta", {}).get("style") == "spoken"
    except (OSError, ValueError):
        return False


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as wf:
        json.dump(obj, wf, indent=2, ensure_ascii=False)


def get_iterators(dataset_name: str, args, client) -> List[Tuple[str, list]]:
    """Return [(split_name, samples)], each already cut to the sample budget."""
    budget = SampleBudget.from_args(args)

    if dataset_name in SEPARATE_FILE_LOADERS:
        return split_by_file(dataset_name, args, budget)

    if dataset_name == "socraticlm":
        path = corpus_path(dataset_name, "all", args)
        return split_by_label(list(iter_socrateach(path, max_variants=args.max_variants)), budget)

    if dataset_name == "persuader":
        if not args.input_path:
            raise ValueError(
                "--input_path is required for --dataset persuader: pass the path "
                "to the raw DailyPersuasion dataset json file you downloaded "
                "(see README for the source and expected filename)."
            )
        return split_in_halves(
            list(iter_persuader(args.input_path, max_variants=args.max_variants)), budget
        )

    if dataset_name == "interviewer":
        ds = load_dataset("Anthropic/AnthropicInterviewer")
        return split_in_halves(list(iter_interviewer(ds, split=args.interviewer_subset)), budget)

    if dataset_name == "soda":
        # SODA is the one source filtered by an LLM as it streams, so its
        # budget has to be pushed into the iterator to stop it early rather
        # than applied to a materialized list.
        ds = load_dataset("allenai/soda")
        return [
            (
                split,
                list(
                    iter_soda(
                        ds[split],
                        max_samples=budget.for_split(split),
                        client=client,
                        model=args.llm_model_name,
                    )
                ),
            )
            for split in SPLITS
        ]

    raise ValueError(f"unsupported --dataset: {dataset_name}")


def process_dataset(dataset_name: str, args, client: Optional[OpenAI]):
    splits = get_iterators(dataset_name, args, client)

    for split_label, samples in splits:
        save_dir = split_output_dir(args.save_dir, dataset_name, split_label, args.test_parse)
        save_dir.mkdir(parents=True, exist_ok=True)

        verb = "Parsing" if args.test_parse else "Converting"
        for ex_id, turns, context in tqdm(samples, desc=f"{verb} {dataset_name} ({split_label})"):
            filename = f"{ex_id}_{split_label}.json" if dataset_name == "soda" else f"{ex_id}.json"
            out_path = save_dir / filename
            if out_path.exists() and (args.test_parse or is_converted(out_path)):
                continue

            if not turns:
                continue

            if args.test_parse:
                write_json(
                    out_path,
                    {
                        "example_id": ex_id,
                        "speakers": ["user", "assistant"],
                        "context": context,
                        "history": turns,
                        "meta": {"split": split_label},
                    },
                )
                continue

            utterances = speechify_full_dialogue(
                llm_model_name=args.llm_model_name,
                client=client,
                source_turns=turns,
                context=context,
                temperature=args.temperature,
                concise=args.concise,
            )

            result = {
                "example_id": ex_id,
                "speakers": ["user", "assistant"],
                "config": CONFIG,
                "history": utterances,
                "context": context,
                "meta": {
                    "style": "spoken",
                    "disfluency_target": "user",
                    "dataset": dataset_name,
                    "split": split_label,
                },
            }
            write_json(out_path, result)


def main():
    parser = argparse.ArgumentParser(
        description="Convert text dialogues to speech-friendly dialogues."
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        choices=DATASET_CHOICES + ["all"],
        required=True,
        help="Scenario to convert, or 'all' for every one of them.",
    )
    parser.add_argument(
        "--data-root",
        "--data_root",
        type=str,
        default=None,
        help="Root dir containing the raw third-party dataset copies "
        "(multiwoz/, CraigslistBargain/, SocraticLM/). These are "
        "third-party datasets not redistributed in this repo -- "
        "download them yourself (see README) and point --data-root "
        "at the directory that contains them. Not needed for "
        "--dataset interviewer/soda (load from the Hugging Face Hub) "
        "or persuader (pass --input_path directly instead).",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=None,
        help="Path to a single-file corpus, overriding --data-root. Required "
        "for persuader; optional for socraticlm.",
    )
    parser.add_argument("--save_dir", type=str, default="results/")
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=DEFAULT_SAMPLES_PER_SPLIT,
        help="Dialogues to convert for the train split, sampled at even "
        "spacing across the source. 0 converts every one of them. "
        "Each dialogue costs one LLM call. (default: %(default)s)",
    )
    parser.add_argument(
        "--max_test_samples",
        type=int,
        default=DEFAULT_SAMPLES_PER_SPLIT,
        help="Same, for the test split. 0 converts every one of them. " "(default: %(default)s)",
    )
    parser.add_argument(
        "--max_variants",
        type=int,
        default=1,
        help="Dialogue variants to keep per source record, for the corpora "
        "that ship several retellings of one scenario (socraticlm, "
        "persuader). (default: %(default)s)",
    )
    parser.add_argument(
        "--interviewer_subset",
        type=str,
        default="workforce",
        help="Which Anthropic Interviewer subset to convert. (default: %(default)s)",
    )
    parser.add_argument("--llm_model_name", type=str, default="gpt-4.1")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for the conversion. (default: %(default)s)",
    )
    parser.add_argument("--api_key", type=str, default="EMPTY")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--concise", action="store_true", default=False)
    parser.add_argument(
        "--test-parse",
        "--test_parse",
        action="store_true",
        default=False,
        help="Parse and dump the source dialogues without calling the LLM.",
    )
    args = parser.parse_args()

    client = None
    if not args.test_parse:
        client = make_client(args.llm_model_name, args.api_key, args.base_url)

    datasets_to_run = DATASET_CHOICES if args.dataset == "all" else [args.dataset]

    for ds_name in datasets_to_run:
        process_dataset(ds_name, args, client)


if __name__ == "__main__":
    main()
