import argparse
import json
import os
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

from tqdm import tqdm

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

# Note: build_boundary_annotation_queue is removed as the new logic
# handles detection dynamically within speechify_turn_by_turn
from src.llm_client import make_client
from src.synthesis.core import speechify_turn_by_turn
from src.synthesis.prompts import _normalize_ws


# HF classification model support (lazy import to keep synthesis env lightweight)
def _load_hf_tt_model(model_name_or_path: str, peft_path: Optional[str], load_in_4bit: bool):
    from src.inference_turntaking_hf import get_token_classification_model
    from transformers import AutoTokenizer

    # For PEFT adapter checkpoints, load the tokenizer from the base model to
    # avoid a transformers version mismatch where extra_special_tokens is saved
    # as a list instead of the expected dict.
    tokenizer_path = model_name_or_path
    adapter_cfg_path = os.path.join(model_name_or_path, "adapter_config.json")
    if os.path.isfile(adapter_cfg_path):
        with open(adapter_cfg_path, "r", encoding="utf-8") as _f:
            _adapter_cfg = json.load(_f)
        tokenizer_path = _adapter_cfg.get("base_model_name_or_path", model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = get_token_classification_model(
        model_name_or_path=model_name_or_path,
        load_in_4bit=load_in_4bit,
        peft_path=peft_path,
    )
    return model, tokenizer


DATASET_CHOICES = ["socraticlm", "multiwoz", "interviewer", "negotiator", "persuader", "soda"]
SPLIT_CHOICES = ["train", "test"]


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as rf:
        return json.load(rf)


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as wf:
        json.dump(obj, wf, indent=2, ensure_ascii=False)


def iter_input_jsons(root: Path, dataset: str, split: str) -> List[Path]:
    """
    results/llm_qwen/text_dialogue_{DATASET}/{SPLIT}/*.json
    """
    in_dir = root / f"text_dialogue_{dataset}" / split
    return sorted(in_dir.glob("*.json"))


def ensure_source_turns(example: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    Converts example["history"] (list of {"role","content",...}) into (role, content) tuples.
    """
    hist = example.get("history", [])
    turns: List[Tuple[str, str]] = []
    for h in hist:
        role = h.get("role")
        content = h.get("content", "")
        if role not in ("user", "assistant"):
            raise ValueError(f"Invalid role in history: {role}")
        turns.append((role, content))
    return turns


def extract_scenario(example: Dict[str, Any]) -> str:
    """
    Attempts to find a scenario description from common JSON keys.
    """
    if example.get("context"):
        return str(example["context"])
    return "A generic conversation."


def main():
    parser = argparse.ArgumentParser(
        description="Apply turn-taking predictor (TOKEN_BC/TOKEN_FT insertion) using dynamic boundary detection and probability estimation."
    )
    parser.add_argument(
        "-d", "--dataset", type=str, choices=DATASET_CHOICES + ["all"], required=True
    )
    parser.add_argument("-s", "--split", type=str, choices=SPLIT_CHOICES, required=True)

    # No default: this used to name a Hub repo path pointing at the *annotations*
    # half, which is neither a local directory nor the half Stage 4 reads. Nothing
    # matched, every dataset was skipped with a WARN, and the run exited 0 having
    # generated nothing. Point it at the tree tools/unpack_corpus.py --kind
    # dialogues writes (data-dialogues/ by default), or at Stage 1 output.
    parser.add_argument("--input_root", type=str, required=True)
    parser.add_argument("--save_root", type=str, default="outputs/generated_dialogues_with_tt")

    parser.add_argument(
        "--max_dialogues",
        type=int,
        default=1000,
        help="Max dialogues to process per dataset/split (for quick testing)",
    )

    # --- Client 1: Main Content Generator (Writer) ---
    parser.add_argument("--llm_model_name", type=str, default="gpt-4.1")
    parser.add_argument("--api_key", type=str, default="EMPTY")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1")

    # --- Client 2: Boundary Detector (Linguist) ---
    parser.add_argument(
        "--boundary_model_name",
        type=str,
        default="gpt-4.1-mini",
        help="Model for detecting clause boundaries",
    )
    parser.add_argument(
        "--boundary_api_key", type=str, default=None, help="Defaults to env OPENAI_API_KEY if None"
    )
    parser.add_argument("--boundary_base_url", type=str, default=None)

    # --- Client 3: Turn-Taking Probability Estimator (Scorer), LLM-based ---
    parser.add_argument(
        "--tt_model_name",
        type=str,
        default="Qwen/Qwen3-14B",
        help="Model for estimating floor/backchannel probs (LLM-based)",
    )
    parser.add_argument("--tt_api_key", type=str, default="EMPTY")
    parser.add_argument(
        "--tt_base_url",
        type=str,
        default="http://localhost:8000/v1",
        help="Often a separate vLLM instance",
    )

    # --- HF Classification-based Turn-Taking Predictor (alternative to Client 3) ---
    parser.add_argument(
        "--hf_model_name_or_path",
        type=str,
        default=None,
        help="HuggingFace token-classification model for TT prediction. "
        "When set, disables LLM-based (Client 3) scoring.",
    )
    parser.add_argument(
        "--hf_peft_path",
        type=str,
        default=None,
        help="Path to PEFT/LoRA checkpoint for the HF TT model.",
    )
    parser.add_argument(
        "--hf_load_in_4bit",
        action="store_true",
        help="Load HF TT model in 4-bit quantisation (requires bitsandbytes).",
    )
    parser.add_argument(
        "--hf_max_seq_length",
        type=int,
        default=1024,
        help="Max token length for HF TT model inputs.",
    )
    parser.add_argument(
        "--hf_use_last_n_history",
        type=int,
        default=4,
        help="Number of previous turns to include as context for the HF TT model.",
    )
    parser.add_argument(
        "--hf_batch_size", type=int, default=8, help="Batch size for HF TT model inference."
    )

    # --- Pipeline Config ---
    parser.add_argument("--max_turns", type=int, default=20)
    parser.add_argument("--temperature_user", type=float, default=0.7)
    parser.add_argument("--temperature_ai", type=float, default=0.7)

    args = parser.parse_args()

    input_root = Path(args.input_root)
    save_root = Path(args.save_root)

    # 1. Initialize Clients
    # Main generation client
    client = make_client(args.llm_model_name, args.api_key, args.base_url)

    # Boundary detection client (usually GPT-4-mini)
    b_key = args.boundary_api_key or os.getenv("OPENAI_API_KEY")
    b_url = args.boundary_base_url  # If None, OpenAI default
    client_boundary = make_client(args.boundary_model_name, b_key, b_url)

    # Turn-taking scoring: HF classification model OR LLM-based client
    hf_model = None
    hf_tokenizer = None
    if args.hf_model_name_or_path:
        print(f"Loading HF TT classification model: {args.hf_model_name_or_path}")
        hf_model, hf_tokenizer = _load_hf_tt_model(
            model_name_or_path=args.hf_model_name_or_path,
            peft_path=args.hf_peft_path,
            load_in_4bit=args.hf_load_in_4bit,
        )
        client_tt = None  # not needed when using HF model
    else:
        client_tt = make_client(args.tt_model_name, args.tt_api_key, args.tt_base_url)

    # Resolve datasets
    if args.dataset == "all":
        datasets = DATASET_CHOICES
    else:
        datasets = [args.dataset]

    n_inputs_found = 0
    for dataset in datasets:
        in_paths = iter_input_jsons(input_root, dataset, args.split)
        if not in_paths:
            print(f"[WARN] No input JSONs for {dataset}/{args.split}, skipping.")
            continue
        n_inputs_found += len(in_paths)

        in_paths = in_paths[: args.max_dialogues]  # limit for quick testing

        out_dir = save_root / f"text_dialogue_{dataset}" / args.split
        out_dir.mkdir(parents=True, exist_ok=True)

        for in_path in tqdm(in_paths, desc=f"Applying TT predictor [{dataset}/{args.split}]"):
            out_path = out_dir / in_path.name
            if out_path.exists():
                continue

            ex = read_json(in_path)

            source_turns = ensure_source_turns(ex)
            scenario_desc = extract_scenario(ex)

            utterances = speechify_turn_by_turn(
                llm_model_name=args.llm_model_name,
                client=client,
                client_tt=client_tt,
                client_boundary=client_boundary,
                source_turns=source_turns,
                scenario_description=scenario_desc,
                max_turns=args.max_turns,
                temperature_user=args.temperature_user,
                temperature_ai=args.temperature_ai,
                dataset=dataset,
                tt_model_name=args.tt_model_name,
                boundary_model_name=args.boundary_model_name,
                hf_model=hf_model,
                hf_tokenizer=hf_tokenizer,
                hf_max_seq_length=args.hf_max_seq_length,
                hf_use_last_n_history=args.hf_use_last_n_history,
                hf_batch_size=args.hf_batch_size,
            )

            for u in utterances:
                u["content"] = _normalize_ws(u["content"])

            result = dict(ex)
            result["history"] = utterances
            result["meta"] = dict(ex.get("meta", {}))
            result["meta"]["turn_taking_applied"] = True
            result["meta"]["tt_model"] = args.hf_model_name_or_path or args.tt_model_name
            result["meta"]["tt_mode"] = "hf_classification" if hf_model else "llm_verbalized"
            result["meta"]["boundary_model"] = args.boundary_model_name
            result["meta"]["input_path"] = str(in_path)

            write_json(out_path, result)

    # Every dataset warned and was skipped: the root is wrong (the published HF
    # layout rather than text_dialogue_<dataset>/<split>/, say). Exiting 0 here
    # reads as "nothing left to do" when nothing was ever read.
    if n_inputs_found == 0:
        raise SystemExit(
            f"No input dialogues under {input_root} for split {args.split!r}. "
            f"--input_root must contain text_dialogue_<dataset>/{args.split}/*.json; "
            f"convert the released corpus with "
            f"`tools/unpack_corpus.py --kind dialogues` first."
        )


if __name__ == "__main__":
    main()
