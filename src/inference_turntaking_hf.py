import argparse
import glob
import json
import os
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from peft import PeftModel
from peft.utils.save_and_load import load_peft_weights
from transformers import (
    MODEL_MAPPING,
    AutoConfig,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from transformers.modeling_layers import GenericForTokenClassification
from tqdm import tqdm

from src.hf_attn import CHOICES as ATTN_CHOICES, resolve_attn_implementation
from src.turntaking_common import human_probs_to_train_dist
from src.models.gemma3_token_classification import Gemma3ForTokenClassification

LABELS = ["backchannel", "interruption", "none"]
TRAIN_LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
ID_TO_TRAIN_LABEL = {i: label for i, label in enumerate(LABELS)}


def train_dist_to_human_dict(dist: List[float]) -> Dict[str, float]:
    return {
        "backchannel": float(dist[TRAIN_LABEL_TO_ID["backchannel"]]),
        "take_floor": float(dist[TRAIN_LABEL_TO_ID["interruption"]]),
        "silent": float(dist[TRAIN_LABEL_TO_ID["none"]]),
    }


# The classifier head is trained on LABELS, but every field this script writes
# out -- and everything src/inference_turntaking_llm.py writes -- speaks the
# human-annotation spelling. Translate the argmax too, so `pred_label` names a
# key that actually exists in the `pred_probs` beside it.
TRAIN_LABEL_TO_HUMAN = {
    "backchannel": "backchannel",
    "interruption": "take_floor",
    "none": "silent",
}


def safe_kl(human: np.ndarray, pred: np.ndarray, eps: float = 1e-12) -> float:
    human = np.asarray(human, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    pred = np.clip(pred, eps, 1.0)
    mask = human > 0
    return float(np.sum(human[mask] * np.log(human[mask] / pred[mask])))


def list_human_scenarios(input_root: str) -> List[str]:
    dirs = sorted(glob.glob(os.path.join(input_root, "text_dialogue_*")))
    return sorted({os.path.basename(d)[len("text_dialogue_") :] for d in dirs if os.path.isdir(d)})


def resolve_scenarios(input_root: str, scenarios: str) -> List[str]:
    if scenarios == "all":
        out = list_human_scenarios(input_root)
        if not out:
            raise ValueError(f"No scenarios found under: {input_root}")
        return out
    parsed = [s.strip() for s in scenarios.split(",") if s.strip()]
    if not parsed:
        raise ValueError("At least one scenario is required.")
    return parsed


def build_human_boundary_examples(
    input_root: str,
    split: str,
    scenarios: str,
    use_last_n_history: int,
) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    scenario_list = resolve_scenarios(input_root, scenarios)

    for scenario in scenario_list:
        files = sorted(
            glob.glob(
                os.path.join(
                    input_root,
                    f"text_dialogue_{scenario}",
                    split,
                    "*.json",
                )
            )
        )
        for file_path in files:
            with open(file_path, "r", encoding="utf-8") as f:
                obj = json.load(f)

            history = obj.get("history", [])
            if not isinstance(history, list):
                continue

            for turn_idx, turn in enumerate(history):
                boundaries = turn.get("boundaries")
                content = turn.get("content", "")
                role = turn.get("role", "user")

                if not isinstance(boundaries, list) or not content or role != "user":
                    continue

                words = content.split()
                if not words:
                    continue

                context_turns = history[max(0, turn_idx - use_last_n_history) : turn_idx]
                context = []
                for ct in context_turns:
                    c_role = ct.get("role", "user")
                    c_content = ct.get("content", "")
                    if isinstance(c_content, str) and c_content.strip():
                        context.append({"role": c_role, "content": c_content})

                for b in boundaries:
                    if not isinstance(b, dict):
                        continue
                    word_idx = b.get("word_index")
                    if not isinstance(word_idx, int):
                        try:
                            word_idx = int(word_idx)
                        except Exception:
                            continue
                    if word_idx < 0 or word_idx >= len(words):
                        continue

                    probs = human_probs_to_train_dist(b.get("probabilities", {}))
                    prefix = " ".join(words[: word_idx + 1]).strip()
                    if not prefix:
                        continue

                    examples.append(
                        {
                            "scenario": scenario,
                            "split": split,
                            "example_id": obj.get("example_id", os.path.basename(file_path)),
                            "source_file": file_path,
                            "turn_index": int(turn_idx),
                            "word_index": int(word_idx),
                            "context": context,
                            "user_utterance": prefix,
                            "target_probs": probs,
                        }
                    )
    return examples


def tokenize_examples(
    examples: List[Dict[str, Any]], tokenizer, max_seq_length: int
) -> List[Dict[str, Any]]:
    tokenized = []
    tokenizer_kwargs = {}
    if "Qwen/Qwen3" in tokenizer.name_or_path:
        tokenizer_kwargs["enable_thinking"] = False

    for ex in examples:
        messages = ex["context"] + [{"role": "user", "content": ex["user_utterance"]}]
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
        )
        input_ids = out["input_ids"]
        if not input_ids:
            continue
        tokenized.append(
            {
                "meta": ex,
                "input_ids": input_ids,
                "attention_mask": out["attention_mask"],
                "boundary_index": len(input_ids) - 1,
            }
        )
    return tokenized


def collate_batch(batch: List[Dict[str, Any]], pad_token_id: int) -> Dict[str, Any]:
    input_ids = [torch.tensor(item["input_ids"], dtype=torch.long) for item in batch]
    attention_mask = [torch.tensor(item["attention_mask"], dtype=torch.long) for item in batch]
    boundary_index = torch.tensor([item["boundary_index"] for item in batch], dtype=torch.long)

    input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    attention_mask = nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "boundary_index": boundary_index,
        "meta": [item["meta"] for item in batch],
    }


def get_token_classification_model(
    model_name_or_path: str,
    load_in_4bit: bool,
    peft_path: str = None,
    attn_implementation: str = "auto",
):
    # If model_name_or_path is a PEFT adapter directory (no model_type in config),
    # resolve the actual base model from adapter_config.json.
    adapter_cfg_path = os.path.join(model_name_or_path, "adapter_config.json")
    base_model_path = model_name_or_path
    if os.path.isfile(adapter_cfg_path):
        with open(adapter_cfg_path, "r", encoding="utf-8") as _f:
            _adapter_cfg = json.load(_f)
        base_model_path = _adapter_cfg.get("base_model_name_or_path", model_name_or_path)
        print(f"Detected PEFT adapter dir; using base model: {base_model_path}")
        # If peft_path not explicitly set, treat model_name_or_path as the peft_path
        if peft_path is None:
            peft_path = model_name_or_path

    config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    if hasattr(config, "text_config"):
        config = config.text_config
    config.num_labels = len(LABELS)
    config.label2id = TRAIN_LABEL_TO_ID
    config.id2label = ID_TO_TRAIN_LABEL

    attn_impl = resolve_attn_implementation(attn_implementation)

    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=(
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            ),
            llm_int8_skip_modules=["score"],
        )

    if "gemma-3" in base_model_path.lower():
        model = Gemma3ForTokenClassification.from_pretrained(
            base_model_path,
            config=config,
            quantization_config=bnb_config,
            device_map="auto" if load_in_4bit else None,
            torch_dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16),
            attn_implementation=attn_impl,
            trust_remote_code=True,
        )
    else:
        pretrained_model_class = MODEL_MAPPING[type(config)].__base__
        class_name = "".join([w.capitalize() for w in config.model_type.split("_")])
        token_classification_model = type(
            f"{class_name}ForTokenClassification",
            (GenericForTokenClassification, pretrained_model_class),
            {},
        )
        model = token_classification_model.from_pretrained(
            base_model_path,
            config=config,
            quantization_config=bnb_config,
            device_map="auto" if load_in_4bit else None,
            torch_dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16),
            attn_implementation=attn_impl,
            trust_remote_code=True,
        )

    if peft_path:
        peft_weights = load_peft_weights(peft_path, device="cpu")
        score_keys = [k for k in peft_weights.keys() if "score" in k]
        if not score_keys:
            raise ValueError(
                f"No score.* weights found in PEFT checkpoint: {peft_path}. "
                "This means the token-classification head was not saved."
            )
        model = PeftModel.from_pretrained(model, peft_path)
        print(
            f"Loaded PEFT checkpoint from {peft_path} with {len(score_keys)} score-related tensors."
        )

    if torch.cuda.is_available() and not load_in_4bit:
        model = model.cuda()
    model.eval()
    return model


def run_inference(
    model,
    tokenized_examples: List[Dict[str, Any]],
    batch_size: int,
    pad_token_id: int,
) -> List[Dict[str, Any]]:
    outputs: List[Dict[str, Any]] = []
    n = len(tokenized_examples)

    with torch.no_grad():
        for i in tqdm(range(0, n, batch_size), desc="Inferencing", unit="batch"):
            batch = tokenized_examples[i : i + batch_size]
            collated = collate_batch(batch, pad_token_id=pad_token_id)

            input_ids = collated["input_ids"]
            attention_mask = collated["attention_mask"]
            boundary_index = collated["boundary_index"]
            metas = collated["meta"]

            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                boundary_index = boundary_index.cuda()

            logits = model(input_ids=input_ids, attention_mask=attention_mask)["logits"]
            selected_logits = logits[
                torch.arange(logits.size(0), device=logits.device), boundary_index
            ]
            probs = torch.softmax(selected_logits.float(), dim=-1).detach().cpu().numpy()

            for meta, pred in zip(metas, probs):
                human = np.array(meta["target_probs"], dtype=np.float64)
                pred = np.array(pred, dtype=np.float64)
                pred = pred / np.sum(pred)
                kl = safe_kl(human, pred)

                outputs.append(
                    {
                        "scenario": meta["scenario"],
                        "split": meta["split"],
                        "example_id": meta["example_id"],
                        "source_file": meta["source_file"],
                        "turn_index": meta["turn_index"],
                        "word_index": meta["word_index"],
                        "user_utterance": meta["user_utterance"],
                        "human_probs": train_dist_to_human_dict(human.tolist()),
                        "pred_probs": train_dist_to_human_dict(pred.tolist()),
                        "pred_label": TRAIN_LABEL_TO_HUMAN[ID_TO_TRAIN_LABEL[int(np.argmax(pred))]],
                        "kl_human_pred": float(kl),
                    }
                )
    return outputs


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Inference for turn-taking token classifier on human-annotated "
            "boundary dataset (default split=train)."
        )
    )
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--peft_path", type=str, default=None)
    parser.add_argument(
        "--input_root",
        type=str,
        required=True,
        help="Root holding text_dialogue_<scenario>/<split>/*.json slot annotations "
        "(e.g. a local checkout of the DuplexGen/duplexgen-corpus annotations split).",
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--scenarios", type=str, default="all")
    parser.add_argument("--use_last_n_history", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        choices=ATTN_CHOICES,
        default="auto",
        help="Attention backend. 'auto' uses FlashAttention-2 when the optional "
        "flash-attn package is installed and PyTorch SDPA otherwise. "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/result_swbd/train_inference_turntaking_hf.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Building boundary examples...")
    examples = build_human_boundary_examples(
        input_root=args.input_root,
        split=args.split,
        scenarios=args.scenarios,
        use_last_n_history=args.use_last_n_history,
    )
    if not examples:
        raise ValueError("No examples found. Check --input_root / --split / --scenarios.")
    print(f"Loaded examples: {len(examples)}")

    print("Tokenizing...")
    tokenized_examples = tokenize_examples(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
    )
    print(f"Tokenized examples: {len(tokenized_examples)}")

    print("Loading model...")
    model = get_token_classification_model(
        model_name_or_path=args.model_name_or_path,
        load_in_4bit=args.load_in_4bit,
        peft_path=args.peft_path,
        attn_implementation=args.attn_implementation,
    )

    print("Running inference...")
    per_sample_results = run_inference(
        model=model,
        tokenized_examples=tokenized_examples,
        batch_size=args.batch_size,
        pad_token_id=tokenizer.pad_token_id,
    )

    mean_kl = float(np.mean([x["kl_human_pred"] for x in per_sample_results]))
    out = {
        "split": args.split,
        "scenarios": args.scenarios,
        "n_samples": len(per_sample_results),
        "mean_kl_human_pred": mean_kl,
        "per_sample_results": per_sample_results,
    }

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Wrote: {args.output_path}")
    print(f"n_samples={out['n_samples']} mean_kl_human_pred={out['mean_kl_human_pred']:.6f}")


if __name__ == "__main__":
    main()
