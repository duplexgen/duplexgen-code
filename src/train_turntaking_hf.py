import argparse
import glob
import io
import json
import os
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForTokenClassification,
    Gemma3ForCausalLM,
    GenerationConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from transformers.modeling_layers import GenericForTokenClassification

import torch.nn.functional as F
from datasets import Dataset, DatasetDict, load_dataset

from src.hf_attn import CHOICES as ATTN_CHOICES, resolve_attn_implementation
from src.turntaking_common import human_probs_to_train_dist
from src.models.gemma3_token_classification import Gemma3ForTokenClassification

LABELS = [
    "backchannel",
    "interruption",
    "none",
]
HUMAN_TO_TRAIN_LABEL = {
    "backchannel": "backchannel",
    "take_floor": "interruption",
    "silent": "none",
}


class ConfusionMatrixLoggingCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        # Initialize writer with the training args logging_dir
        self.writer = SummaryWriter(args.logging_dir)

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if not state.is_world_process_zero:
            return

        # labels_names should match what's in compute_metrics
        # Determine labels names based on the metrics keys present
        if "eval_cm_3_3" in metrics:
            labels_names = ["BC", "INT", "NONE", "OTHR"]
            num_labels = 4
        else:
            labels_names = ["BC", "INT", "NONE"]
            num_labels = 3

        # Reconstruct CM from metrics
        cm = np.zeros((num_labels, num_labels))
        for i in range(num_labels):
            for j in range(num_labels):
                key = f"eval_cm_{i}_{j}"
                if key in metrics:
                    cm[i, j] = metrics[key]

        # Row-normalize for coloring (Ratio per ground-truth class)
        cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-9)

        # Create plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm_norm,  # Use normalized values for coloring
            annot=cm,  # Use raw counts for text
            fmt="g",  # Display as general numbers
            cmap="Blues",
            xticklabels=labels_names,
            yticklabels=labels_names,
            vmin=0,
            vmax=1,
        )
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title(f"Confusion Matrix (Step {state.global_step})")

        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)

        # Convert to numpy array for TensorBoard
        # Convert to RGB to ensure compatibility and consistent colors
        image = Image.open(buf).convert("RGB")
        image_np = np.array(image)

        # Log to TensorBoard
        self.writer.add_image(
            "confusion_matrix",
            image_np,
            global_step=state.global_step,
            dataformats="HWC",
        )
        self.writer.flush()
        print(f"Logged confusion matrix heatmap to TensorBoard at step {state.global_step}")


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def preprocess_chat_train(examples, tokenizer, max_seq_length):
    batch_messages = examples["messages"]
    tokenized_batch = {"input_ids": [], "attention_mask": [], "labels": []}

    for messages in batch_messages:
        # Separate the last assistant response (label) from context
        context_messages = messages[:-1]

        tokenizer_kwargs = {}
        if "Qwen/Qwen3" in tokenizer.name_or_path:
            tokenizer_kwargs["enable_thinking"] = False

        # Format the context and the whole conversation
        full_text = tokenizer.apply_chat_template(messages, tokenize=False, **tokenizer_kwargs)
        context_text = tokenizer.apply_chat_template(
            context_messages,
            tokenize=False,
            add_generation_prompt=True,
            **tokenizer_kwargs,
        )

        # Tokenize both
        full_tokenized = tokenizer(
            text=full_text,
            truncation=True,
            max_length=max_seq_length,
            add_special_tokens=False,
        )
        context_tokenized = tokenizer(
            text=context_text,
            truncation=True,
            max_length=max_seq_length,
            add_special_tokens=False,
        )

        input_ids = full_tokenized["input_ids"]
        context_len = len(context_tokenized["input_ids"])

        # Create labels: mask context area with -100
        labels = [-100] * (context_len - 1) + input_ids[context_len:]
        input_ids = input_ids[:-1]

        tokenized_batch["input_ids"].append(input_ids)
        tokenized_batch["attention_mask"].append(full_tokenized["attention_mask"])
        tokenized_batch["labels"].append(labels)

    return tokenized_batch


def preprocess_chat_eval(examples, tokenizer, max_seq_length):
    batch_messages = examples["messages"]
    tokenized_batch = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }

    for messages in batch_messages:
        # Separate the last assistant response (label) from context
        context_messages = messages[:-1]
        label_message = messages[-1]

        tokenizer_kwargs = {}
        if "Qwen/Qwen3" in tokenizer.name_or_path:
            tokenizer_kwargs["enable_thinking"] = False

        # Format the context and the whole conversation
        input_ids = tokenizer.apply_chat_template(
            context_messages,
            add_generation_prompt=True,
            truncation=True,
            max_length=max_seq_length,
            **tokenizer_kwargs,
        )

        labels = tokenizer.encode(label_message["content"], add_special_tokens=False)

        tokenized_batch["input_ids"].append(input_ids)
        tokenized_batch["attention_mask"].append([1] * len(input_ids))
        tokenized_batch["labels"].append(labels)

    return tokenized_batch


def preprocess_classification(examples, tokenizer, max_seq_length, label_map):
    batch_contexts = examples["context"]
    batch_utterances = examples["user_utterance"]
    batch_trps = examples["trps"]

    tokenized_batch = {"input_ids": [], "attention_mask": [], "labels": []}

    for context, utterance, trps in zip(batch_contexts, batch_utterances, batch_trps):
        messages = context + [{"role": "user", "content": utterance}]

        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        tokenized = tokenizer(
            text=full_text,
            truncation=True,
            max_length=max_seq_length,
            return_offsets_mapping=True,
        )

        input_ids = tokenized["input_ids"]
        offsets = tokenized["offset_mapping"]
        labels = [-100] * len(input_ids)

        words = utterance.split()
        word_end_offsets = []
        curr_pos = 0
        for w in words:
            start = utterance.find(w, curr_pos)
            end = start + len(w)
            word_end_offsets.append(end)
            curr_pos = end

        utterance_start_char = full_text.find(utterance)
        if utterance_start_char != -1:
            for trp in trps:
                w_idx = trp["word_idx"]
                target_char_pos = utterance_start_char + word_end_offsets[w_idx]

                for t_idx, (start, end) in enumerate(offsets):
                    if start < target_char_pos <= end:
                        # Standard CausalLM: label at i+1 is what is predicted by hidden state at i.
                        # So we set labels[t_idx + 1] if within bounds.
                        if t_idx < len(labels):
                            labels[t_idx] = label_map[trp["label"]]
                        break

        tokenized_batch["input_ids"].append(input_ids)
        tokenized_batch["attention_mask"].append(tokenized["attention_mask"])
        tokenized_batch["labels"].append(labels)

    return tokenized_batch


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


SWB_TYPE_TO_LABEL = {
    "BACKCHANNEL": "backchannel",
    "TAKE_FLOOR": "take_floor",
    None: "silent",
}


def build_swb1_boundary_examples(
    input_root: str,
    split: str,
    use_last_n_history: int,
    train_ratio: float = 0.9,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Build boundary examples from swb1_dialogact_annot data.

    Files are split into train/test by filename hash for reproducibility.
    Each turn with boundaries becomes a single example containing all slots.
    """
    import hashlib

    files = sorted(glob.glob(os.path.join(input_root, "*.json")))
    if not files:
        raise ValueError(f"No JSON files found under: {input_root}")

    # Deterministic train/test split by file hash
    def file_is_train(path: str) -> bool:
        h = int(hashlib.md5(os.path.basename(path).encode()).hexdigest(), 16)
        return (h % 10000) / 10000 < train_ratio

    if split == "train":
        files = [f for f in files if file_is_train(f)]
    elif split == "test":
        files = [f for f in files if not file_is_train(f)]
    # split == "all" keeps everything

    examples: List[Dict[str, Any]] = []

    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        history = obj.get("history", [])
        if not isinstance(history, list):
            continue

        for turn_idx, turn in enumerate(history):
            boundaries = turn.get("boundary_word_indices", [])
            content = turn.get("content", "")

            if not isinstance(boundaries, list) or not content:
                continue

            words = content.split()
            if not words:
                continue

            # Build context from preceding turns
            context_turns = history[max(0, turn_idx - use_last_n_history) : turn_idx]
            context = []
            for ct in context_turns:
                c_role = ct.get("role", "user")
                c_content = ct.get("content", "")
                if isinstance(c_content, str) and c_content.strip():
                    context.append({"role": c_role, "content": c_content})

            # Collect all valid boundaries for this turn
            word_indices = []
            all_probs = []
            for b in boundaries:
                if not isinstance(b, dict):
                    continue
                word_idx = b.get("idx")
                if not isinstance(word_idx, int):
                    try:
                        word_idx = int(word_idx)
                    except Exception:
                        continue
                if word_idx < 0 or word_idx >= len(words):
                    continue

                raw_type = b.get("type")  # None, "BACKCHANNEL", or "TAKE_FLOOR"
                label_str = SWB_TYPE_TO_LABEL.get(raw_type, "silent")
                train_label = HUMAN_TO_TRAIN_LABEL[label_str]
                hard_label = LABELS.index(train_label)

                probs = [0.0] * len(LABELS)
                probs[hard_label] = 1.0

                word_indices.append(word_idx)
                all_probs.append(probs)

            if not word_indices:
                continue

            examples.append(
                {
                    "scenario": "swb1",
                    "split": split,
                    "example_id": obj.get("example_id", os.path.basename(file_path)),
                    "turn_index": turn_idx,
                    "context": context,
                    "user_utterance": content,
                    "trps": [
                        {"word_idx": wi, "target_probs": p}
                        for wi, p in zip(word_indices, all_probs)
                    ],
                }
            )

    return examples


def build_human_boundary_examples(
    input_root: str,
    split: str,
    scenarios: str,
    use_last_n_history: int,
    subsample_ratio: float = 1.0,
    subsample_seed: int = 42,
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
        # Deterministic dialogue-level subsampling of the training set.
        # Applied per scenario so each scenario contributes proportionally.
        if subsample_ratio < 1.0 and files:
            import random as _random

            k = max(1, int(round(len(files) * subsample_ratio)))
            rng = _random.Random(subsample_seed)
            files = sorted(rng.sample(files, k))
            print(
                f"[subsample] scenario={scenario} split={split} "
                f"ratio={subsample_ratio} kept {k}/{len(glob.glob(os.path.join(input_root, f'text_dialogue_{scenario}', split, '*.json')))} dialogues"
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

                if not isinstance(boundaries, list) or not content:
                    continue
                if role != "user":
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

                # Collect all valid boundaries for this turn
                word_indices = []
                all_probs = []
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
                    word_indices.append(word_idx)
                    all_probs.append(probs)

                if not word_indices:
                    continue

                examples.append(
                    {
                        "scenario": scenario,
                        "split": split,
                        "example_id": obj.get("example_id", os.path.basename(file_path)),
                        "turn_index": turn_idx,
                        "context": context,
                        "user_utterance": content,
                        "trps": [
                            {"word_idx": wi, "target_probs": p}
                            for wi, p in zip(word_indices, all_probs)
                        ],
                    }
                )
    return examples


def flatten_to_single_boundary(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Expand multi-slot examples into one example per boundary (prefix-based).

    Each output example has a single ``target_probs`` list and
    ``user_utterance`` truncated to the boundary word (prefix).
    """
    flat: List[Dict[str, Any]] = []
    for ex in examples:
        words = ex["user_utterance"].split()
        for trp in ex["trps"]:
            w_idx = trp["word_idx"]
            prefix = " ".join(words[: w_idx + 1]).strip()
            if not prefix:
                continue
            flat.append(
                {
                    "context": ex["context"],
                    "user_utterance": prefix,
                    "target_probs": trp["target_probs"],
                }
            )
    return flat


def preprocess_soft_classification_single(examples, tokenizer, max_seq_length):
    """Tokenize single-boundary examples (one boundary per example, prefix-based)."""
    batch_contexts = examples["context"]
    batch_utterances = examples["user_utterance"]
    batch_probs = examples["target_probs"]

    tokenized_batch = {
        "input_ids": [],
        "attention_mask": [],
        "boundary_indices": [],
        "target_probs": [],
        "num_boundaries": [],
    }

    tokenizer_kwargs = {}
    if "Qwen/Qwen3" in tokenizer.name_or_path:
        tokenizer_kwargs["enable_thinking"] = False

    for context, utterance, probs in zip(batch_contexts, batch_utterances, batch_probs):
        messages = context + [{"role": "user", "content": utterance}]
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            **tokenizer_kwargs,
        )
        tokenized = tokenizer(
            text=full_text,
            truncation=True,
            max_length=max_seq_length,
        )
        input_ids = tokenized["input_ids"]
        if not input_ids:
            continue

        tokenized_batch["input_ids"].append(input_ids)
        tokenized_batch["attention_mask"].append(tokenized["attention_mask"])
        tokenized_batch["boundary_indices"].append([len(input_ids) - 1])
        tokenized_batch["target_probs"].append([probs])
        tokenized_batch["num_boundaries"].append(1)

    return tokenized_batch


def preprocess_soft_classification(examples, tokenizer, max_seq_length):
    """Tokenize full utterance once; locate all boundary word positions via offset mapping.

    Each example now has multiple boundary slots (trps).  We produce one
    tokenized sequence per turn and attach parallel lists of boundary token
    indices and their soft target distributions.
    """
    batch_contexts = examples["context"]
    batch_utterances = examples["user_utterance"]
    batch_trps = examples["trps"]

    tokenized_batch = {
        "input_ids": [],
        "attention_mask": [],
        "boundary_indices": [],  # List[List[int]], variable-length per example
        "target_probs": [],  # List[List[List[float]]]
        "num_boundaries": [],  # List[int]
    }

    tokenizer_kwargs = {}
    if "Qwen/Qwen3" in tokenizer.name_or_path:
        tokenizer_kwargs["enable_thinking"] = False

    for context, utterance, trps in zip(batch_contexts, batch_utterances, batch_trps):
        messages = context + [{"role": "user", "content": utterance}]
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            **tokenizer_kwargs,
        )
        tokenized = tokenizer(
            text=full_text,
            truncation=True,
            max_length=max_seq_length,
            return_offsets_mapping=True,
        )
        input_ids = tokenized["input_ids"]
        offsets = tokenized["offset_mapping"]
        if not input_ids:
            continue

        # Map each boundary word_idx → token position
        words = utterance.split()
        word_end_offsets = []
        curr_pos = 0
        for w in words:
            start = utterance.find(w, curr_pos)
            end = start + len(w)
            word_end_offsets.append(end)
            curr_pos = end

        utterance_start_char = full_text.find(utterance)
        if utterance_start_char == -1:
            continue

        b_indices = []
        b_probs = []
        for trp in trps:
            w_idx = trp["word_idx"]
            if w_idx >= len(word_end_offsets):
                continue
            target_char_pos = utterance_start_char + word_end_offsets[w_idx]

            # Find the token whose span covers this character position
            tok_idx = None
            for t_idx, (s, e) in enumerate(offsets):
                if s < target_char_pos <= e:
                    tok_idx = t_idx
                    break
            if tok_idx is not None:
                b_indices.append(tok_idx)
                b_probs.append(trp["target_probs"])

        if not b_indices:
            continue

        tokenized_batch["input_ids"].append(input_ids)
        tokenized_batch["attention_mask"].append(tokenized["attention_mask"])
        tokenized_batch["boundary_indices"].append(b_indices)
        tokenized_batch["target_probs"].append(b_probs)
        tokenized_batch["num_boundaries"].append(len(b_indices))

    return tokenized_batch


class SoftLabelTokenClassificationTrainer(Trainer):
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        boundary_indices = inputs.pop("boundary_indices")  # (B, max_slots) padded with -1
        target_probs = inputs.pop("target_probs")  # (B, max_slots, C) padded with 0
        num_boundaries = inputs.pop("num_boundaries")  # (B,)

        outputs = model(**inputs)
        logits = outputs.logits  # (B, T, C)

        # Gather logits at all valid boundary positions
        all_selected = []
        all_targets = []
        for i in range(logits.size(0)):
            n = num_boundaries[i].item()
            if n == 0:
                continue
            idx = boundary_indices[i, :n]  # (n,)
            all_selected.append(logits[i, idx])  # (n, C)
            all_targets.append(target_probs[i, :n])  # (n, C)

        if not all_selected:
            loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
            return (loss, outputs) if return_outputs else loss

        selected_logits = torch.cat(all_selected, dim=0)  # (total_slots, C)
        target = torch.cat(all_targets, dim=0)  # (total_slots, C)

        loss = F.kl_div(
            F.log_softmax(selected_logits, dim=-1),
            target,
            reduction="batchmean",
        )
        return (loss, outputs) if return_outputs else loss


def get_token_classification_model(
    model_name_or_path, quantization_config, local_rank, attn_implementation="auto"
):
    attn_impl = resolve_attn_implementation(attn_implementation)
    config = AutoConfig.from_pretrained(model_name_or_path)
    if hasattr(config, "text_config"):
        config = config.text_config

    config.num_labels = len(LABELS)
    config.label2id = {label: i for i, label in enumerate(LABELS)}
    config.id2label = {i: label for i, label in enumerate(LABELS)}

    pretrained_model_class = MODEL_MAPPING[type(config)].__base__
    class_name = "".join([w.capitalize() for w in config.model_type.split("_")])
    TokenClassificationModel = type(
        f"{class_name}ForTokenClassification",
        (GenericForTokenClassification, pretrained_model_class),
        {},
    )
    if "gemma-3" in model_name_or_path:
        model = Gemma3ForTokenClassification.from_pretrained(
            model_name_or_path,
            config=config,
            quantization_config=quantization_config,
            device_map={"": local_rank},
            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            attn_implementation=attn_impl,
        )
    else:
        model = TokenClassificationModel.from_pretrained(
            model_name_or_path,
            config=config,
            quantization_config=quantization_config,
            device_map={"": local_rank},
            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            attn_implementation=attn_impl,
        )
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--validation_file", type=str, default=None)
    parser.add_argument(
        "--input_root",
        type=str,
        required=True,
        help="Root holding text_dialogue_<scenario>/<split>/*.json slot annotations "
        "(e.g. a local checkout of the DuplexGen/duplexgen-corpus annotations split).",
    )
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--validation_split", type=str, default="test")
    parser.add_argument("--train_scenarios", type=str, default="all")
    parser.add_argument("--validation_scenarios", type=str, default="all")
    parser.add_argument("--use_last_n_history", type=int, default=4)
    parser.add_argument(
        "--dataset_format",
        type=str,
        default="human_boundary",
        choices=["legacy", "human_boundary", "swb1_dialogact"],
    )
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument(
        "--style",
        type=str,
        default="soft_classification",
        choices=["chat", "classification", "soft_classification"],
    )
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--attn_implementation",
        type=str,
        choices=ATTN_CHOICES,
        default="auto",
        help="Attention backend. 'auto' uses FlashAttention-2 when the optional "
        "flash-attn package is installed and PyTorch SDPA otherwise. "
        "(default: %(default)s)",
    )
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--num_saves", type=int, default=3)
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,  # `--resume` alone => True (latest in output_dir)
        default=False,
        help="Resume from latest checkpoint in output_dir, or pass a checkpoint path.",
    )
    parser.add_argument(
        "--init_from_checkpoint",
        type=str,
        default=None,
        help="Initialize adapter weights from this PEFT checkpoint without restoring training state.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Train/test split ratio for swb1_dialogact format (by file hash).",
    )
    parser.add_argument(
        "--single_boundary",
        action="store_true",
        help="Expand each boundary into a separate prefix-based example (old strategy). "
        "Default is multi-slot (all boundaries per utterance in one example).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train_subsample_ratio",
        type=float,
        default=1.0,
        help="Fraction of training (finetune) dialogues to use, per scenario. "
        "Deterministic dialogue-level subsample. 1.0 = full training set.",
    )
    parser.add_argument(
        "--subsample_seed",
        type=int,
        default=42,
        help="Seed for deterministic training-set subsampling.",
    )
    args = parser.parse_args()
    return args


def build_model(args):
    set_seed(args.seed)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # 1. Load Model and Tokenizer via Standard Transformers
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.load_in_4bit,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=(
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        ),
        llm_int8_skip_modules=["score"],
    )
    quantization_config = bnb_config if args.load_in_4bit else None
    attn_impl = resolve_attn_implementation(args.attn_implementation)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.style == "chat":
        if "gemma-3" in args.model_name_or_path:
            model = Gemma3ForCausalLM.from_pretrained(
                args.model_name_or_path,
                quantization_config=quantization_config,
                device_map={"": local_rank},
                dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16),
                attn_implementation=attn_impl,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                quantization_config=quantization_config,
                device_map={"": local_rank},
                dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16),
                attn_implementation=attn_impl,
            )
    elif args.style in {"classification", "soft_classification"}:
        model = get_token_classification_model(
            args.model_name_or_path,
            quantization_config,
            local_rank,
            attn_implementation=args.attn_implementation,
        )

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # after model is created
    if hasattr(model, "score"):
        model.score = model.score.to(
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        # make sure it’s trainable
        for p in model.score.parameters():
            p.requires_grad = True

    # 2. Add LoRA
    lora_config = LoraConfig(
        r=args.lora_rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM" if args.style == "chat" else "TOKEN_CLS",
    )
    model = get_peft_model(model, lora_config)

    # Load adapter weights from a prior checkpoint without restoring training state.
    # Use this to fine-tune from a pretrained PEFT checkpoint (e.g. SWBD) on a new task.
    if args.init_from_checkpoint:
        model.load_adapter(args.init_from_checkpoint, adapter_name="default")
        print(f"Initialized adapter weights from: {args.init_from_checkpoint}")

    # Gradient checkpointing and input requiring grads
    # model.enable_input_require_grads()
    model.config.use_cache = False  # Required for gradient checkpointing

    return model, tokenizer


def build_datasets(args, tokenizer):
    # 3. Load Dataset
    if args.dataset_format == "swb1_dialogact":
        train_examples = build_swb1_boundary_examples(
            input_root=args.input_root,
            split="train",
            use_last_n_history=args.use_last_n_history,
            train_ratio=args.train_ratio,
            seed=args.seed,
        )
        validation_examples = build_swb1_boundary_examples(
            input_root=args.input_root,
            split="test",
            use_last_n_history=args.use_last_n_history,
            train_ratio=args.train_ratio,
            seed=args.seed,
        )
        if args.single_boundary:
            train_examples = flatten_to_single_boundary(train_examples)
            validation_examples = flatten_to_single_boundary(validation_examples)
        dataset = DatasetDict(
            {
                "train": Dataset.from_list(train_examples),
                "validation": Dataset.from_list(validation_examples),
            }
        )
        print(
            f"Loaded SWB1 dialogact dataset: train={len(train_examples)}, "
            f"validation={len(validation_examples)}"
        )
    elif args.dataset_format == "human_boundary":
        train_examples = build_human_boundary_examples(
            input_root=args.input_root,
            split=args.train_split,
            scenarios=args.train_scenarios,
            use_last_n_history=args.use_last_n_history,
            subsample_ratio=args.train_subsample_ratio,
            subsample_seed=args.subsample_seed,
        )
        validation_examples = build_human_boundary_examples(
            input_root=args.input_root,
            split=args.validation_split,
            scenarios=args.validation_scenarios,
            use_last_n_history=args.use_last_n_history,
        )
        if args.single_boundary:
            train_examples = flatten_to_single_boundary(train_examples)
            validation_examples = flatten_to_single_boundary(validation_examples)
        dataset = DatasetDict(
            {
                "train": Dataset.from_list(train_examples),
                "validation": Dataset.from_list(validation_examples),
            }
        )
        print(
            f"Loaded human boundary dataset: train={len(train_examples)}, "
            f"validation={len(validation_examples)}"
        )
    else:
        if not args.train_file or not args.validation_file:
            raise ValueError(
                "--train_file and --validation_file are required with --dataset_format legacy"
            )
        dataset = load_dataset(
            "json",
            data_files={"train": args.train_file, "validation": args.validation_file},
        )

    label_map = {"backchannel": 0, "interruption": 1, "none": 2}

    if args.style == "chat":
        tokenized_datasets = DatasetDict(
            {
                "train": dataset["train"].map(
                    preprocess_chat_train,
                    fn_kwargs={
                        "tokenizer": tokenizer,
                        "max_seq_length": args.max_seq_length,
                    },
                    batched=True,
                    remove_columns=dataset["train"].column_names,
                    desc="Tokenizing chat dataset",
                ),
                "validation": dataset["validation"].map(
                    preprocess_chat_eval,
                    fn_kwargs={
                        "tokenizer": tokenizer,
                        "max_seq_length": args.max_seq_length,
                    },
                    batched=True,
                    remove_columns=[
                        col
                        for col in dataset["validation"].column_names
                        if col not in ["example_id", "turn_range"]
                    ],
                    desc="Tokenizing chat dataset",
                ),
            }
        )
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    elif args.style == "classification":
        tokenized_datasets = dataset.map(
            preprocess_classification,
            fn_kwargs={
                "tokenizer": tokenizer,
                "max_seq_length": args.max_seq_length,
                "label_map": label_map,
            },
            batched=True,
            remove_columns=dataset["train"].column_names,
            desc="Tokenizing classification dataset",
        )

        def collate_fn(batch):
            input_ids = [torch.tensor(item["input_ids"]) for item in batch]
            attention_mask = [torch.tensor(item["attention_mask"]) for item in batch]
            labels = [torch.tensor(item["labels"]) for item in batch]

            input_ids = nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
            )
            attention_mask = nn.utils.rnn.pad_sequence(
                attention_mask, batch_first=True, padding_value=0
            )
            labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

        data_collator = collate_fn
    else:
        preprocess_fn = (
            preprocess_soft_classification_single
            if args.single_boundary
            else preprocess_soft_classification
        )
        tokenized_datasets = dataset.map(
            preprocess_fn,
            fn_kwargs={
                "tokenizer": tokenizer,
                "max_seq_length": args.max_seq_length,
            },
            batched=True,
            remove_columns=dataset["train"].column_names,
            desc="Tokenizing soft classification dataset",
        )

        def soft_collate_fn(batch):
            input_ids = [torch.tensor(item["input_ids"]) for item in batch]
            attention_mask = [torch.tensor(item["attention_mask"]) for item in batch]

            input_ids = nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
            )
            attention_mask = nn.utils.rnn.pad_sequence(
                attention_mask, batch_first=True, padding_value=0
            )

            # Pad boundary_indices and target_probs to max number of slots in batch
            num_boundaries = torch.tensor(
                [item["num_boundaries"] for item in batch], dtype=torch.long
            )
            max_slots = max(item["num_boundaries"] for item in batch)
            num_labels = len(LABELS)

            boundary_indices = torch.full((len(batch), max_slots), -1, dtype=torch.long)
            target_probs = torch.zeros((len(batch), max_slots, num_labels), dtype=torch.float32)
            for i, item in enumerate(batch):
                n = item["num_boundaries"]
                boundary_indices[i, :n] = torch.tensor(item["boundary_indices"], dtype=torch.long)
                target_probs[i, :n] = torch.tensor(item["target_probs"], dtype=torch.float32)

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "boundary_indices": boundary_indices,
                "target_probs": target_probs,
                "num_boundaries": num_boundaries,
            }

        data_collator = soft_collate_fn

    return tokenized_datasets, data_collator


def build_trainer(args, model, tokenizer, tokenized_datasets, data_collator):
    label_map = {"backchannel": 0, "interruption": 1, "none": 2}

    def compute_metrics(eval_preds):
        predictions, labels = eval_preds

        # 1. Processing for Chat Mode (Generation)
        if args.style == "chat":
            # If Seq2SeqTrainer.predict_with_generate=True, predictions are token IDs
            if predictions.ndim == 3:
                predictions = np.argmax(predictions, axis=-1)

            # Both predictions and labels may contain -100 as padding from gathering
            labels[labels == -100] = tokenizer.pad_token_id
            predictions[predictions == -100] = tokenizer.pad_token_id

            # Decode to strings for response-level accuracy
            decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            mapped_preds = []
            mapped_labels = []

            for p, l in zip(decoded_preds, decoded_labels):
                p_clean = p.strip().lower()
                l_clean = l.strip().lower()

                mapped_l = label_map.get(l_clean, 3)

                mapped_p = 3  # "other"
                for label_str, val in label_map.items():
                    if label_str in p_clean:
                        mapped_p = val
                        break

                mapped_preds.append(mapped_p)
                mapped_labels.append(mapped_l)

            predictions = np.array(mapped_preds)
            labels = np.array(mapped_labels)
            num_metrics_labels = 4
            labels_names = ["BC  ", "INT ", "NONE", "OTHR"]

            # Log predictions with metadata
            eval_dataset = tokenized_datasets["validation"]
            # trainer is accessible from outer scope
            step = (
                trainer.state.global_step
                if (trainer is not None and hasattr(trainer, "state"))
                else 0
            )
            log_filename = f"eval_predictions_step_{step}.jsonl"
            log_path = os.path.join(args.output_dir, log_filename)

            # Make sure we only log if the lengths match (they should in evaluation)
            if len(decoded_preds) == len(eval_dataset):
                with open(log_path, "w") as f:
                    for i in range(len(decoded_preds)):
                        entry = {
                            "example_id": eval_dataset[i]["example_id"],
                            "turn_range": eval_dataset[i]["turn_range"],
                            "prediction": decoded_preds[i],
                            "label": decoded_labels[i],
                            "mapped_prediction": labels_names[mapped_preds[i]].strip(),
                            "mapped_label": labels_names[mapped_labels[i]].strip(),
                        }
                        f.write(json.dumps(entry) + "\n")
                print(f"\nSaved evaluation predictions to {log_path}")

        # 2. Processing for Classification Mode (Custom Head)
        else:
            if predictions.ndim == 3:
                predictions = np.argmax(predictions, axis=-1)

            # Flatten and mask
            predictions = predictions.reshape(-1)
            labels = labels.reshape(-1)
            mask = labels != -100
            predictions = predictions[mask]
            labels = labels[mask]

            num_metrics_labels = 3
            labels_names = ["BC  ", "INT ", "NONE"]

        if len(labels) == 0:
            return {}

        acc = (predictions == labels).mean()

        precision = []
        recall = []
        f1 = []

        cm = np.zeros((num_metrics_labels, num_metrics_labels), dtype=int)
        for p, l in zip(predictions, labels):
            if 0 <= p < num_metrics_labels and 0 <= l < num_metrics_labels:
                cm[l, p] += 1

        for i in range(3):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp

            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0

            precision.append(p)
            recall.append(r)
            f1.append(f)

        print("\nConfusion Matrix:")
        header = "Pred -> | " + " | ".join(labels_names) + " |"
        print(header)
        print("-" * len(header))
        for idx, row in enumerate(cm):
            print(f"GT {labels_names[idx]} | {' | '.join(f'{x:4d}' for x in row)} |")

        metrics = {
            "accuracy": acc,
            "f1_macro": np.mean(f1),
            "precision_macro": np.mean(precision),
            "recall_macro": np.mean(recall),
        }
        for i in range(num_metrics_labels):
            for j in range(num_metrics_labels):
                metrics[f"cm_{i}_{j}"] = cm[i, j]

        return metrics

    # 4. Training Arguments
    common_args = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "optim": "adamw_8bit",
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "logging_steps": args.logging_steps,
        "eval_strategy": "steps",
        "eval_steps": args.eval_steps,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": args.num_saves,
        "fp16": not torch.cuda.is_bf16_supported(),
        "bf16": torch.cuda.is_bf16_supported(),
        "report_to": "tensorboard",
        "load_best_model_at_end": True,
        "remove_unused_columns": True,
        "ddp_find_unused_parameters": False,
        "gradient_checkpointing": True,
    }

    if args.style == "chat":
        training_args = Seq2SeqTrainingArguments(
            **common_args,
            predict_with_generate=True,
            generation_config=GenerationConfig(max_new_tokens=16),
        )
    else:
        if args.style == "soft_classification":
            common_args["label_names"] = ["boundary_indices", "target_probs", "num_boundaries"]
        training_args = TrainingArguments(**common_args)

    # 5. Trainer
    if args.style == "chat":
        trainer_cls = Seq2SeqTrainer
    elif args.style == "soft_classification":
        trainer_cls = SoftLabelTokenClassificationTrainer
    else:
        trainer_cls = Trainer

    callbacks = (
        [ConfusionMatrixLoggingCallback()] if args.style in {"chat", "classification"} else []
    )
    trainer = None  # Initialize for closure
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics if args.style in {"chat", "classification"} else None,
        preprocess_logits_for_metrics=(
            preprocess_logits_for_metrics if args.style == "classification" else None
        ),
        callbacks=callbacks,
    )

    return trainer


def main():
    args = parse_args()
    model, tokenizer = build_model(args)
    tokenized_datasets, data_collator = build_datasets(args, tokenizer)
    trainer = build_trainer(args, model, tokenizer, tokenized_datasets, data_collator)

    # 6. Train
    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume)

    # 7. Save
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model and tokenizer saved to {args.output_dir}")


if __name__ == "__main__":
    main()
