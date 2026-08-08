import argparse
import glob
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import numpy as np
from openai import OpenAI
from tqdm import tqdm

from src.llm_client import is_openai_model, make_client, no_thinking_extra_body
from src.turntaking_common import human_probs_to_train_dist

LABELS = ["backchannel", "take_floor", "silent"]

SYSTEM_PROMPT = """\
You are an expert at analyzing spoken conversation dynamics and predicting turn-taking behavior.

At each word boundary in a conversation, you must estimate the probability of three listener behaviors:
- "backchannel": The listener gives a brief acknowledgment ("uh-huh", "yeah", "I see", "mm") while the speaker continues talking.
- "take_floor": The listener interrupts or takes over the conversation turn.
- "silent": The listener remains silent and lets the speaker continue without any response.

Respond ONLY with a JSON object on a single line. The three probabilities must sum to 1.0.
Example: {"backchannel": 0.1, "take_floor": 0.2, "silent": 0.7}
"""

TASK_PROMPT_TEMPLATE = """\
Below is a conversation. The speaker's utterance is cut off at a word boundary (marked with <<BOUNDARY>>).
Predict the probability of each listener behavior at this exact moment.

{context_str}Speaker: {partial_utterance} <<BOUNDARY>>

Output JSON only: {{"backchannel": <float>, "take_floor": <float>, "silent": <float>}}"""


# ---------------------------------------------------------------------------
# Data loading (same logic as inference_turntaking_swbd.py)
# ---------------------------------------------------------------------------


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

                    examples.append(
                        {
                            "scenario": scenario,
                            "split": split,
                            "example_id": obj.get("example_id", os.path.basename(file_path)),
                            "source_file": file_path,
                            "turn_index": int(turn_idx),
                            "word_index": int(word_idx),
                            "context": context,
                            "user_utterance": content,
                            "target_probs": probs,
                        }
                    )
    return examples


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def build_prompt(example: Dict[str, Any]) -> str:
    words = example["user_utterance"].split()
    word_idx = example["word_index"]
    partial = " ".join(words[: word_idx + 1])

    role_map = {"user": "Speaker", "assistant": "Listener"}
    context_lines = []
    for turn in example["context"]:
        role_label = role_map.get(turn["role"], turn["role"].capitalize())
        context_lines.append(f"{role_label}: {turn['content']}")
    context_str = "\n".join(context_lines) + "\n" if context_lines else ""

    return TASK_PROMPT_TEMPLATE.format(
        context_str=context_str,
        partial_utterance=partial,
    )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> block from response, returning only content after </think>."""
    idx = text.rfind("</think>")
    if idx != -1:
        return text[idx + len("</think>") :].strip()
    return text


def parse_response(text: str) -> Optional[List[float]]:
    """Extract backchannel/take_floor/silent probs from LLM output."""
    # Try to find a JSON object in the response
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group())
    except json.JSONDecodeError:
        return None

    bc = float(obj.get("backchannel", 0.0) or 0.0)
    tf = float(obj.get("take_floor", 0.0) or 0.0)
    sl = float(obj.get("silent", 0.0) or 0.0)

    s = bc + tf + sl
    if s <= 0:
        return None
    return [bc / s, tf / s, sl / s]


# ---------------------------------------------------------------------------
# KL divergence
# ---------------------------------------------------------------------------


def safe_kl(human: np.ndarray, pred: np.ndarray, eps: float = 1e-12) -> float:
    human = np.asarray(human, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    pred = np.clip(pred, eps, 1.0)
    mask = human > 0
    return float(np.sum(human[mask] * np.log(human[mask] / pred[mask])))


# ---------------------------------------------------------------------------
# Single-example inference
# ---------------------------------------------------------------------------


def infer_one(
    example: Dict[str, Any],
    client: OpenAI,
    model: str,
    max_retries: int,
    retry_delay: float,
    temperature: float,
    thinking: bool,
    supports_thinking_toggle: bool = True,
) -> Dict[str, Any]:
    prompt = build_prompt(example)
    human = np.array(example["target_probs"], dtype=np.float64)

    # Thinking mode: allow room for <think> block; non-thinking: compact output
    max_tokens = 4096 if thinking else 64
    # The thinking-off `extra_body` is a vLLM extension api.openai.com rejects
    # with a 400; no_thinking_extra_body() returns None for an OpenAI model.
    extra_body = None
    if not thinking and supports_thinking_toggle:
        extra_body = no_thinking_extra_body(model)

    pred = None
    raw_response = ""
    status = "ok"
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra_body,
            )
            raw_response = resp.choices[0].message.content or ""
            content = strip_thinking(raw_response) if thinking else raw_response
            pred = parse_response(content)
            if pred is not None:
                break
            status = "unparsable_response"
        except Exception as e:
            status = "api_error"
            raw_response = f"ERROR: {e}"
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))

    if pred is None:
        # Fallback: uniform distribution. Flagged so the caller can tell a real
        # prediction from a placeholder -- an all-fallback run still produces a
        # finite mean KL, which used to look like a successful evaluation.
        pred = [1 / 3, 1 / 3, 1 / 3]
    else:
        status = "ok"

    pred_arr = np.array(pred, dtype=np.float64)
    kl = safe_kl(human, pred_arr)

    return {
        "scenario": example["scenario"],
        "split": example["split"],
        "example_id": example["example_id"],
        "source_file": example["source_file"],
        "turn_index": example["turn_index"],
        "word_index": example["word_index"],
        "user_utterance": example["user_utterance"],
        "human_probs": {
            "backchannel": float(human[0]),
            "take_floor": float(human[1]),
            "silent": float(human[2]),
        },
        "pred_probs": {
            "backchannel": float(pred_arr[0]),
            "take_floor": float(pred_arr[1]),
            "silent": float(pred_arr[2]),
        },
        "pred_label": LABELS[int(np.argmax(pred_arr))],
        "kl_human_pred": float(kl),
        "status": status,
        "is_fallback": status != "ok",
        "raw_response": raw_response,
    }


# ---------------------------------------------------------------------------
# Batched inference with optional parallelism
# ---------------------------------------------------------------------------


def run_inference(
    examples: List[Dict[str, Any]],
    client: OpenAI,
    model: str,
    max_workers: int,
    max_retries: int,
    retry_delay: float,
    temperature: float,
    thinking: bool,
    supports_thinking_toggle: bool = True,
) -> List[Dict[str, Any]]:
    outputs: List[Optional[Dict[str, Any]]] = [None] * len(examples)

    def task(idx_ex):
        idx, ex = idx_ex
        return idx, infer_one(
            ex,
            client,
            model,
            max_retries,
            retry_delay,
            temperature,
            thinking,
            supports_thinking_toggle,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(task, (i, ex)): i for i, ex in enumerate(examples)}
        for fut in tqdm(as_completed(futures), total=len(examples), desc="Inferencing"):
            idx, result = fut.result()
            outputs[idx] = result

    return [o for o in outputs if o is not None]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Prompting-based inference for turn-taking prediction using "
            "an OpenAI-compatible LLM API (e.g. Qwen/Qwen3-14B via vLLM)."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-14B",
        help="Model name as served by the API.",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=None,
        help="OpenAI-compatible endpoint of the self-hosted server "
        "(default http://{--server}:{--server_port}/v1). Ignored for OpenAI models, "
        "which always read OPENAI_API_KEY and talk to api.openai.com.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="EMPTY",
        help="Key for the self-hosted endpoint. Ignored for OpenAI models.",
    )
    parser.add_argument(
        "--server",
        type=str,
        default="localhost",
        help="Server hostname for http://{server}:{server_port}/v1 (used only if --base_url is not given).",
    )
    parser.add_argument(
        "--server_port",
        type=int,
        default=8000,
        help="Server port for the API (default: 8000).",
    )
    parser.add_argument(
        "--input_root",
        type=str,
        required=True,
        help="Root dir of input annotated dialogues (e.g. a local copy of the DuplexGen/duplexgen-corpus annotations).",
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--scenarios", type=str, default="all")
    parser.add_argument("--use_last_n_history", type=int, default=4)
    parser.add_argument("--max_workers", type=int, default=8, help="Number of parallel API calls.")
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--retry_delay", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable thinking mode (default: True). Use --no-thinking to disable.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs/inference_turntaking_prompting.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Same backend rule as every other stage: the model name picks the backend.
    openai_backend = is_openai_model(args.model)
    base_url = args.base_url or f"http://{args.server}:{args.server_port}/v1"
    client = make_client(args.model, api_key=args.api_key, base_url=base_url)

    print(f"API base: {'api.openai.com (OPENAI_API_KEY)' if openai_backend else base_url}")
    print(f"Model: {args.model}")
    print(f"Thinking mode: {args.thinking}")

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

    print("Running inference...")
    per_sample_results = run_inference(
        examples=examples,
        client=client,
        model=args.model,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        temperature=args.temperature,
        thinking=args.thinking,
        supports_thinking_toggle=not openai_backend,
    )

    fallbacks = [x for x in per_sample_results if x["is_fallback"]]
    n_fallback = len(fallbacks)
    if n_fallback == len(per_sample_results):
        first_error = fallbacks[0]["raw_response"] if fallbacks else "(no response)"
        raise RuntimeError(
            f"All {n_fallback} requests failed, so every prediction is the uniform "
            "fallback and the reported KL would be meaningless. First failure:\n"
            f"  {first_error}\n"
            "Check that the model name and endpoint match: OpenAI models need "
            "OPENAI_API_KEY, self-hosted models need a server reachable at "
            f"{base_url} (--base_url / --server / --server_port)."
        )

    mean_kl = float(np.mean([x["kl_human_pred"] for x in per_sample_results]))
    out = {
        "model": args.model,
        "api_base": "openai" if openai_backend else base_url,
        "split": args.split,
        "scenarios": args.scenarios,
        "temperature": args.temperature,
        "thinking": args.thinking,
        "n_samples": len(per_sample_results),
        "n_fallback": n_fallback,
        "mean_kl_human_pred": mean_kl,
        "per_sample_results": per_sample_results,
    }

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Wrote: {args.output_path}")
    if n_fallback:
        print(
            f"[WARN] {n_fallback}/{out['n_samples']} requests fell back to the uniform "
            "distribution; the mean KL below is diluted by them."
        )
    print(f"n_samples={out['n_samples']} mean_kl_human_pred={out['mean_kl_human_pred']:.6f}")


if __name__ == "__main__":
    main()
