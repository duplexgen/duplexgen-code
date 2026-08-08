import json
from pathlib import Path
from typing import Dict, Any
from tqdm import tqdm

from src.llm_client import make_client, no_thinking_extra_body

# ================================
# Configuration
# ================================
# Defaults are now Qwen (local, free). Override via CLI. The legacy gpt-4.1-mini
# path is still available with --prompt legacy --model_name gpt-4.1-mini.
MODEL_NAME = "Qwen/Qwen3-14B"
BASE_URL = "http://localhost:8008/v1"
API_KEY = "EMPTY"
PROMPT_KIND = "qwen"  # "qwen" (new, engineered) | "legacy" (original gpt prompt)

# Set from --input_root/--output_root, which are required: the old defaults
# named a directory no stage writes, so a bare run globbed nothing and exited 0.
INPUT_ROOT = None
OUTPUT_ROOT = None

client = None  # built in main() from the resolved config

# ================================
# Prompt
# ================================
SYSTEM_PROMPT = """
You are generating a LISTENER backchannel to insert into a conversation.

# Backchannel definition:
- A very short listener vocalization showing attention/understanding
- It does NOT take the floor and does NOT change the topic
- The speaker continues immediately after it

# Insertion rule:
- Output what the listener says exactly at [INSERT] while the speaker keeps talking.
- Do not add any other text, quotes, labels, or explanations.

# Output constraints:
- 1–2 words only (3 only if absolutely necessary)
- lowercase
- no emojis, no ellipses, no exclamation marks
- "huh?" may include a question mark; otherwise avoid punctuation

# Decision rubric (choose the mildest that fits):
- ack/continue: mhm, mm-hmm, uh-huh, yeah, right, okay
- mild surprise/interest (new/unexpected info): oh, wow, really
- confusion (ONLY if unclear/contradictory): huh?

# Default:
- If uncertain, output "mhm".

# Allowed outputs (choose exactly one):
mhm | mm-hmm | uh-huh | yeah | right | okay | oh | wow | really | huh?"""

# New, engineered prompt for Qwen (do NOT edit SYSTEM_PROMPT above). Goal: pick the
# backchannel a real listener would actually say HERE, matched to the speaker's
# content and emotional register, while staying a true backchannel (no floor-take,
# no new topic, no question that demands an answer).
SYSTEM_PROMPT_QWEN = """You insert a LISTENER BACKCHANNEL at the [INSERT] point: the tiny
sound the listener makes while the speaker keeps talking. It does NOT take the floor,
NOT change the topic, and NOT ask anything that needs an answer. The speaker continues
right after it.

Pick the ONE backchannel a real person would say HERE. Match it to what the speaker is
saying and their emotional tone — do not default to the same token every time:
- neutral attention / "I'm following": mhm, mm-hmm, uh-huh, right, okay, yeah
- agreement / affirmation: yeah, right, exactly, totally, for sure
- new or surprising info: oh, oh wow, really, no way
- bad/sad news: oh no, aw, oh man
- mild understanding of a problem: i see, gotcha, ah
- genuine confusion (only if the speaker was unclear/contradictory): huh?, wait

Rules:
- 1-3 words, lowercase. No emojis, no ellipses, no exclamation marks. A question mark
  is allowed only for "huh?" / "wait?".
- Output ONLY the backchannel text — no quotes, labels, or explanation.
- Prefer the mildest option that fits; reserve "oh wow / really / no way / oh no" for
  genuinely notable content, not routine statements.
- It must read as natural spoken English in this exact spot."""


def _build_messages(context: str):
    sys_p = SYSTEM_PROMPT_QWEN if PROMPT_KIND == "qwen" else SYSTEM_PROMPT
    return [
        {"role": "system", "content": sys_p},
        {
            "role": "user",
            "content": f"# Conversation context:\n{context}\n\nGenerate the most natural listener backchannel for the [INSERT] point above.",
        },
    ]


def generate_backchannel(context: str) -> str:
    kwargs = dict(
        model=MODEL_NAME, messages=_build_messages(context), temperature=0.7, max_tokens=12
    )
    # disable thinking so the reply is just the backchannel; never sent to
    # api.openai.com, which 400s on the unknown field
    kwargs["extra_body"] = no_thinking_extra_body(MODEL_NAME)
    response = client.chat.completions.create(**kwargs)
    out = (response.choices[0].message.content or "").strip()
    out = out.split("</think>")[-1].strip().strip('"').strip().lower()
    return out


# ================================
# Core logic
# ================================
def process_dialogue(dialogue: Dict[str, Any]) -> Dict[str, Any]:
    turns = dialogue.get("history", [])

    for t_idx, turn in enumerate(turns):
        if "history" not in turn:
            continue

        words = turn["content"].split()
        for item in turn["history"]:
            if item.get("decision") != "backchannel":
                continue

            idx = item["word_index"]
            if idx >= len(words):
                continue

            # Current turn prefix up to the backchannel point
            current_prefix = " ".join(words[: idx + 1])
            current_postfix = " ".join(words[idx + 1 :])

            # Full context passed to LLM
            context_lines = []
            if t_idx > 0:
                context_lines.append("Conversation:")
                context_lines.append("[PREVIOUS]")
                context_lines.append(f"{turns[t_idx-1].get('content','').strip()}")

            context_lines.append("[SPEAKER CURRENT]")
            context_lines.append(f"{current_prefix} [INSERT] {current_postfix}")
            context = "\n".join(context_lines)

            # 3. Generate and Update
            bc = generate_backchannel(context)
            item["content"] = bc

    return dialogue


# ================================
# File traversal
# ================================
def collect_files(dataset=None, split=None):
    ds_part = f"text_dialogue_{dataset}" if dataset else "text_dialogue_*"
    split_part = split or "*"
    return list(INPUT_ROOT.glob(f"{ds_part}/{split_part}/*.json"))


# ================================
# Entry point
# ================================
def main(dataset=None, split=None):
    files = collect_files(dataset, split)
    if not files:
        raise SystemExit(
            f"No dialogues under {INPUT_ROOT}. --input_root must contain "
            f"text_dialogue_<dataset>/<split>/*.json, i.e. the --save_root of "
            f"`python -m src.synthesis.run`."
        )

    for in_path in tqdm(files, desc="Adding backchannels"):
        rel_path = in_path.relative_to(INPUT_ROOT)
        out_path = OUTPUT_ROOT / rel_path

        # Skip if already processed
        if out_path.exists():
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Defensive: a truncated/empty STAGE-1 input (e.g. a write interrupted by
        # a disk-quota error) must not abort the whole stage; skip it.
        try:
            with open(in_path, "r", encoding="utf-8") as f:
                dialogue = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"\n[run_add_bc] SKIP unreadable input {in_path.name}: {exc}")
            continue

        # A single malformed/empty LLM reply (e.g. JSONDecodeError inside
        # process_dialogue) must not abort the whole stage. Retry a few times
        # (transient 122B hiccups), then skip the dialogue as acceptable loss.
        processed = None
        for attempt in range(3):
            try:
                processed = process_dialogue(dialogue)
                break
            except Exception as exc:
                print(f"\n[run_add_bc] WARN {in_path.name} attempt {attempt+1}/3 failed: {exc}")
        if processed is None:
            print(f"[run_add_bc] SKIP {in_path.name} after 3 failed attempts (no bc added)")
            continue

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(processed, f, indent=2, ensure_ascii=False)


# ================================
# CLI
# ================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    parser.add_argument("--split")
    parser.add_argument("--input_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--model_name", default=MODEL_NAME, help="default Qwen/Qwen3-14B (local)")
    parser.add_argument(
        "--base_url", default=BASE_URL, help="vLLM OpenAI endpoint (ignored for gpt-* models)"
    )
    parser.add_argument("--api_key", default=API_KEY)
    parser.add_argument(
        "--prompt",
        choices=["qwen", "legacy"],
        default=PROMPT_KIND,
        help="qwen=new engineered prompt; legacy=original gpt prompt",
    )

    args = parser.parse_args()

    MODEL_NAME = args.model_name
    PROMPT_KIND = args.prompt
    client = make_client(MODEL_NAME, args.api_key, args.base_url)

    INPUT_ROOT = args.input_root
    OUTPUT_ROOT = args.output_root

    main(
        dataset=args.dataset,
        split=args.split,
    )
# ================================
