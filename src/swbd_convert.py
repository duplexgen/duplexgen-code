# swbd_convert_mp.py
import os
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from openai import OpenAI
from src.swbd_parse import parse_file, convert_to_streaming_dialogue_json

INPUT_ROOT = Path("swb1_dialogact_annot")
OUTPUT_ROOT = Path("results/swb1_dialogact_annot")


def process_one_file(utt_path_str: str) -> str:
    utt_path = Path(utt_path_str)

    with open(utt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    meta, utterances = parse_file(lines)

    # Create client inside the worker process
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    dialogue_json = convert_to_streaming_dialogue_json(
        utterances=utterances,
        context="Switchboard telephone conversation",
        meta=meta,
        client=client,
        filename=str(utt_path.stem),
    )

    out_path = OUTPUT_ROOT / (utt_path.stem + ".json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dialogue_json, f, ensure_ascii=False, indent=2)

    return str(out_path)


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    utt_files = sorted(INPUT_ROOT.glob("*/*.utt"))
    print(f"Found {len(utt_files)} .utt files")

    # Tune this.
    # If LLM calls are enabled, too many workers can just hammer rate limits.
    # If client=None (heuristics-only), you can increase this a lot.
    max_workers = min(os.cpu_count(), 8)

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(process_one_file, str(p)) for p in utt_files]

        for fut in tqdm(as_completed(futures), total=len(futures)):
            fut.result()  # will raise if worker errored


if __name__ == "__main__":
    main()
