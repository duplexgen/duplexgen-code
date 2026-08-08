"""Package DuplexGen generated dialogues + slot annotations into HF-ready JSONL.
No third-party source text is read; only DuplexGen-generated outputs."""

import argparse, json, pathlib, sys

# private dir-name -> canonical code
DIR2CODE = {
    "socraticlm": "TEA",
    "multiwoz": "PLN",
    "interviewer": "INT",
    "neg": "NEG",
    "negotiator": "NEG",
    "persuader": "PER",
    "soda": "SOC",
}
CODE2LICENSE = {
    "TEA": "Apache-2.0",
    "PLN": "MIT",
    "INT": "CC-BY-4.0",
    "NEG": "MIT",
    "PER": "Apache-2.0",
    "SOC": "CC-BY-4.0",
}


def pack_dialogue(record, scenario, license_id):
    # Per-word turn-taking decisions live in the NESTED turn["history"] list; preserve it as "segments".
    # Each segment is either {"full_content": ...} or
    # {"word_index", "probs":{floor_taking,backchannel,silence}, "decision", "inserted_token", (opt "content")}.
    hist = []
    for t in record.get("history", []):
        hist.append(
            {
                "role": t.get("role"),
                "content": t.get("content"),
                "segments": t.get("history", []),
            }
        )
    meta = record.get("meta") or {}
    return {
        "example_id": record.get("example_id"),
        "scenario": scenario,
        "license": license_id,
        "context": record.get("context"),
        "style": meta.get("style"),
        "disfluency_target": meta.get("disfluency_target"),
        "speakers": record.get("speakers"),
        "history": hist,
    }


def pack_annotation(record, scenario, license_id):
    hist = []
    for t in record.get("history", []):
        turn = {
            "role": t.get("role"),
            "content": t.get("content"),
            "boundaries": t.get("boundaries", []),
        }
        hist.append(turn)
    return {
        "example_id": record.get("example_id"),
        "scenario": scenario,
        "license": license_id,
        "history": hist,
    }


def _iter_json(dir_path):
    for f in sorted(pathlib.Path(dir_path).glob("*.json")):
        yield json.loads(f.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["dialogues", "annotations"], required=True)
    ap.add_argument("--src-dir", required=True, help="dir of *.json for one scenario+split")
    ap.add_argument("--dir-name", required=True, help="private scenario dir name (e.g. persuader)")
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument(
        "--out-root", default=str(pathlib.Path.home() / "duplexgen-release" / "hf-corpus")
    )
    a = ap.parse_args()
    code = DIR2CODE[a.dir_name]
    lic = CODE2LICENSE[code]
    fn = pack_dialogue if a.kind == "dialogues" else pack_annotation
    out = pathlib.Path(a.out_root) / a.kind / code
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    with (out / f"{a.split}.jsonl").open("w") as w:
        for rec in _iter_json(a.src_dir):
            w.write(json.dumps(fn(rec, code, lic), ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} records -> {out}/{a.split}.jsonl", file=sys.stderr)


if __name__ == "__main__":
    main()
