"""Expand the released HuggingFace corpus back into the per-dialogue JSON
layout the pipeline reads.

Inverse of pack_corpus.py:

    <kind>/<CODE>/<split>.jsonl   ->   text_dialogue_<dirname>/<split>/*.json

Lossy by design: pack_corpus drops `config`, `meta.dataset`, `meta.split`, and
`meta.reconstructed_from`. dataset/split are reconstructed from the CLI
arguments; reconstructed_from is intentionally not resurrected.
"""

import argparse, json, pathlib, re, sys

# Canonical release code -> private dir name. Inverse of pack_corpus.DIR2CODE.
#
# HAZARD: this repo carries two scenario vocabularies that collide under
# case-folding. Release "SOC" means soda, while the internal lowercase code
# "soc" means socraticlm. Never casefold a scenario code and never compare
# across the two namespaces without going through a table.
CODE2DIR = {
    "TEA": "socraticlm",
    "PLN": "multiwoz",
    "INT": "interviewer",
    "NEG": "negotiator",
    "PER": "persuader",
    "SOC": "soda",
}

# Distinct per kind on purpose. Both kinds unpack to the same
# text_dialogue_<dirname>/<split>/ subpath, so a single default root would put
# generated dialogues and human annotations under one name -- and Stage 3 reads
# only the latter while Stage 4 reads only the former. Keeping the roots apart
# by default means --input_root names which half you meant.
DEFAULT_OUT_ROOT = {
    "dialogues": "data-dialogues",
    "annotations": "data-annotations",
}

_UNSAFE = re.compile(r"[^\w\-.]")


def safe_id(example_id) -> str:
    """Filename sanitization, identical to src/prepare_corpus.py:376."""
    return _UNSAFE.sub("_", str(example_id))


def _meta(record, dir_name, split):
    meta = {"dataset": dir_name, "split": split}
    if record.get("license") is not None:
        meta["license"] = record["license"]
    return meta


def unpack_dialogue(record, dir_name, split):
    hist = []
    for t in record.get("history", []):
        hist.append(
            {
                "role": t.get("role"),
                "content": t.get("content"),
                # pack_corpus renamed the nested per-word decision list to
                # "segments"; the pipeline reads it back as "history".
                "history": t.get("segments", []),
            }
        )
    meta = _meta(record, dir_name, split)
    for key in ("style", "disfluency_target"):
        if record.get(key) is not None:
            meta[key] = record[key]
    return {
        "example_id": record.get("example_id"),
        "speakers": record.get("speakers"),
        "config": {},
        "history": hist,
        "context": record.get("context"),
        "meta": meta,
    }


def unpack_annotation(record, dir_name, split):
    hist = []
    for t in record.get("history", []):
        turn = {"role": t.get("role"), "content": t.get("content")}
        # Emit "boundaries" ONLY on annotated turns. The release format carries
        # the key on every turn (empty where nothing was annotated), but the
        # pipeline selects annotated turns by key *presence* -- see
        # build_human_boundary_examples() in src/train_turntaking_hf.py.
        # Copying the empty lists through made every unpacked dialogue look
        # like 15 annotated turns instead of 2.
        if t.get("boundaries"):
            turn["boundaries"] = t["boundaries"]
        hist.append(turn)
    return {
        "example_id": record.get("example_id"),
        "history": hist,
        "meta": _meta(record, dir_name, split),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["dialogues", "annotations"], required=True)
    ap.add_argument("--src", required=True, help="path to <CODE>/<split>.jsonl")
    ap.add_argument(
        "--code",
        required=True,
        choices=sorted(CODE2DIR),
        help="release scenario code (case-sensitive)",
    )
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument(
        "--out-root",
        default=None,
        help="written as <out-root>/text_dialogue_<dirname>/<split>/*.json. "
        f"Defaults per --kind ({', '.join(f'{k} -> {v}' for k, v in DEFAULT_OUT_ROOT.items())}), "
        "which keeps the two kinds apart: both write the same "
        "text_dialogue_<dirname>/<split>/ subpath, so one shared root would "
        "mix them under one name.",
    )
    a = ap.parse_args()

    dir_name = CODE2DIR[a.code]
    fn = unpack_dialogue if a.kind == "dialogues" else unpack_annotation
    out_root = a.out_root or DEFAULT_OUT_ROOT[a.kind]
    out = pathlib.Path(out_root) / f"text_dialogue_{dir_name}" / a.split
    out.mkdir(parents=True, exist_ok=True)

    n = 0
    with open(a.src, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = fn(json.loads(line), dir_name, a.split)
            path = out / f"{safe_id(rec['example_id'])}.json"
            with path.open("w", encoding="utf-8") as w:
                json.dump(rec, w, indent=2, ensure_ascii=False)
            n += 1
    print(f"wrote {n} records -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
