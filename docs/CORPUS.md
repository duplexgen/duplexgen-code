# Using the released corpus with this code

`DuplexGen/duplexgen-corpus` ships in the Hugging Face layout, which is not what
the stages read. Convert it once with `tools/unpack_corpus.py`.

**Which half you need.** Stages 1 and 2 need neither — Stage 1 generates
dialogues from the third-party sources, and Stage 2 runs inside the Stage 4
invocation.

| | `dialogues/` | `annotations/` |
|---|---|---|
| Contains | DuplexGen-generated dialogue text + per-word turn-taking decisions | the same dialogues, plus human rater votes at annotated word boundaries |
| Splits | `train` only | `train` and `test` |
| Needed by | Stage 4 input, Stage 5 rendering | Stage 3 training and evaluation (both splits) |

First download the corpus from the Hub:

```bash
.venv/bin/hf download DuplexGen/duplexgen-corpus \
  --repo-type dataset --local-dir duplexgen-corpus/
```

The Hub layout (`<KIND>/<CODE>/<split>.jsonl`, one file per split) and the
layout every stage reads (`text_dialogue_<dataset>/<split>/*.json`, one file per
dialogue) differ in scheme, granularity, and extension. Convert once:

```bash
.venv/bin/python tools/unpack_corpus.py --kind annotations \
  --src duplexgen-corpus/annotations/INT/train.jsonl \
  --code INT --split train
```

This writes `data-annotations/text_dialogue_interviewer/train/*.json`.
`--out-root` defaults per `--kind` — `data-annotations/` and `data-dialogues/`
— so the two halves land apart without your having to name them. `--input_root`
on each stage is that root, and each stage globs for
`text_dialogue_<dataset>/<split>/` underneath it.

The command handles one code and one split, so unpacking the whole corpus is a
loop. Stage 3 evaluates on `test` by default, so unpack both splits:

```bash
for code in TEA PLN INT NEG PER SOC; do
  for split in train test; do
    .venv/bin/python tools/unpack_corpus.py --kind annotations \
      --src duplexgen-corpus/annotations/$code/$split.jsonl \
      --code $code --split $split
  done
done
```

For Stage 4, unpack the dialogues instead; they default to their own root:

```bash
for code in TEA PLN INT NEG PER SOC; do
  .venv/bin/python tools/unpack_corpus.py --kind dialogues \
    --src duplexgen-corpus/dialogues/$code/train.jsonl \
    --code $code --split train
done
```

Note `dialogues/` ships **train only**, so Stage 4's `--split test` has no
corpus behind it — run Stage 4 on `--split train`, or generate your own test
dialogues with Stage 1.

`tools/pack_corpus.py` performs the inverse (local → Hub layout).

> Scenario codes are **not** case-insensitive shorthands for the directory
> names. Release code `SOC` is **soda**, while the internal lowercase `soc` is
> **socraticlm**. Always go through `tools/unpack_corpus.py`'s `CODE2DIR` rather
> than lowercasing a code yourself.
