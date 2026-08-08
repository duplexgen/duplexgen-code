# Official Implementation of DuplexGen: Adaptive Synthesis of Human–AI Turn-Taking Dialogues

[Takyoung Kim](https://youngerous.github.io/)<sup>&ast;</sup>,
[Kang-wook Kim](https://kwkim.me/)<sup>&ast;</sup>,
[Sang Hoon Woo](https://tonywoo.me/),
[Julia Hirschberg](https://www.cs.columbia.edu/~julia/),
[Gunhee Kim](https://vision.snu.ac.kr/gunhee/),
[Dilek Hakkani-Tür](https://siebelschool.illinois.edu/about/people/faculty/dilek)

<sup>&ast;</sup>Equal contribution

[![arXiv](https://img.shields.io/badge/arXiv-2607.26178-b31b1b.svg)](https://arxiv.org/abs/2607.26178)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

This repository contains the **data-generation pipeline for DuplexGen**, including text-dialogue collection, spoken-style conversion, turn-taking slot identification, turn-taking prediction, turn-taking dialogue generation, and speech synthesis with Chatterbox.

📄 [Paper](https://arxiv.org/abs/2607.26178) · 💻 [Code](https://github.com/duplexgen/duplexgen-code) · 🎛️ [Finetune](https://github.com/duplexgen/personaplex-finetune) · 🤗 [Corpus](https://huggingface.co/datasets/DuplexGen/duplexgen-corpus) · 🤗 [Spoken](https://huggingface.co/datasets/DuplexGen/duplexgen-spoken) · 🌐 [Demo](https://duplexgen.github.io/)

## Pipeline overview

| Stage | What it does | Docs |
|---|---|---|
| 1. Spoken-style conversion | Rewrite clean text dialogues from six scenario datasets into spoken-style transcripts. | [stage1-speechify](docs/stage1-speechify.md) |
| 2. Slot identification | Detect candidate intra-utterance action points via heuristic + LLM boundary detection. | [stage2-slots](docs/stage2-slots.md) |
| 3. Turn-taking prediction | Train or run a turn-taking predictor over the identified slots. | [stage3-prediction](docs/stage3-prediction.md) |
| 4. Turn-taking dialogue generation | Query the predictor at each slot and insert the chosen behavior, then filter role-confused outputs. | [stage4-generation](docs/stage4-generation.md) |
| 5. TTS rendering | Render the result to two-channel audio with Chatterbox. | [stage5-tts](docs/stage5-tts.md) |

Also: [using the released corpus](docs/CORPUS.md) ·
[troubleshooting](docs/TROUBLESHOOTING.md) ·
[data licenses](docs/DATA_LICENSES.md)

## Setup

This project needs **two separate environments**. Stage 5 pins
`transformers==4.46.3` through the vendored Chatterbox package, while Stage 3
requires `transformers>=4.53`; they cannot share an interpreter.

```bash
# Stages 1-4: dialogue generation, slot identification, prediction, synthesis
python3 -m venv .venv                       # Python 3.10+
.venv/bin/pip install -r requirements.txt

# Stage 5: TTS rendering. Python 3.11 ONLY (Chatterbox requires >=3.11,<3.12)
python3.11 -m venv .venv-tts
.venv-tts/bin/pip install -r requirements-stage5.txt
.venv-tts/bin/pip install -e duplexgen/tts/chatterbox
```

**Run everything from the repository root.** Stages 1-4 are modules of the
`src/` package, invoked as `.venv/bin/python -m src.<module>`, not as file
paths. The Stage 5 scripts are run as files with the *other* interpreter
(`.venv-tts/bin/python tts_render/convert_spoken.py`). The Stage 1-4
environment can also be installed as a package (`.venv/bin/pip install -e .`),
which puts the `src` modules on the path from anywhere.

**System dependencies.** `nemo-text-processing` (Stage 5 text normalization)
depends on `pynini`, which ships manylinux wheels for x86_64 CPython 3.8-3.13
and installs cleanly there. macOS and `aarch64` have no wheel and need OpenFst
headers first — `sudo apt-get install -y libfst-dev`, or take `pynini` from
conda-forge to avoid the build entirely.

**Reproducing our exact environment.** The requirements files carry ranges;
`constraints.txt` carries the exact versions the pipeline was last validated
against, and covers the Stage 1-4 (`.venv`) environment only. Stage 5's
versions come from the vendored `duplexgen/tts/chatterbox/pyproject.toml` plus
the `numpy<2` bound in `requirements-stage5.txt`.

```bash
.venv/bin/pip install -r requirements.txt -c constraints.txt
```

**LLM backend.** Every generation stage talks to a chat LLM through the OpenAI
Python client, and the backend is chosen by the *model name* you pass:

- **OpenAI models** (`gpt-4.1`, `gpt-4o`, `gpt-5`, ...): reads your key from
  `OPENAI_API_KEY`. `--base_url` / `--api_key` are ignored.
- **Open-weight models** (e.g. `Qwen/Qwen3-32B`): posts to an OpenAI-compatible
  server at `--base_url` (default `http://localhost:8000/v1`, `--api_key`
  default `EMPTY`). Serve it yourself, e.g. `vllm serve Qwen/Qwen3-14B`. Stage 4
  also accepts separate endpoints for the boundary detector
  (`--boundary_base_url`) and the turn-taking predictor (`--tt_base_url`).

To run the test suite and the lint/format checks CI gates on, add the
development tooling (`pytest`, `ruff`, `black`):

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest
```

## Quickstart

The five stages end to end on one scenario. Each stage's own doc covers the
rest of its flags; `--dataset interviewer` needs no local download, so this
runs as written.

```bash
export OPENAI_API_KEY=sk-...

# Stage 1 -- convert source dialogues to spoken style
.venv/bin/python -m src.speechify_run \
  --dataset interviewer --save_dir results/ --llm_model_name gpt-4.1

# Fetch the released corpus and unpack both halves (see docs/CORPUS.md)
.venv/bin/hf download DuplexGen/duplexgen-corpus \
  --repo-type dataset --local-dir duplexgen-corpus/
.venv/bin/python tools/unpack_corpus.py --kind annotations \
  --src duplexgen-corpus/annotations/INT/train.jsonl --code INT --split train
.venv/bin/python tools/unpack_corpus.py --kind dialogues \
  --src duplexgen-corpus/dialogues/INT/train.jsonl --code INT --split train

# Stage 3 -- train the turn-taking predictor
.venv/bin/python -m src.train_turntaking_hf \
  --model_name_or_path Qwen/Qwen3-4B \
  --input_root data-annotations/ \
  --output_dir ./output/turntaking_qwen3_4b

# Stages 2+4 -- detect slots and generate dialogues with turn-taking
.venv/bin/python -m src.synthesis.run \
  --dataset interviewer --split train \
  --input_root data-dialogues/ \
  --save_root outputs/generated_dialogues_with_tt \
  --llm_model_name gpt-4.1 --boundary_model_name gpt-4.1-mini \
  --hf_model_name_or_path ./output/turntaking_qwen3_4b --hf_load_in_4bit

# Stage 5 -- render to two-channel audio (note the other interpreter)
.venv-tts/bin/python tts_render/convert_spoken.py \
  --prompt_dir tts_render/prompt_wavs \
  --input_glob 'outputs/generated_dialogues_with_tt/**/*.json' \
  --save_dir outputs/audios_chatterbox --num_variants 10
```

Stage 3 training is optional if you only want to generate dialogues — Stage 4
can query a chat model instead, see [stage4-generation](docs/stage4-generation.md).
Backchannel text is filled in by a separate pass, `src.synthesis.run_add_bc`.

## Data

- **`DuplexGen/duplexgen-corpus`**: generated dialogues
  (`dialogues/<SCENARIO>/train.jsonl`, with per-word turn-taking decisions) and
  the human slot-level preference annotations
  (`annotations/<SCENARIO>/{train,test}.jsonl`) collected over their slots.
- **`DuplexGen/duplexgen-spoken`**: rendered audio for those dialogues.

Dialogue text is generated by **Qwen3.5-122B-A10B**. No third-party raw source
text is redistributed; only DuplexGen-generated dialogues and our own human
annotations are released.

See [`DATA_CARD.md`](DATA_CARD.md) for the schema and per-scenario counts, and
[`docs/CORPUS.md`](docs/CORPUS.md) for converting the release into the layout
the stages read.

## Fine-tuning PP-DG

**PP-DG** (PersonaPlex trained on DuplexGen data) is the full-duplex model
behind the paper's headline result. The LoRA fine-tuning code that trains it
lives in
[`duplexgen/personaplex-finetune`](https://github.com/duplexgen/personaplex-finetune).
No model checkpoints are included in this release.

## License

Code is released under the **Apache License 2.0** (see [`LICENSE`](LICENSE)).
The released **data** is not covered by it: each scenario inherits the license
of its upstream source dataset. See
[`docs/DATA_LICENSES.md`](docs/DATA_LICENSES.md) for the per-scenario
table and attribution notes.

Audio is rendered with [Chatterbox](https://github.com/resemble-ai/chatterbox)
TTS by Resemble AI (MIT); a trimmed copy is vendored at
`duplexgen/tts/chatterbox/`.

## Citation

If you use this code, the corpus, or the rendered audio, please cite:

```bibtex
@article{kim2026duplexgen,
  title   = {{DuplexGen: Adaptive Synthesis of Human--AI Turn-Taking Dialogues}},
  author  = {Kim, Takyoung and Kim, Kang-wook and Woo, Sang Hoon and
             Hirschberg, Julia and Kim, Gunhee and Hakkani-T{\"u}r, Dilek},
  journal = {arXiv preprint arXiv:2607.26178},
  year    = {2026},
  url     = {https://arxiv.org/abs/2607.26178}
}
```
