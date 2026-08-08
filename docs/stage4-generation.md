# Stage 4: Turn-taking dialogue generation

Queries the predictor at each slot and inserts the chosen behavior, then filters
role-confused outputs.

This stage regenerates each dialogue turn by turn and, at every slot Stage 2
detects inside a user turn, samples a turn-taking behavior from the predictor's
distribution. It drives **three separate LLM roles**, each with its own model
flag, so decide what serves each one before running:

| Role | Flag | Default | What it does |
|---|---|---|---|
| Writer | `--llm_model_name` (`--api_key`, `--base_url`) | `gpt-4.1` | Generates each turn's text and judges when the dialogue is done |
| Boundary detector (Stage 2) | `--boundary_model_name` (`--boundary_api_key`, `--boundary_base_url`) | `gpt-4.1-mini` | Marks clause boundaries, which become the candidate slots |
| Turn-taking predictor (Stage 3) | `--tt_model_name` (`--tt_api_key`, `--tt_base_url`) | `Qwen/Qwen3-14B` | Scores `floor_taking` / `backchannel` / `silence` at each slot |

Each follows the model-name rule from [Setup](../README.md#setup)
independently, so the defaults mean a bare run needs **both** `OPENAI_API_KEY`
and a self-hosted `Qwen/Qwen3-14B` at `http://localhost:8000/v1`. Name every
endpoint explicitly:

```bash
export OPENAI_API_KEY=sk-...
.venv/bin/python -m src.synthesis.run \
  --dataset interviewer --split train \
  --input_root data-dialogues/ \
  --save_root outputs/generated_dialogues_with_tt \
  --llm_model_name gpt-4.1 --boundary_model_name gpt-4.1-mini \
  --tt_model_name Qwen/Qwen3-32B --tt_base_url http://localhost:8000/v1
```

## Using the Stage 3 predictor instead

`--tt_model_name` asks a chat model for verbalized probabilities. To use the
LoRA token classifier Stage 3 trains — the configuration behind the released
corpus, whose dialogues carry `"tt_mode": "hf_classification"` — pass
`--hf_model_name_or_path` instead. It takes priority over `--tt_model_name`,
needs no `--tt_base_url`, and loads in process, so this run needs a GPU:

```bash
.venv/bin/python -m src.synthesis.run \
  --dataset interviewer --split train \
  --input_root data-dialogues/ \
  --save_root outputs/generated_dialogues_with_tt \
  --llm_model_name gpt-4.1 --boundary_model_name gpt-4.1-mini \
  --hf_model_name_or_path ./output/turntaking_qwen3_4b --hf_load_in_4bit
```

Point it at the `--output_dir` of `src.train_turntaking_hf`; the base model is
read from the adapter's `adapter_config.json`, so there is no separate
`--model_name_or_path` here.

## Inputs and outputs

`--dataset` accepts `socraticlm | multiwoz | interviewer | negotiator |
persuader | soda | all`; `--split` is `train | test`. `--input_root` is required
and expects the `text_dialogue_<dataset>/<split>/*.json` layout produced by
`src/prepare_corpus.py` — **not** the published HF layout; convert with
`tools/unpack_corpus.py --kind dialogues` first (see [CORPUS.md](CORPUS.md)).
That half is train-only, so `--split test` needs dialogues of your own from
Stage 1. Results are written to
`<save_root>/text_dialogue_<dataset>/<split>/*.json`, and an existing output
file is skipped, so a re-run resumes. `--max_dialogues` (1000) and `--max_turns`
(20) bound the cost of a trial run.

Backchannel decisions are recorded per slot in each turn's `history`, not
inserted into `content`: only `[TAKE_FLOOR]` appears in the transcript text.
This matches the released corpus, and Stage 5 renders backchannels from the
slot metadata.

## Backchannel content

Optional, and comes last, filling in what each backchannel slot actually says.
It is a fourth endpoint, defaulting to `Qwen/Qwen3-14B` at
`http://localhost:8008/v1` — port **8008**, not the 8000 the other stages
default to, so that it can be a second server:

```bash
.venv/bin/python -m src.synthesis.run_add_bc \
  --input_root outputs/generated_dialogues_with_tt \
  --output_root outputs/generated_dialogues_with_hf_swbd_plus_backchannels \
  --model_name Qwen/Qwen3-32B --base_url http://localhost:8000/v1
```

Both roots are required. It writes the generated text into each backchannel
slot's `content` field, leaving the transcript text unchanged.

Source: `src/synthesis/*.py`.

Next: [Stage 5](stage5-tts.md).
