# Stage 1: Spoken-style conversion

Rewrites clean text dialogues from six scenario datasets into spoken-style
transcripts. See [Setup](../README.md#setup) for the environment and the
model-name rule that picks the LLM backend.

`interviewer` and `soda` pull their source dialogues from the Hugging Face Hub,
so they run with nothing downloaded ahead of time:

```bash
export OPENAI_API_KEY=sk-...
.venv/bin/python -m src.speechify_run \
  --dataset interviewer \
  --save_dir results/ \
  --llm_model_name gpt-4.1
```

The other scenarios read a local copy of their upstream corpus and need
`--data-root` (or, for `persuader`, `--input_path`):

```bash
.venv/bin/python -m src.speechify_run \
  --dataset socraticlm \
  --data-root /path/to/raw/datasets \
  --save_dir results/ \
  --llm_model_name gpt-4.1
```

Output lands in `<save_dir>/text_dialogue_<dataset>/{train,test}/*.json` — the
same layout Stages 2-4 read. Existing files are skipped, so a re-run resumes.

**Sample budget.** Every dialogue costs one LLM call, so each split is capped
by `--max_train_samples` / `--max_test_samples` (25 each by default; `0` means
no cap). Both count *dialogues*, and both sample at even spacing across the
whole source rather than taking a prefix, so a small budget still spans the
corpus. `--max_variants` controls how many retellings of one scenario to keep
for the corpora that ship several (`socraticlm`, `persuader`); raising it does
not inflate the per-split counts.

To smoke-test the stage without spending anything, `--test-parse` dumps the
parsed source dialogues and never calls the LLM:

```bash
.venv/bin/python -m src.speechify_run \
  --dataset interviewer --save_dir results/ --test-parse
```

The dump lands in `<save_dir>/parsed_source/<dataset>/<split>/*.json`, apart
from the converted output, so a later real run into the same `--save_dir` still
converts every dialogue.

## Source datasets

Two scenarios download automatically through `datasets.load_dataset`; the other
four must be obtained yourself and placed under `--data-root` at the exact paths
in `LOCAL_CORPUS_PATHS` (`src/speechify_run.py:54`):

| `--dataset` | Source | Obtain from | Expected path under `--data-root` |
|---|---|---|---|
| `interviewer` | Anthropic Interviewer | auto (`Anthropic/AnthropicInterviewer`) | — |
| `soda` | SODA | auto (`allenai/soda`) | — |
| `multiwoz` | MultiWOZ 2.2 | <https://github.com/budzianowski/multiwoz> | `multiwoz/data/MultiWOZ_2.2/train/`, `multiwoz/data/MultiWOZ_2.2/test/` |
| `negotiator` | CraigslistBargain | <https://stanfordnlp.github.io/cocoa/> | `CraigslistBargain/train.json`, `CraigslistBargain/test.json` |
| `socraticlm` | SocraticLM (SocraTeach) | <https://github.com/Ljyustc/SocraticLM> | `SocraticLM/data/SocraTeach_multi.json` |
| `persuader` | DailyPersuasion (PersuGPT) | see the PersuGPT release | pass the json directly via `--input_path` |

`--data-root` is required only for the three scenarios with a local source
(`multiwoz`, `negotiator`, `socraticlm`); omitting it for one of those raises a
message naming the flag. `persuader` ignores `--data-root` and **requires**
`--input_path`; it raises immediately without it. `socraticlm` also accepts
`--input_path` to override the default location.

`--dataset all` runs all six scenarios, so it needs both `--data-root` and
`--input_path`.

Other knobs: `--llm_model_name` and `--temperature` (0.7) control the
conversion call, `--interviewer_subset` (`workforce`) picks which Anthropic
Interviewer subset to read, and `--concise` switches to a prompt that shortens
the dialogue as it converts.

Source: `src/speechify_run.py`, `src/speechify_core.py`,
`src/speechify_datasets.py`, `src/speechify_prompts.py`.

Next: [Stage 2](stage2-slots.md).
