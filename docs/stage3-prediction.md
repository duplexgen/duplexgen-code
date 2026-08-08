# Stage 3: Turn-taking prediction

Trains or runs a turn-taking predictor over the slots Stage 2 identifies.

Two interchangeable predictors, both scored by the KL divergence between the
human annotation distribution and the predicted one. The **HF predictor** is a
LoRA token-classification head fine-tuned on the annotations; the **LLM
predictor** asks a chat model for verbalized probabilities zero-shot.

`--input_root` points at the unpacked annotation layout — convert the released
annotations first with `tools/unpack_corpus.py` (see [CORPUS.md](CORPUS.md)). It
has no default on any of these entry points.

## HF predictor

Training defaults to 4-bit QLoRA, so a 4B base model fits well under the 24GB
inference budget quoted in [Setup](../README.md#setup):

```bash
.venv/bin/python -m src.train_turntaking_hf \
  --model_name_or_path Qwen/Qwen3-4B \
  --input_root data-annotations/ \
  --output_dir ./output/turntaking_qwen3_4b
```

`--style` defaults to `soft_classification`, which trains against the full
human distribution over the three labels (KL loss) rather than its argmax.

Then score it, pointing `--peft_path` at the LoRA adapter the run saved:

```bash
.venv/bin/python -m src.inference_turntaking_hf \
  --model_name_or_path Qwen/Qwen3-4B \
  --peft_path ./output/turntaking_qwen3_4b \
  --input_root data-annotations/ --split test --scenarios all \
  --load_in_4bit \
  --output_path results/turntaking_hf_test.json
```

The final adapter lands at `--output_dir` itself; the `checkpoint-<step>/`
subdirectories underneath are the intermediate saves, and either path works as
`--peft_path`. `--model_name_or_path` must be the same base model the adapter
was trained on — the adapter carries no base weights. Training evaluates on
`--validation_split` (`test` by default), so unpack both splits first.

`--attn_implementation` defaults to `auto`: FlashAttention-2 when the optional
`flash-attn` package is importable, PyTorch SDPA otherwise. Pass
`--attn_implementation flash_attention_2` to require it.

## LLM predictor

The backend follows the model-name rule from [Setup](../README.md#setup).

```bash
# Proprietary
export OPENAI_API_KEY=sk-...
.venv/bin/python -m src.inference_turntaking_llm \
  --input_root data-annotations/ --split test --scenarios all \
  --model gpt-4.1-mini

# Self-hosted
.venv/bin/python -m src.inference_turntaking_llm \
  --input_root data-annotations/ --split test --scenarios all \
  --model Qwen/Qwen3-32B --base_url http://localhost:8000/v1
```

`--server` / `--server_port` compose a `--base_url` if you prefer to give the
host and port separately. A request that fails or comes back unparsable falls
back to a uniform distribution for that slot and is counted in `n_fallback`; a
run where *every* request failed aborts rather than reporting the KL of an
all-uniform prediction.

Source: `src/train_turntaking_hf.py`, `src/inference_turntaking_hf.py`,
`src/inference_turntaking_llm.py`, `src/boundary_annotator.py`, and
`src/swbd_parse.py` / `src/swbd_convert.py` (Switchboard preprocessing for the
generic human-human prior).

Next: [Stage 4](stage4-generation.md).
