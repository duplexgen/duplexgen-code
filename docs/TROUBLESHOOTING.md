# Troubleshooting

**`ImportError: cannot import name 'GenericForTokenClassification'`**
You installed Stage 5 into the Stage 1-4 environment. The vendored Chatterbox
package pins `transformers==4.46.3`; Stage 3 needs `>=4.53`. Use the two
environments described in [Setup](../README.md#setup).

**`ERROR: Package 'chatterbox-tts' requires a different Python`**
Stage 5 is Python 3.11 only. Create `.venv-tts` with `python3.11`.

**`[WARN] No input JSONs` / `Number of human annotations: 0`**
Your `--input_root` points at the released Hugging Face layout, which is not
what the stages read. Convert it first with `tools/unpack_corpus.py` — see
[CORPUS.md](CORPUS.md).

**`ImportError: FlashAttention2 has been toggled on, but it cannot be used`**
You passed `--attn_implementation flash_attention_2` without the optional
`flash-attn` package. Install it (`.venv/bin/pip install flash-attn
--no-build-isolation`, which compiles against your CUDA toolkit) or drop the
flag — the default `auto` falls back to PyTorch SDPA.

**`RuntimeError: All N requests failed` from `src.inference_turntaking_llm`**
Nothing answered at the endpoint, so every prediction would have been the
uniform fallback. Check that `--base_url` points at a running server, or that
`OPENAI_API_KEY` is exported if you passed an OpenAI model name. The message
quotes the first underlying error.

**`No input dialogues under <path> for split 'train'` from `src.synthesis.run`**
`--input_root` must be the *parent* of `text_dialogue_<dataset>/`, and Stage 4
reads the `dialogues/` half of the corpus, not `annotations/`. Unpack it with
`tools/unpack_corpus.py --kind dialogues`, whose default root is
`data-dialogues/`. Note the released `dialogues/` half is train-only.

**`ModuleNotFoundError: No module named 'src'`**
Run from the repository root, or `.venv/bin/pip install -e .`.

**Building `pynini` fails**
It needs OpenFst headers. `sudo apt-get install libfst-dev`, or install
`pynini` from conda-forge.

**`No input dialogues matched --input_glob`** from `tts_render/convert_spoken.py`
Stage 5 reads Stage 4's output, `<save_root>/text_dialogue_<dataset>/<split>/
*.json`. Quote the glob — an unquoted `**/*.json` is expanded by the shell
before Python sees it, and most shells do not expand `**` recursively by
default.

**`--num_variants N needs that many distinct user voices`**
The bundled `tts_render/librispeech_samples/` holds 12 speakers, and each
variant of a dialogue takes a different one. Lower `--num_variants` or point
`--librispeech_root` at a full LibriSpeech download.

**Rendered backchannels sound flat or mispronounced**
Expected, and not something a flag fixes. Chatterbox is weak on two-phoneme
interjections; the paper's dialogues used the ElevenLabs API for backchannels
instead. See the note in [Stage 5](stage5-tts.md) for what to replace.
