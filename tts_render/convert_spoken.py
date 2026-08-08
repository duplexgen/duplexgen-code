import os
import argparse
import shutil
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import random
import json
import re
import tempfile
import zlib
from pathlib import Path
from silero_vad import load_silero_vad, get_speech_timestamps
import whisperx
from glob import glob
import copy

from chatterbox.tts_turbo import ChatterboxTurboTTS

from bc_cache import BackchannelCache

import nltk

nltk.download("punkt")
nltk.download("punkt_tab")
from nltk import sent_tokenize

from nemo_text_processing.text_normalization.normalize import Normalizer

import logging

logging.getLogger("matplotlib").setLevel(logging.WARNING)

TARGET_SR = 24000  # overridden from model.sr after loading
PROMPT_SR = 16000

TURN_GAP_SEC = 0.16  # fixed silence gap between speaker turns (seconds)
USER_INTERRUPT_OVERLAP_SEC = 0.64  # max overlap when user interrupts assistant (seconds)
USER_INTERRUPT_PROB = 0.5  # probability that user interrupts assistant [reduced from 0.6]
MAX_PROMPT_SECS = 10  # max seconds of audio to keep in cumulative voice prompt

# LibriSpeech speaker behind prompt_wavs/assistant_en.wav (see PROVENANCE.md).
# Held out of the user voice pool so the assistant and a user variant can never
# share a voice.
ASSISTANT_SPEAKER_ID = "458"

DEFAULT_SPEAKERS = ["user", "assistant"]
DEFAULT_BC_CANDIDATES = ["yeah", "uh-huh", "mm-hmm", "right", "okay"]
DEFAULT_STYLE = "A speaker with normal speaking rate"
TAKE_FLOOR_TOKEN = "[TAKE_FLOOR]"

_word_re = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?")


def normalize_audio(waveform, ratio=1.0):
    max_val = torch.abs(waveform).max()
    normalized_waveform = waveform / max_val * ratio if max_val != 0 else waveform
    return normalized_waveform


def load_assistant_prompt_path(
    prompt_dir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt_wavs"),
):
    """Return the fixed assistant prompt audio file path (assistant_en.wav).

    A single held-out LibriSpeech train-clean-100 speaker, never drawn from the
    per-variant user voice pool. See prompt_wavs/PROVENANCE.md.
    """
    path = os.path.join(prompt_dir, "assistant_en.wav")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path


def list_librispeech_speakers(librispeech_root: str, subsets, exclude=None):
    """Return {speaker_id: [flac_path, ...]} for the requested LibriSpeech subsets.

    `exclude` holds speaker IDs that must never be drawn as a user voice --
    notably the assistant prompt's speaker, which would otherwise appear on
    both sides of a dialogue when running against the full corpus.
    """
    excluded = {str(s) for s in (exclude or ())}
    speakers = {}
    for subset in subsets:
        subset_dir = os.path.join(librispeech_root, subset)
        if not os.path.isdir(subset_dir):
            continue
        for spk in os.listdir(subset_dir):
            if spk in excluded:
                continue
            spk_dir = os.path.join(subset_dir, spk)
            if not os.path.isdir(spk_dir):
                continue
            flacs = glob(os.path.join(spk_dir, "*", "*.flac"))
            if flacs:
                speakers.setdefault(spk, []).extend(flacs)
    if not speakers:
        raise FileNotFoundError(
            f"No LibriSpeech speakers found under {librispeech_root} for subsets {subsets}"
        )
    return speakers


def sample_librispeech_prompt(
    speaker_pool: dict,
    rng: random.Random,
    min_duration_sec: float = 5.0,
    max_attempts_per_speaker: int = 8,
):
    """Sample one (speaker_id, flac_path) whose duration > min_duration_sec.

    Iterates random speakers; per speaker tries up to ``max_attempts_per_speaker``
    files. Falls back to scanning all files of a speaker if nothing fits.
    """
    speaker_ids = list(speaker_pool.keys())
    rng.shuffle(speaker_ids)
    for spk in speaker_ids:
        files = list(speaker_pool[spk])
        rng.shuffle(files)
        # quick attempts
        for f in files[:max_attempts_per_speaker]:
            try:
                info = torchaudio.info(f)
            except Exception:
                continue
            if info.num_frames / info.sample_rate > min_duration_sec:
                return spk, f
        # fall through and try every remaining file for this speaker
        for f in files[max_attempts_per_speaker:]:
            try:
                info = torchaudio.info(f)
            except Exception:
                continue
            if info.num_frames / info.sample_rate > min_duration_sec:
                return spk, f
    raise FileNotFoundError(
        f"No LibriSpeech file longer than {min_duration_sec}s found in pool of "
        f"{len(speaker_pool)} speakers"
    )


def sample_n_librispeech_prompts(
    speaker_pool: dict, n: int, rng: random.Random, min_duration_sec: float = 5.0
):
    """Sample ``n`` (speaker_id, flac_path) pairs with distinct speaker_ids when possible."""
    available = {spk: list(files) for spk, files in speaker_pool.items()}
    chosen = []
    while len(chosen) < n and available:
        spk_ids = list(available.keys())
        rng.shuffle(spk_ids)
        spk_id = spk_ids[0]
        files = available[spk_id]
        rng.shuffle(files)
        picked = None
        for f in files:
            try:
                info = torchaudio.info(f)
            except Exception:
                continue
            if info.num_frames / info.sample_rate > min_duration_sec:
                picked = f
                break
        del available[spk_id]
        if picked is not None:
            chosen.append((spk_id, picked))
    if len(chosen) < n:
        raise FileNotFoundError(f"Only found {len(chosen)} usable LibriSpeech speakers (need {n})")
    return chosen


# --- Schema conversion helpers (unchanged from original) ---


def _clean_text(s: str) -> str:
    return s.replace("[BACKCHANNEL]", "").replace("[TAKE_FLOOR]", "").strip()


def _split_backchannel_segments(text: str, specific_contents: list = None, role2spk: dict = None):
    parts = re.split(r"\[BACKCHANNEL\]", text)
    segments = []
    bc_counter = 0

    for i, seg in enumerate(parts):
        seg = seg.strip(" ,")
        if seg:
            segments.append((seg, None, None))

        if i < len(parts) - 1:
            if specific_contents and bc_counter < len(specific_contents):
                bc_text = specific_contents[bc_counter]
                bc_counter += 1
            else:
                bc_text = random.choice(DEFAULT_BC_CANDIDATES)

            bc_spk_override = role2spk.get("assistant")
            segments.append((bc_text, "backchannel", bc_spk_override))

    return segments


def _sent_split(text: str):
    try:
        sents = sent_tokenize(text)
        if not sents:
            return [text]
        return sents
    except Exception:
        chunks = re.split(r"(?<=[\.\\?\\!])\s+", text)
        chunks = [c for c in chunks if c]
        return chunks or [text]


def count_words_for_align(text: str, normalizer: Normalizer = None) -> int:
    t = text.replace("[BACKCHANNEL]", "").replace("[TAKE_FLOOR]", "")
    t = (
        t.replace("[interrupted]", "")
        .replace("[MASK1]", "")
        .replace("[MASK2]", "")
        .replace("[sigh]", "")
    )
    t = t.strip()
    if normalizer is not None:
        t = normalizer.normalize(t, verbose=False, punct_post_process=True)
    return len(_word_re.findall(t))


def _cut_at_take_floor_plus_one_word(text: str) -> str:
    if TAKE_FLOOR_TOKEN not in text:
        return text

    left, right = text.split(TAKE_FLOOR_TOKEN, 1)
    left = left.strip()
    right = right.strip()

    m = _word_re.search(right)
    if not m:
        return left

    first_word = m.group(0)
    if left:
        return f"{left} {first_word}".strip()
    return first_word.strip()


# --- LLM-artifact sanitizer -------------------------------------------------

_INTENTIONAL_BRACKET_TOKENS = {
    "[BACKCHANNEL]",
    "[TAKE_FLOOR]",
    "[MASK1]",
    "[MASK2]",
    "[interrupted]",
    "[sigh]",
}

_STAGE_DIRECTION_WORDS = {
    "sigh",
    "sighs",
    "sighing",
    "yawn",
    "yawns",
    "yawning",
    "laugh",
    "laughs",
    "laughing",
    "chuckle",
    "chuckles",
    "cough",
    "coughs",
    "clears throat",
    "clearing throat",
    "whistle",
    "whistles",
    "whistling",
    "wink",
    "winks",
    "wink wink",
    "smile",
    "smiles",
    "smiling",
    "grin",
    "grins",
    "pause",
    "pauses",
    "hands receipt",
    "three times",
}

_NUM_ROLE_RX = re.compile(r"^\s*\d+\)\s*(user|assistant|system)\s*:\s*", re.IGNORECASE)
_NUM_ROLE_ANY_RX = re.compile(r"\d+\)\s*(user|assistant|system)\s*:", re.IGNORECASE)
_BARE_ROLE_RX = re.compile(r"^\s*(user|assistant|system)\s*:\s*", re.IGNORECASE)
_MD_BOLD_RX = re.compile(r"\*\*[^*\n]{1,200}\*\*")
_AST_TURN_MARKER_RX = re.compile(
    r"\*[^*\n]*?(?:next\s+turn|user\s*:|assistant\s*:|user\s*'?s?\s+next|assistant\s*'?s?\s+next)[^*\n]*\*",
    re.IGNORECASE,
)
_STAR_PHRASE_RX = re.compile(r"\*([^*\n]{1,60})\*")
_BRACKET_RX = re.compile(r"\[([^\[\]\n]{1,60})\]")


def _is_template_placeholder(token: str) -> bool:
    """token includes the surrounding [brackets]. [User Turn]-style script markers are
    soft-stripped instead, so they should NOT be classified as drop-worthy here."""
    if token in _INTENTIONAL_BRACKET_TOKENS:
        return False
    inner = token[1:-1]
    if inner.upper() == "TAKING_FLOOR":
        return False
    low = inner.lower()
    # Script/stage markers handled by sanitize_text (soft strip), not drop
    if low in {"user turn", "user's turn", "assistant turn", "assistant's turn"}:
        return False
    if low in {"email protected", "email-protected"}:
        return True
    if re.match(
        r"^(?:your|some|city|state|country|old|new|insert|enter|placeholder)\b", inner, re.I
    ):
        return True
    if re.match(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}$", inner):
        return True
    return False


def should_drop_dialogue(d: dict) -> tuple:
    """Return (drop, reason). Inspect every utterance's content + nested full_content."""
    for hi, msg in enumerate(d.get("history", [])):
        role = (msg.get("role") or "").lower()
        candidates = []
        outer = msg.get("content")
        if isinstance(outer, str):
            candidates.append(outer)
        for sub in msg.get("history", []):
            fc = sub.get("full_content") if isinstance(sub, dict) else None
            if isinstance(fc, str):
                candidates.append(fc)

        for s in candidates:
            if len(_NUM_ROLE_ANY_RX.findall(s)) >= 2:
                return True, f"msg{hi}: multi-turn concat"
            if re.search(r"</?think\b", s, re.IGNORECASE):
                return True, f"msg{hi}: think-block leak"
            if len(s) > 1500:
                return True, f"msg{hi}: very-long ({len(s)} chars)"
            if _MD_BOLD_RX.search(s):
                return True, f"msg{hi}: markdown bold leak"
            if _AST_TURN_MARKER_RX.search(s):
                return True, f"msg{hi}: asterisk turn marker"
            for m in _BRACKET_RX.finditer(s):
                tok = m.group(0)
                if _is_template_placeholder(tok):
                    return True, f"msg{hi}: template placeholder {tok}"
            m = _NUM_ROLE_RX.match(s) or _BARE_ROLE_RX.match(s)
            if m and role:
                embedded = m.group(1).lower()
                if embedded != role:
                    return True, f"msg{hi}: role-mismatch outer={role} inline={embedded}"
    return False, ""


def _strip_stage_directions(s: str) -> str:
    def _repl(m):
        inner = m.group(1).strip(" ,").lower()
        if inner in _STAGE_DIRECTION_WORDS:
            return ""
        # Otherwise treat as emphasis (e.g. *really*, *$1,520*): strip the asterisks, keep the word
        return m.group(1)

    return _STAR_PHRASE_RX.sub(_repl, s)


def _peel_role_prefix(s: str) -> str:
    prev = None
    while prev != s:
        prev = s
        s = _NUM_ROLE_RX.sub("", s)
        s = _BARE_ROLE_RX.sub("", s)
    return s


def sanitize_text(s: str) -> str:
    if not isinstance(s, str):
        return s
    s = re.sub(r"\[TAKING_FLOOR\]", "[TAKE_FLOOR]", s, flags=re.IGNORECASE)
    s = re.sub(r"\[User Turn\]|\[User turn\]|\[Assistant Turn\]|\[Assistant turn\]", "", s)
    s = _peel_role_prefix(s)
    s = _strip_stage_directions(s)
    return re.sub(r"\s{2,}", " ", s).strip()


def reconstruct_text_with_tokens(msg_data: dict) -> str:
    if "history" not in msg_data or not msg_data["history"]:
        return msg_data.get("content", "")

    full_content = msg_data["history"][0].get("full_content", "")
    decisions = [item for item in msg_data["history"] if "word_index" in item]

    words = full_content.split()

    for d in sorted(decisions, key=lambda x: x["word_index"], reverse=True):
        token = d.get("inserted_token")
        if token:
            idx = min(d["word_index"] + 1, len(words))
            words.insert(idx, token)

    return " ".join(words)


def convert_original_to_expected(original: dict) -> dict:
    history = original.get("history", [])
    speakers = DEFAULT_SPEAKERS[:]
    role2spk = {"user": speakers[0], "assistant": speakers[1]}

    utterances = []
    triggers = []

    for msg in history:
        role = msg.get("role")
        if role not in role2spk:
            continue

        msg_role_spk = role2spk[role]

        text = reconstruct_text_with_tokens(msg)
        text = sanitize_text(text)
        if not text.strip():
            continue

        bc_contents_list = []
        if "history" in msg:
            decisions = [h for h in msg["history"] if "word_index" in h]
            decisions.sort(key=lambda x: x["word_index"])
            for d in decisions:
                if d.get("decision") == "backchannel" or d.get("inserted_token") == "[BACKCHANNEL]":
                    bc_content = d.get("content", random.choice(DEFAULT_BC_CANDIDATES))
                    bc_contents_list.append(bc_content)

        take_floor = TAKE_FLOOR_TOKEN in text
        if take_floor:
            text = _cut_at_take_floor_plus_one_word(text)

        segments = _split_backchannel_segments(
            text, specific_contents=bc_contents_list, role2spk=role2spk
        )

        for seg_idx, (seg_text_raw, bc_type, spk_override) in enumerate(segments):
            seg_text = _clean_text(seg_text_raw)
            if not seg_text and bc_type is None:
                continue

            uttr_type = "backchannel" if bc_type == "backchannel" else None
            current_spk = spk_override if spk_override else msg_role_spk

            item = {
                "speaker": current_spk,
                "uttr_type": uttr_type,
                "texts": seg_text,
                "split_texts": _sent_split(seg_text) if uttr_type is None else [seg_text],
            }

            if take_floor and uttr_type is None:
                is_last_non_bc = all(s[1] is not None for s in segments[seg_idx + 1 :])
                if is_last_non_bc and item["split_texts"]:
                    other_spk = speakers[1] if msg_role_spk == speakers[0] else speakers[0]
                    triggers.append((other_spk, len(utterances) + 1))

            utterances.append(item)

    utterances_with_bc = []
    for ut in utterances:
        utterances_with_bc.append(
            {
                "speaker": ut["speaker"],
                "uttr_type": ut["uttr_type"],
                "texts": ut["texts"],
                "texts_styled": [{"text": ut["texts"], "style": DEFAULT_STYLE}],
            }
        )

    for target_spk, start_idx in triggers:
        if start_idx < len(utterances_with_bc):
            if utterances_with_bc[start_idx]["speaker"] == target_spk:
                utterances_with_bc[start_idx]["uttr_type"] = "interrupt"

    return {
        "num_turns": len(utterances_with_bc),
        "speakers": speakers,
        "utterances": utterances,
        "utterances_with_bc": utterances_with_bc,
    }


def generate_delay(pad_size=None, mode=None, behavior=None):
    if mode is None:
        if behavior is None:
            delay = np.random.normal(loc=0.38, scale=0.2)
        elif behavior == 0:
            delay = 0
        elif behavior == 1:
            delay = np.random.normal(loc=0.2, scale=0.01)
        elif behavior == 2:
            delay = np.random.normal(loc=0.8, scale=0.2)
    elif mode == "backchannel_mhm":
        delay = np.random.normal(loc=0.13, scale=0.02)
    elif mode == "backchannel":
        delay = 0.0

    # max(0, pad_size): a negative pad_size means the backchannel is longer than the
    # gap it was anchored to. np.clip(x, 0, negative) returns the negative bound, so
    # this used to hand back a *negative* delay, which the caller added to a sample
    # index -- placing the backchannel before the start of the track.
    delay = int(np.clip(delay * TARGET_SR, 0, max(0, pad_size)))
    return delay


def place_backchannel(listener_speech, tts_speech, bc_speech, start):
    """Write `bc_speech` into the listener track at `start`, growing both tracks to fit.

    `start` is derived from a forced alignment of the host turn and can land past the
    end of it when a backchannel is longer than the words it was anchored to. Slicing
    alone does not survive that: a start beyond the tensor yields a zero-length
    destination and raises `The expanded size of the tensor (0) must match ...`, which
    used to kill the whole variant. Padding keeps the backchannel intact, and both
    tracks are padded together because they are stacked into one stereo tensor
    immediately after this.
    """
    start = max(0, start)
    overrun = start + bc_speech.size(1) - listener_speech.size(1)
    if overrun > 0:
        listener_speech = F.pad(listener_speech, (0, overrun, 0, 0))
        tts_speech = F.pad(tts_speech, (0, overrun, 0, 0))
    listener_speech[:, start : start + bc_speech.size(1)] = bc_speech
    return listener_speech, tts_speech


def aggregate_speech(total_speech, total_speech_meta):
    merged_speech = total_speech[0].clone()
    gap_samples = int(TURN_GAP_SEC * TARGET_SR)
    overlap_samples_max = int(USER_INTERRUPT_OVERLAP_SEC * TARGET_SR)

    for i, (speech, meta) in enumerate(zip(total_speech, total_speech_meta)):
        if i == 0:
            continue
        prev_meta = total_speech_meta[i - 1]

        if meta["speaker"] != prev_meta["speaker"] and meta["uttr_type"] == "interrupt":
            overlap_len = np.random.normal(loc=0.45, scale=0.05)
            overlap_len = int(
                np.clip(overlap_len * TARGET_SR, 0, min(merged_speech.size(1), speech.size(1)))
            )
            overlap_len = round(overlap_len)

            merged_speech = torch.cat(
                (
                    merged_speech[:, :-overlap_len],
                    merged_speech[:, -overlap_len:] + speech[:, :overlap_len],
                    speech[:, overlap_len:],
                ),
                dim=1,
            )

        elif meta["speaker"] != prev_meta["speaker"]:
            prev_is_assistant = prev_meta["speaker"] == 1
            curr_is_user = meta["speaker"] == 0

            if prev_is_assistant and curr_is_user and random.random() < USER_INTERRUPT_PROB:
                BC_CHECK_SAMPLES = int(2.0 * TARGET_SR)
                ai_lead = speech[1, : min(BC_CHECK_SAMPLES, speech.size(1))]
                has_leading_bc = ai_lead.abs().max().item() > 1e-6

                if has_leading_bc:
                    padded_speech = F.pad(speech, (gap_samples, 0, 0, 0))
                    merged_speech = torch.cat((merged_speech, padded_speech), dim=1)
                else:
                    overlap_samples = min(
                        overlap_samples_max, merged_speech.size(1), speech.size(1)
                    )
                    merged_speech = torch.cat(
                        (
                            merged_speech[:, :-overlap_samples],
                            merged_speech[:, -overlap_samples:] + speech[:, :overlap_samples],
                            speech[:, overlap_samples:],
                        ),
                        dim=1,
                    )
            else:
                padded_speech = F.pad(speech, (gap_samples, 0, 0, 0))
                merged_speech = torch.cat((merged_speech, padded_speech), dim=1)

        else:
            merged_speech = torch.cat((merged_speech, speech), dim=1)

    return merged_speech


def generate_audio(model, tts_text, audio_prompt_path):
    """Generate speech with ChatterboxTurboTTS. Returns [1, T] tensor at TARGET_SR."""
    wav = model.generate(tts_text, audio_prompt_path=audio_prompt_path)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    return wav


def _save_cumulative_prompt(audio: torch.Tensor, path: str):
    """Trim to MAX_PROMPT_SECS and save to path for use as next voice prompt."""
    max_samples = MAX_PROMPT_SECS * TARGET_SR
    trimmed = audio[:, -max_samples:] if audio.size(1) > max_samples else audio
    torchaudio.save(path, trimmed, TARGET_SR)


def main_process(
    model, vad_model, align, args, index, origin_dialogue, prompt_paths, tmpdir, normalizer=None
):
    model_a = align["model_a"]
    metadata = align["metadata"]

    dialogue = copy.deepcopy(origin_dialogue)

    total_speech = []
    total_speech_meta = []

    init_spk = dialogue["utterances_with_bc"][0]["speaker"]
    spk_offset = 1 if init_spk == DEFAULT_SPEAKERS[1] else 0

    # Cumulative prompt paths: start as copies of the initial prompt files
    cumulative_prompt_paths = [os.path.join(tmpdir, f"cumulative_{i}.wav") for i in range(2)]
    for i, p in enumerate(prompt_paths):
        shutil.copyfile(p, cumulative_prompt_paths[i])
        os.chmod(cumulative_prompt_paths[i], 0o644)

    # Keep cumulative audio tensors for trimming/updating
    cumulative_audio = []
    for p in prompt_paths:
        wav, sr = torchaudio.load(p)
        if sr != TARGET_SR:
            wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)(wav)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        cumulative_audio.append(wav)

    bc_cache = BackchannelCache()
    bc_queue = [[], []]
    uttered_bc_list = []
    backchannel_list = []
    tts_texts = ["", ""]
    prev_uttr_type = "first"
    interrupt_flag = [False, False]
    accumulated_flag = [False, False]

    modified_utterances = dialogue["utterances_with_bc"]

    for turn, utterance in enumerate(dialogue["utterances_with_bc"]):
        curr_idx = (turn + spk_offset) % 2
        other_idx = 1 - curr_idx

        if turn == len(dialogue["utterances_with_bc"]) - 1:
            next_uttr_type = "last"
        else:
            next_uttr_type = dialogue["utterances_with_bc"][turn + 1]["uttr_type"]

        for text_idx, texts_styled in enumerate(utterance["texts_styled"]):
            cleaned_tts_text = (
                texts_styled["text"]
                .replace("[interrupted]", "")
                .replace("[MASK1]", "")
                .replace("[MASK2]", "")
                .replace("[sigh]", "")
                .strip()
            )

            is_last_text = text_idx == len(utterance["texts_styled"]) - 1

            if utterance["uttr_type"] == "interrupt" and text_idx == 0:
                interrupt_flag[curr_idx] = True

            if not cleaned_tts_text or cleaned_tts_text in ["-", "...", ".", ","]:
                modified_utterances[turn]["texts_styled"][text_idx]["isUttered"] = False
                if text_idx == 0 and prev_uttr_type == "backchannel":
                    modified_utterances[bc_queue[other_idx][-1]["turn"]]["texts_styled"][
                        bc_queue[other_idx][-1]["text_idx"]
                    ]["isUttered"] = False
                    bc_queue[other_idx] = bc_queue[other_idx][:-1]
                if next_uttr_type == "backchannel" or not is_last_text:
                    continue
                if not accumulated_flag[curr_idx]:
                    continue

            elif prev_uttr_type == "backchannel" and text_idx == 0:
                tts_texts[curr_idx] += " " + texts_styled["text"]
            else:
                tts_texts[curr_idx] = texts_styled["text"]

            if next_uttr_type == "backchannel" and is_last_text:
                accumulated_flag[curr_idx] = True
                continue

            else:
                accumulated_flag[curr_idx] = False
                tts_text = tts_texts[curr_idx]
                tts_text = (
                    tts_text.replace("[interrupted]", "")
                    .replace("[MASK1]", "")
                    .replace("[MASK2]", "")
                    .strip()
                )

                sentences = sent_tokenize(tts_text)
                generated_speech_list = []

                for st_idx, sentence_ in enumerate(sentences):
                    sentence_clean = sentence_.replace("-", " ")

                    if utterance["uttr_type"] == "backchannel":
                        bc_sentence = sentence_clean
                        if bc_sentence.strip().lower() == "yeah":
                            bc_sentence = bc_sentence.rstrip() + "?"
                        speech_ = bc_cache.get(curr_idx, sentence_)
                        if speech_ is None:
                            speech_ = generate_audio(
                                model, bc_sentence, cumulative_prompt_paths[curr_idx]
                            )
                            bc_cache.put(curr_idx, sentence_, speech_)
                        else:
                            speech_ = speech_.clone()
                    else:
                        speech_ = generate_audio(
                            model, sentence_clean, cumulative_prompt_paths[curr_idx]
                        )

                    # VAD trim between sentences (not the last one)
                    if st_idx != len(sentences) - 1:
                        speech_16k = torchaudio.transforms.Resample(
                            orig_freq=TARGET_SR, new_freq=PROMPT_SR
                        )(speech_)
                        speech_timestamps = get_speech_timestamps(
                            speech_16k, vad_model, threshold=0.3, sampling_rate=PROMPT_SR
                        )
                        try:
                            speech_end_idx = int(
                                speech_timestamps[-1]["end"] * TARGET_SR / PROMPT_SR
                            )
                            speech_ = speech_[:, :speech_end_idx]
                        except Exception:
                            print(f"VAD failed for sentence: {sentence_}. Using full audio.")

                    volume_ratio = 0.5 if utterance["uttr_type"] == "backchannel" else 1.0
                    speech_ = normalize_audio(speech_, ratio=volume_ratio)

                    # Update cumulative voice prompt with non-backchannel audio
                    if utterance["uttr_type"] != "backchannel":
                        cumulative_audio[curr_idx] = torch.cat(
                            (cumulative_audio[curr_idx], speech_), dim=1
                        )
                        _save_cumulative_prompt(
                            cumulative_audio[curr_idx], cumulative_prompt_paths[curr_idx]
                        )

                    generated_speech_list.append(speech_)

                tts_speech = torch.cat(generated_speech_list, dim=1)

                # VAD on full utterance
                tts_speech_16k = torchaudio.transforms.Resample(
                    orig_freq=TARGET_SR, new_freq=PROMPT_SR
                )(tts_speech)
                speech_timestamps = get_speech_timestamps(
                    tts_speech_16k, vad_model, threshold=0.3, sampling_rate=PROMPT_SR
                )

                if len(speech_timestamps) > 0:
                    if utterance["uttr_type"] == "backchannel":
                        speech_start_idx = int(
                            speech_timestamps[0]["start"] * TARGET_SR / PROMPT_SR
                        )
                    elif text_idx > 0 or not total_speech:
                        speech_start_idx = 0
                    else:
                        speech_start_idx = int(
                            speech_timestamps[0]["start"] * TARGET_SR / PROMPT_SR
                        )

                    if utterance["uttr_type"] == "backchannel":
                        speech_end_idx = tts_speech.size(1)
                    elif next_uttr_type == "interrupt" and is_last_text:
                        speech_end_idx = int(speech_timestamps[-1]["end"] * TARGET_SR / PROMPT_SR)
                    elif not is_last_text or next_uttr_type == "last":
                        speech_end_idx = tts_speech.size(1)
                    else:
                        speech_end_idx = int(speech_timestamps[-1]["end"] * TARGET_SR / PROMPT_SR)

                    # Trim trailing silence from cumulative prompt
                    if tts_speech.size(1) != speech_end_idx:
                        sil_len = tts_speech.size(1) - speech_end_idx
                        cumulative_audio[curr_idx] = cumulative_audio[curr_idx][:, :-sil_len]
                        _save_cumulative_prompt(
                            cumulative_audio[curr_idx], cumulative_prompt_paths[curr_idx]
                        )

                    tts_speech = tts_speech[:, speech_start_idx:speech_end_idx]

                tts_texts[curr_idx] = ""

                if is_last_text:
                    if next_uttr_type is None or next_uttr_type == "interrupt":
                        # Reset cumulative prompt to initial
                        shutil.copyfile(prompt_paths[curr_idx], cumulative_prompt_paths[curr_idx])
                        os.chmod(cumulative_prompt_paths[curr_idx], 0o644)
                        wav, sr = torchaudio.load(prompt_paths[curr_idx])
                        if sr != TARGET_SR:
                            wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)(
                                wav
                            )
                        if wav.shape[0] > 1:
                            wav = wav.mean(dim=0, keepdim=True)
                        cumulative_audio[curr_idx] = wav

            if utterance["uttr_type"] == "backchannel":
                host_text = tts_texts[other_idx]
                wc = count_words_for_align(host_text, normalizer)

                bc_queue[curr_idx].append(
                    {
                        "speech": tts_speech.clone(),
                        "word_count": wc,
                        "turn": turn,
                        "text_idx": text_idx,
                        "text": tts_text,
                    }
                )

            else:
                listener_speech = torch.zeros_like(tts_speech)

                if bc_queue[other_idx]:
                    if normalizer is not None:
                        tts_text_for_align = normalizer.normalize(
                            tts_text, verbose=False, punct_post_process=True
                        )
                    else:
                        tts_text_for_align = tts_text
                    segments = [
                        {
                            "text": " " + tts_text_for_align.strip(),
                            "start": 0.0,
                            "end": tts_speech.size(1) / TARGET_SR,
                        }
                    ]
                    tts_speech_16k = torchaudio.transforms.Resample(
                        orig_freq=TARGET_SR, new_freq=PROMPT_SR
                    )(tts_speech)

                    align_results = whisperx.align(
                        segments,
                        model_a,
                        metadata,
                        tts_speech_16k,
                        "cuda",
                        return_char_alignments=False,
                    )
                    all_words = []
                    for segment in align_results["segments"]:
                        all_words.extend(segment["words"])
                    all_words = [w for w in all_words if "end" in w]

                if bc_queue[other_idx] and not all_words:
                    # No word survived alignment, so there is nothing to anchor the
                    # backchannels to. Drop them rather than guess a position; they stay
                    # isUttered=False so meta.json does not claim audio that is not there.
                    logging.warning(
                        f"Alignment produced no words for turn {turn}; dropping "
                        f"{len(bc_queue[other_idx])} backchannel(s)."
                    )
                    bc_queue[other_idx] = []

                if bc_queue[other_idx]:
                    bc_pos = []
                    for bc in bc_queue[other_idx]:
                        # Anchor after the word the backchannel was placed at in Stage 4,
                        # clamped into the words the aligner actually returned.
                        word_idx = max(0, min(bc["word_count"] - 1, len(all_words) - 2))
                        bc_pos.append(round(all_words[word_idx]["end"] * TARGET_SR))
                    bc_pos.append(tts_speech.size(1))

                    uttered_bc_list = []
                    # Backchannels come from one listener, so they cannot overlap each
                    # other: `cursor` pushes a backchannel past the previous one when the
                    # alignment anchors them closer together than they are long. Without
                    # it the later write silently overwrote the tail of the earlier one,
                    # which meta.json still reported as uttered in full.
                    cursor = 0
                    for bc_idx, bc in enumerate(bc_queue[other_idx]):
                        bc_speech = bc["speech"]
                        start = max(bc_pos[bc_idx], cursor)
                        pad_size = bc_pos[bc_idx + 1] - start - bc_speech.size(1)
                        start += generate_delay(pad_size, mode="backchannel")

                        # The last backchannel of a turn may have to extend the host turn
                        # to fit. That is fine unless the next utterance interrupts this
                        # one, where the extension would run into the interrupter -- there
                        # the backchannel is dropped instead.
                        if (
                            pad_size <= 0
                            and bc_idx == len(bc_queue[other_idx]) - 1
                            and text_idx == len(utterance["texts_styled"]) - 1
                            and next_uttr_type == "interrupt"
                        ):
                            continue

                        listener_speech, tts_speech = place_backchannel(
                            listener_speech, tts_speech, bc_speech, start
                        )
                        cursor = start + bc_speech.size(1)

                        modified_utterances[bc["turn"]]["texts_styled"][bc["text_idx"]][
                            "isUttered"
                        ] = True

                        uttered_bc_list.append({"speech": bc_speech, "text": bc["text"]})

                if curr_idx == 0:
                    speech = [tts_speech, listener_speech]
                else:
                    speech = [listener_speech, tts_speech]

                if bc_queue[other_idx]:
                    bc_queue[other_idx] = []

                speech = torch.cat(speech, dim=0)

                if interrupt_flag[curr_idx]:
                    total_speech.append(speech.clone())
                    total_speech_meta.append(
                        {
                            "uttr_type": "interrupt",
                            "speaker": curr_idx,
                            "tts_text": tts_text.strip(),
                        }
                    )
                    interrupt_flag[curr_idx] = False
                else:
                    total_speech.append(speech)
                    total_speech_meta.append(
                        {"uttr_type": None, "speaker": curr_idx, "tts_text": tts_text.strip()}
                    )

                if uttered_bc_list:
                    total_speech_meta[-1]["backchannels"] = []
                    for bc_idx, bc in enumerate(uttered_bc_list):
                        total_speech_meta[-1]["backchannels"].append(
                            {"idx": bc_idx, "tts_text": bc["text"]}
                        )
                        backchannel_list.append(bc["speech"].clone())
                    uttered_bc_list = []

        prev_uttr_type = utterance["uttr_type"]

    merged_speech = aggregate_speech(total_speech, total_speech_meta)

    dialogue["utterances_with_bc"] = modified_utterances
    dialogue["speech_meta"] = total_speech_meta

    return merged_speech, total_speech, backchannel_list, dialogue


def main(args):
    global TARGET_SR

    # force=True: importing whisperx/numba/nemo installs root handlers of their
    # own, and without it basicConfig() is a silent no-op -- which used to leave
    # the run at DEBUG, burying the per-dialogue progress under megabytes of
    # numba IR dumps and urllib3 chatter.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True
    )

    # Resolve every input before touching a GPU: loading Chatterbox, whisperx and the
    # NeMo grammars costs a couple of minutes, and a mistyped --input_glob or a missing
    # prompt wav should not cost that before it is reported.
    assistant_prompt_path = load_assistant_prompt_path(args.prompt_dir)

    # Build LibriSpeech speaker pool once
    libri_subsets = [s.strip() for s in args.librispeech_subsets.split(",") if s.strip()]
    speaker_pool = list_librispeech_speakers(
        args.librispeech_root, libri_subsets, exclude={ASSISTANT_SPEAKER_ID}
    )
    logging.info(f"LibriSpeech pool: {len(speaker_pool)} speakers from {libri_subsets}")

    if len(speaker_pool) < args.num_variants:
        raise SystemExit(
            f"--num_variants {args.num_variants} needs that many distinct user voices, but "
            f"{args.librispeech_root} yields only {len(speaker_pool)} speakers for subsets "
            f"{libri_subsets}. Lower --num_variants, or point --librispeech_root at a full "
            "LibriSpeech download (https://www.openslr.org/12) and widen "
            "--librispeech_subsets. The bundled sample holds 12 speakers."
        )

    seen = set()
    json_files = []
    for g in args.input_glob:
        for p in sorted(glob(g, recursive=True)):
            if p not in seen:
                seen.add(p)
                json_files.append(p)
    json_files.sort()

    if not json_files:
        raise SystemExit(
            f"No input dialogues matched --input_glob {args.input_glob}. Stage 5 reads the "
            "Stage 4 output layout, <save_root>/text_dialogue_<dataset>/<split>/*.json -- e.g. "
            "--input_glob 'outputs/generated_dialogues_with_hf_swbd_plus_backchannels/**/*.json'. "
            "Quote the glob so the shell does not expand it, and note '**' needs the two-level "
            "scenario/split path underneath the root."
        )

    if args.exclude_ids_file:
        excl_path = Path(args.exclude_ids_file)
        excluded = {ln.strip() for ln in excl_path.read_text().splitlines() if ln.strip()}
        before = len(json_files)
        json_files = [p for p in json_files if Path(p).stem not in excluded]
        logging.info(
            f"exclude_ids_file={excl_path}: {len(excluded)} ids; "
            f"{before} -> {len(json_files)} files"
        )

    # Optional shard partitioning so multiple workers can run in parallel safely
    if args.num_shards > 1:
        json_files = [p for i, p in enumerate(json_files) if i % args.num_shards == args.shard_id]
        logging.info(f"Shard {args.shard_id}/{args.num_shards}: {len(json_files)} files")

    if args.max_dialogues > 0:
        json_files = json_files[: args.max_dialogues]
        logging.info(f"max_dialogues={args.max_dialogues} -> {len(json_files)} files")

    logging.info(f"Found {len(json_files)} files to process.")

    # Patch Perth watermarker if the native library fails to initialize
    import perth

    try:
        perth.PerthImplicitWatermarker()
    except TypeError:

        class _NoOpWatermarker:
            def apply_watermark(self, audio, sample_rate):
                return audio

        perth.PerthImplicitWatermarker = _NoOpWatermarker

    # Load models
    model = ChatterboxTurboTTS.from_pretrained(device="cuda")
    TARGET_SR = model.sr

    vad_model = load_silero_vad()
    model_a, metadata = whisperx.load_align_model(language_code="en", device="cuda")
    align = {"model_a": model_a, "metadata": metadata}

    normalizer = Normalizer(input_case="cased", lang="en")

    n_rendered = n_skipped = n_dropped = n_failed = 0

    for fpath_str in json_files:
        fpath = Path(fpath_str)
        logging.info(f"Processing: {fpath}")

        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        drop, drop_reason = should_drop_dialogue(data)
        if drop:
            n_dropped += 1
            logging.warning(f"DROP-DIALOGUE: {fpath.name} ({drop_reason})")
            continue

        dialogue_data = convert_original_to_expected(data)

        # Per-dialogue deterministic RNG so the same (input, seed) reproduces voice choices.
        # crc32, not hash(): PYTHONHASHSEED randomizes str hashing per process, so hash()
        # gave a different voice cast on every run -- and a resumed run that skipped var00
        # would then fill var01.. from a different draw than the one var00 came from.
        dialogue_seed = args.seed + zlib.crc32(fpath.name.encode("utf-8"))
        dialogue_rng = random.Random(dialogue_seed)

        # Sample N distinct LibriSpeech speakers up-front
        try:
            voice_picks = sample_n_librispeech_prompts(
                speaker_pool, args.num_variants, dialogue_rng, min_duration_sec=5.0
            )
        except FileNotFoundError as e:
            logging.warning(f"Skip (voice sampling failed): {fpath} ({e})")
            continue

        dialogue_id = fpath.stem
        rel_dialogue_dir = Path(*fpath.parts[-3:-1]) / dialogue_id  # text_dialogue_X/train/<id>

        for variant_idx, (libri_spk, libri_path) in enumerate(voice_picks):
            variant_dir = args.save_dir / rel_dialogue_dir / f"var{variant_idx:02d}"
            save_audio_fpath = variant_dir / "dialogues" / "dialogue.wav"
            save_json_fpath = variant_dir / "meta.json"

            if save_audio_fpath.exists() and save_json_fpath.exists():
                n_skipped += 1
                logging.info(f"Skip (already exists): {variant_dir}")
                continue

            prompt_paths = [libri_path, assistant_prompt_path]
            logging.info(f"  var{variant_idx:02d}: speaker={libri_spk} prompt={libri_path}")

            # Reset Python/torch RNGs each variant so timing/sampling stays consistent.
            # dialogue_rng above chose the voice cast and is deliberately not reused here:
            # every dialogue's var<NN> draws the same TTS/timing randomness.
            seed_var = args.seed + variant_idx
            random.seed(seed_var)
            np.random.seed(seed_var)
            torch.manual_seed(seed_var)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed_var)

            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    merged_speech, total_speech, backchannel_list, total_speech_meta = main_process(
                        model,
                        vad_model,
                        align,
                        args,
                        0,
                        dialogue_data,
                        prompt_paths,
                        tmpdir,
                        normalizer,
                    )
            except Exception as e:
                n_failed += 1
                logging.exception(f"FAILED dialogue={dialogue_id} variant={variant_idx} ({e})")
                continue

            # Swap stereo channels so assistant lands at channel 0 (for personaplex-finetune).
            merged_speech = merged_speech.flip(dims=[0])
            total_speech = [s.flip(dims=[0]) for s in total_speech]
            total_speech_meta["speakers"] = ["assistant", "user"]
            for sm in total_speech_meta.get("speech_meta", []):
                spk = sm.get("speaker")
                if spk in (0, 1):
                    sm["speaker"] = 1 - spk

            save_audio_dir = variant_dir / "dialogues"
            save_audio_dir.mkdir(parents=True, exist_ok=True)
            torchaudio.save(str(save_audio_fpath), merged_speech, TARGET_SR)

            bc_cnt = 0
            for idx, speech in enumerate(total_speech):
                save_utterance_fpath = variant_dir / "utterances" / f"{idx:02d}.wav"
                save_utterance_fpath.parent.mkdir(parents=True, exist_ok=True)
                torchaudio.save(str(save_utterance_fpath), speech, TARGET_SR)

                if "backchannels" in total_speech_meta["speech_meta"][idx].keys():
                    for bc_idx, _bc_meta in enumerate(
                        total_speech_meta["speech_meta"][idx]["backchannels"]
                    ):
                        save_bc_fpath = variant_dir / "backchannels" / f"{idx:02d}_{bc_idx}.wav"
                        save_bc_fpath.parent.mkdir(parents=True, exist_ok=True)
                        torchaudio.save(str(save_bc_fpath), backchannel_list[bc_cnt], TARGET_SR)
                        bc_cnt += 1

            total_speech_meta["user_prompt_wav"] = str(libri_path)
            total_speech_meta["user_prompt_speaker_id"] = libri_spk
            total_speech_meta["variant_idx"] = variant_idx
            total_speech_meta["turn_gap_sec"] = TURN_GAP_SEC

            with open(save_json_fpath, "w") as jf:
                json.dump(total_speech_meta, jf, indent=4)

            n_rendered += 1

    logging.info(
        f"Done: {n_rendered} variants rendered, {n_skipped} already present, "
        f"{n_dropped} dialogues dropped by the sanitizer, {n_failed} variants failed."
    )
    if n_rendered == 0 and n_skipped == 0:
        raise SystemExit(
            f"Rendered nothing from {len(json_files)} input dialogues "
            f"({n_dropped} dropped by should_drop_dialogue, {n_failed} failed). "
            "Exiting non-zero rather than reporting success on an empty render."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt_dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt_wavs"),
        help="Directory containing assistant_en.wav (assistant prompt)",
    )
    parser.add_argument(
        "--voices_dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_wavs"),
        help="[Deprecated] kept for backwards compat; user voices now come from LibriSpeech.",
    )
    # Defaults to the tiny 12-speaker sample shipped in tts_render/librispeech_samples/
    # so this runs out of the box. For paper-scale rendering, download the full
    # corpus from https://www.openslr.org/12 and point --librispeech_root at it
    # (e.g. --librispeech_root /path/to/LibriSpeech --librispeech_subsets train-clean-100,train-clean-360).
    parser.add_argument(
        "--librispeech_root",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "librispeech_samples"),
        help="Root containing LibriSpeech subset directories (train-clean-100, etc.). "
        "Defaults to the bundled sample set; download the full corpus from "
        "https://www.openslr.org/12 for paper-scale rendering.",
    )
    parser.add_argument(
        "--librispeech_subsets",
        default="train-clean-100",
        help="Comma-separated LibriSpeech subsets to sample user voices from. "
        "Use train-clean-100,train-clean-360 with the full corpus.",
    )
    parser.add_argument(
        "--input_glob",
        type=str,
        required=True,
        nargs="+",
        help="One or more globs for input text-dialogue JSONs. Files from all globs are concatenated and de-duplicated before sharding.",
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        default=Path("audios_chatterbox"),
        help="Root output directory; per-dialogue paths are <save_dir>/<scenario_dir>/train/<id>/var<NN>/",
    )
    parser.add_argument(
        "--num_variants",
        type=int,
        default=10,
        help="Number of distinct user-voice variants to generate per text dialogue.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for voice selection and TTS sampling.",
    )
    parser.add_argument(
        "--num_shards", type=int, default=1, help="Total number of shards (parallel workers)."
    )
    parser.add_argument(
        "--shard_id", type=int, default=0, help="This worker's shard index in [0, num_shards)."
    )
    parser.add_argument(
        "--max_dialogues",
        type=int,
        default=0,
        help="If >0, process only the first N dialogues from the (sharded) list. Useful for smoke tests.",
    )
    parser.add_argument(
        "--exclude_ids_file",
        type=str,
        default="",
        help="Optional path to a newline-delimited file of dialogue stems to skip (e.g. a blocklist).",
    )
    args = parser.parse_args()

    main(args)
