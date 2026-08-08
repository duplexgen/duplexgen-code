"""Per-variant backchannel audio cache.

Backchannels are synthesized by Chatterbox using the speaker's own evolving
voice prompt, so cached audio is only valid within a single dialogue variant.
Callers MUST clear() between variants -- the cumulative prompt changes as a
dialogue progresses, and a cache that outlived the variant would leak a stale
voice into the next one.

Deliberately stdlib-only so it can be unit-tested without importing the heavy
TTS stack.
"""

import re

_TRAIL = re.compile(r"[\s\.,!\?\-]+$")
_LEAD = re.compile(r"^[\s\.,!\?\-]+")


def normalize_bc_text(text: str) -> str:
    """Canonical cache key for a backchannel token: trimmed, unpunctuated, lowercased."""
    return _LEAD.sub("", _TRAIL.sub("", text.strip())).lower()


class BackchannelCache:
    """Maps (speaker_idx, normalized_text) -> synthesized audio.

    The backchannel vocabulary is tiny (12 tokens), so this removes nearly all
    redundant synthesis within a variant at negligible memory cost.
    """

    def __init__(self):
        self._store = {}

    def get(self, speaker_idx: int, text: str):
        return self._store.get((int(speaker_idx), normalize_bc_text(text)))

    def put(self, speaker_idx: int, text: str, audio) -> None:
        self._store[(int(speaker_idx), normalize_bc_text(text))] = audio

    def clear(self) -> None:
        self._store.clear()

    @property
    def size(self) -> int:
        return len(self._store)
