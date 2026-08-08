import bc_cache


def test_normalize_strips_edge_punctuation_and_casefolds():
    assert bc_cache.normalize_bc_text("  Uh-huh.  ") == "uh-huh"
    assert bc_cache.normalize_bc_text("Yeah?") == "yeah"
    assert bc_cache.normalize_bc_text("!!Right!!") == "right"
    assert bc_cache.normalize_bc_text("mm-hmm") == "mm-hmm"


def test_get_returns_none_on_miss():
    c = bc_cache.BackchannelCache()
    assert c.get(0, "yeah") is None


def test_put_then_get_roundtrips_and_normalizes_the_key():
    c = bc_cache.BackchannelCache()
    c.put(0, "Yeah?", "AUDIO")
    assert c.get(0, "  yeah. ") == "AUDIO"
    assert c.size == 1


def test_speakers_do_not_share_entries():
    c = bc_cache.BackchannelCache()
    c.put(0, "yeah", "SPK0")
    c.put(1, "yeah", "SPK1")
    assert c.get(0, "yeah") == "SPK0"
    assert c.get(1, "yeah") == "SPK1"
    assert c.size == 2


def test_clear_empties_the_cache():
    c = bc_cache.BackchannelCache()
    c.put(0, "yeah", "AUDIO")
    c.clear()
    assert c.get(0, "yeah") is None
    assert c.size == 0
