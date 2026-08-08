import pack_corpus as pc
import unpack_corpus as uc


def test_pack_dialogue_keeps_nested_slots_and_stamps_meta():
    # Real schema: per-word turn-taking decisions live in the NESTED turn["history"] list.
    rec = {
        "example_id": "ex1",
        "speakers": {"A": "user", "B": "assistant"},
        "config": {},
        "context": "A persuasion dialogue.",
        "meta": {
            "style": "spoken",
            "disfluency_target": "user",
            "dataset": "persuader",
            "split": "train",
            "reconstructed_from": "tts_render/audios_260413/...",
        },
        "history": [
            {
                "role": "user",
                "content": "developing countries is broad",
                "history": [
                    {"full_content": "developing countries is broad"},
                    {
                        "word_index": 5,
                        "probs": {"floor_taking": 0.012, "backchannel": 0.207, "silence": 0.781},
                        "decision": "silence",
                        "inserted_token": None,
                    },
                ],
            },
        ],
    }
    out = pc.pack_dialogue(rec, "PER", "Apache-2.0")
    assert out["scenario"] == "PER"
    assert out["license"] == "Apache-2.0"
    assert out["example_id"] == "ex1"
    assert out["context"] == "A persuasion dialogue."
    assert out["style"] == "spoken"
    segs = out["history"][0]["segments"]
    assert segs[0] == {"full_content": "developing countries is broad"}
    assert segs[1]["decision"] == "silence"
    assert segs[1]["probs"]["silence"] == 0.781
    assert "reconstructed_from" not in out and "reconstructed_from" not in str(out)


def test_pack_annotation_keeps_boundaries():
    rec = {
        "example_id": "a1",
        "history": [
            {
                "role": "user",
                "content": "ok sure",
                "boundaries": [
                    {
                        "word_index": 2,
                        "total_count": 5,
                        "counts": {"silent": 1, "backchannel": 1, "floor_taking": 3},
                        "probabilities": {"silent": 0.2, "backchannel": 0.2, "floor_taking": 0.6},
                    }
                ],
            }
        ],
    }
    out = pc.pack_annotation(rec, "TEA", "Apache-2.0")
    assert out["scenario"] == "TEA"
    assert out["license"] == "Apache-2.0"
    b = out["history"][0]["boundaries"][0]
    assert b["total_count"] == 5
    assert b["probabilities"]["floor_taking"] == 0.6


def test_code2dir_covers_every_code_and_does_not_collide_with_internal_codes():
    # HF release codes are UPPERCASE and map to private dir names. Note the trap:
    # release "SOC" is soda, while the internal lowercase "soc" is socraticlm.
    assert uc.CODE2DIR == {
        "TEA": "socraticlm",
        "PLN": "multiwoz",
        "INT": "interviewer",
        "NEG": "negotiator",
        "PER": "persuader",
        "SOC": "soda",
    }
    assert uc.CODE2DIR["SOC"] == "soda"


def test_dialogue_survives_pack_then_unpack():
    rec = {
        "example_id": "ex/1",
        "speakers": ["user", "assistant"],
        "config": {},
        "context": "A persuasion dialogue.",
        "meta": {
            "style": "spoken",
            "disfluency_target": "user",
            "dataset": "persuader",
            "split": "train",
        },
        "history": [
            {
                "role": "user",
                "content": "developing countries is broad",
                "history": [
                    {"full_content": "developing countries is broad"},
                    {
                        "word_index": 5,
                        "probs": {"floor_taking": 0.012, "backchannel": 0.207, "silence": 0.781},
                        "decision": "silence",
                        "inserted_token": None,
                    },
                ],
            },
        ],
    }
    packed = pc.pack_dialogue(rec, "PER", "Apache-2.0")
    out = uc.unpack_dialogue(packed, "persuader", "train")

    assert out["example_id"] == "ex/1"
    assert out["context"] == rec["context"]
    assert out["speakers"] == rec["speakers"]
    assert out["config"] == {}
    assert out["meta"]["dataset"] == "persuader"
    assert out["meta"]["split"] == "train"
    assert out["meta"]["style"] == "spoken"
    assert out["meta"]["disfluency_target"] == "user"
    assert out["meta"]["license"] == "Apache-2.0"
    # "segments" must land back on the nested "history" key the pipeline reads.
    assert out["history"][0]["history"] == rec["history"][0]["history"]
    assert "segments" not in out["history"][0]


def test_annotation_survives_pack_then_unpack():
    rec = {
        "example_id": "a1",
        "history": [
            {
                "role": "user",
                "content": "ok sure",
                "boundaries": [
                    {
                        "word_index": 2,
                        "total_count": 5,
                        "counts": {"silent": 1, "backchannel": 1, "floor_taking": 3},
                        "probabilities": {"silent": 0.2, "backchannel": 0.2, "floor_taking": 0.6},
                    }
                ],
            }
        ],
    }
    packed = pc.pack_annotation(rec, "TEA", "Apache-2.0")
    out = uc.unpack_annotation(packed, "socraticlm", "test")

    assert out["example_id"] == "a1"
    assert out["meta"] == {"dataset": "socraticlm", "split": "test", "license": "Apache-2.0"}
    assert out["history"][0]["boundaries"] == rec["history"][0]["boundaries"]


def test_safe_id_matches_prepare_corpus_sanitization():
    assert uc.safe_id("ex/1") == "ex_1"
    assert uc.safe_id("a-b.c_d") == "a-b.c_d"
    assert uc.safe_id("x y:z") == "x_y_z"
