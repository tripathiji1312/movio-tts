"""Quick sanity check for IndicF5Engine — run in a separate terminal.

Usage:
    uv run python scripts/test_tts.py

Checks:
1. Vocab — all Tamil test chars map correctly (no zeros for non-space)
2. Model load — checkpoint loads without shape errors
3. Synthesis — produces non-empty audio for a short Tamil phrase
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_vocab():
    print("\n=== 1. Vocab check ===")
    vocab_path = "models/indicf5_tanglish/vocab.txt"
    vocab = [l.rstrip("\n") for l in open(vocab_path, encoding="utf-8")]
    char_map = {c: i for i, c in enumerate(vocab)}

    test_str = "உங்கள் கேப் ஐந்து நிமிடத்தில் வரும்"
    missing = [c for c in test_str if c not in char_map]
    zeros = [c for c in test_str if char_map.get(c) == 0 and c != " "]

    print(f"Vocab size: {len(vocab)}")
    print(f"Missing chars: {missing or 'none'}")
    print(f"Non-space chars mapping to 0: {zeros or 'none'}")
    assert not missing, f"Missing: {missing}"
    assert not zeros, f"Zero-mapped non-space: {zeros}"
    print("PASS")


def test_model_load():
    print("\n=== 2. Model load check ===")
    import torch
    ckpt = torch.load("models/indicf5_tanglish/model.pt", map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict") or ckpt
    embed_key = next(k for k in sd if "text_embed" in k and k.endswith(".weight") and len(sd[k].shape) == 2)
    embed_shape = list(sd[embed_key].shape)
    vocab_size = sum(1 for _ in open("models/indicf5_tanglish/vocab.txt"))
    expected_rows = vocab_size + 1
    print(f"Embedding shape: {embed_shape}")
    print(f"Vocab lines: {vocab_size}, expected embed rows: {expected_rows}")
    assert embed_shape[0] == expected_rows, f"Shape mismatch: {embed_shape[0]} != {expected_rows}"
    print("PASS")


def test_synthesis():
    print("\n=== 3. Synthesis check ===")
    from movio.textnorm.normalizer import load_settings
    from movio.acoustic.indicf5_engine import IndicF5Engine

    settings = load_settings()
    engine = IndicF5Engine(settings)
    engine.load()
    print("Model loaded.")

    audio = engine.synthesize("உங்கள் கேப் ஐந்து நிமிடத்தில் வரும்.")
    duration = len(audio) / engine.SAMPLE_RATE
    print(f"Output: {len(audio)} samples, {duration:.2f}s at {engine.SAMPLE_RATE}Hz")
    assert len(audio) > engine.SAMPLE_RATE * 0.5, f"Audio too short: {duration:.2f}s"

    import soundfile as sf
    out_path = "/tmp/tts_test_output.wav"
    sf.write(out_path, audio, engine.SAMPLE_RATE)
    print(f"Saved to {out_path} — play it to verify")
    print("PASS")


if __name__ == "__main__":
    test_vocab()
    test_model_load()
    test_synthesis()
    print("\n=== All checks passed ===")
