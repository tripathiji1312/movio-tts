"""Quick smoke test: synthesize sample Tamil/Tanglish sentences and save WAVs.

Usage:
    python scripts/test_synthesis.py
    python scripts/test_synthesis.py --text "உங்கள் கேப் 5 நிமிடத்தில் வரும்"
    python scripts/test_synthesis.py --out output/
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TEST_SENTENCES = [
    "உங்கள் கேப் 5 நிமிடத்தில் வரும்.",
    "Your OTP is 4832. Please share it with the driver.",
    "Unga booking confirm aayiduchu. Driver name Rajesh.",
    "உங்கள் ஓட்டுநர் TN45AB1234 வண்டியில் வருகிறார்.",
    "The fare is Rs. 350 for 8.5 km.",
    "காலை 7:30-க்கு pickup ready-யா இருங்க.",
]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default=None, help="Single sentence to synthesize")
    ap.add_argument("--out", default="output", help="Output directory for WAV files")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from movio.pipeline import SynthesisRequest, TTSPipeline
    from movio.textnorm.normalizer import load_settings
    from movio.utils.audio import wav_bytes

    settings = load_settings()
    pipeline = TTSPipeline(settings)

    print("Loading models...")
    t0 = time.perf_counter()
    await pipeline.warmup()
    print(f"Models loaded in {(time.perf_counter() - t0) * 1000:.0f}ms")
    print(f"Engine stats: {pipeline.engine.stats}")
    print()

    sentences = [args.text] if args.text else TEST_SENTENCES

    for i, text in enumerate(sentences):
        print(f"[{i+1}/{len(sentences)}] {text[:60]}...")
        try:
            t0 = time.perf_counter()
            result = await pipeline.synthesize(SynthesisRequest(text=text))
            elapsed = (time.perf_counter() - t0) * 1000

            wav_data = wav_bytes(result.audio, result.sample_rate)
            fname = out_dir / f"sample_{i+1:02d}.wav"
            fname.write_bytes(wav_data)

            duration_s = len(result.audio) / result.sample_rate
            rtf = elapsed / 1000 / duration_s if duration_s > 0 else float("inf")

            print(f"  OK: {elapsed:.0f}ms | {duration_s:.2f}s audio | RTF={rtf:.3f}")
            print(f"  Normalized: {result.normalized_text[:80]}")
            print(f"  Timings: A={result.timings.stage_a_ms:.1f}ms "
                  f"B={result.timings.stage_b_ms:.1f}ms "
                  f"C={result.timings.stage_c_first_chunk_ms:.1f}ms")
            print(f"  Saved: {fname}")
        except Exception as exc:
            print(f"  FAILED: {exc}")
        print()

    print(f"Done. Output in {out_dir}/")


if __name__ == "__main__":
    asyncio.run(main())
