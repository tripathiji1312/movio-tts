"""02 — Prepare the fine-tune corpus.

- Resample everything to 24 kHz mono WAV (IndicF5 native rate)
- Trim leading/trailing silence
- Filter by duration + basic SNR proxy
- Emit manifest CSV: audio_path|transcript|duration_s|speaker

Inputs:
  --raw-dir     directory tree from 01_download_data.py (wav + json transcripts)
                and/or your own recorded Tanglish data in the same shape:
                  audio/*.wav + transcripts.tsv (filename<TAB>text)

Output:
  /kaggle/working/data/train.csv (+ val/test splits), audio copied into
  data/audio/, ref_pool.csv for zero-shot prompt sampling.
"""

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

import numpy as np


def find_transcript_pairs(raw_dir: Path):
    """Yield (audio_path, transcript) pairs from known dataset layouts."""
    # Layout A: transcripts.tsv sidecar files (custom recordings)
    for tsv in raw_dir.rglob("*.tsv"):
        with open(tsv, encoding="utf-8") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    wav = tsv.parent / parts[0]
                    if not wav.exists():
                        cand = tsv.parent / "audio" / parts[0]
                        if cand.exists():
                            wav = cand
                    if wav.exists():
                        yield wav, parts[1]
    # Layout B: indicvoices_r style — wav next to same-stem .json/.txt
    for wav in raw_dir.rglob("*.wav"):
        stem = wav.with_suffix("")
        transcript = None
        for ext in (".json", ".txt"):
            side = Path(str(stem) + ext)
            if side.exists():
                if ext == ".json":
                    try:
                        meta = json.loads(side.read_text(encoding="utf-8"))
                        transcript = (
                            meta.get("transcript")
                            or meta.get("text")
                            or meta.get("sentence")
                        )
                    except json.JSONDecodeError:
                        pass
                else:
                    transcript = side.read_text(encoding="utf-8").strip()
                break
        if transcript:
            yield wav, transcript


def trim_silence(audio: np.ndarray, sr: int, top_db: int = 40, pad_ms: int = 100) -> np.ndarray:
    import librosa

    intervals = librosa.effects.split(audio, top_db=top_db)
    if len(intervals) == 0:
        return audio
    start, end = intervals[0][0], intervals[-1][1]
    pad = int(sr * pad_ms / 1000)
    start = max(0, start - pad)
    end = min(len(audio), end + pad)
    return audio[start:end]


def snr_proxy(audio: np.ndarray) -> float:
    frame = 2048
    n = len(audio) // frame
    if n < 2:
        return 99.0
    rms = np.array(
        [np.sqrt(np.mean(audio[i * frame : (i + 1) * frame] ** 2)) for i in range(n)]
    )
    speech = np.percentile(rms, 90)
    noise_floor = np.percentile(rms, 10)
    if noise_floor <= 1e-8:
        return 0.0
    return float(20 * np.log10(speech / noise_floor))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out", default="/kaggle/working/data")
    ap.add_argument("--sample-rate", type=int, default=24000)
    ap.add_argument("--min-dur", type=float, default=1.0)
    ap.add_argument("--max-dur", type=float, default=12.0)
    ap.add_argument("--min-snr", type=float, default=12.0)
    ap.add_argument("--val-frac", type=float, default=0.03)
    ap.add_argument("--test-frac", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import soundfile as sf
    import librosa

    raw_dir = Path(args.raw_dir)
    out = Path(args.out)
    audio_out = out / "audio"
    audio_out.mkdir(parents=True, exist_ok=True)

    rows = []
    seen_hashes = set()
    for wav_path, text in find_transcript_pairs(raw_dir):
        try:
            audio, orig_sr = librosa.load(str(wav_path), sr=args.sample_rate, mono=True)
        except Exception as exc:
            print(f"skip unreadable {wav_path}: {exc}")
            continue
        dur = len(audio) / args.sample_rate
        if not (args.min_dur <= dur <= args.max_dur):
            continue
        if snr_proxy(audio) < args.min_snr:
            continue
        audio = trim_silence(audio, args.sample_rate)
        h = hash(audio.tobytes())
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        fname = f"utt_{len(rows):07d}_{wav_path.stem}.wav"
        sf.write(audio_out / fname, audio, args.sample_rate, subtype="PCM_16")
        rows.append(
            {
                "audio": str(audio_out / fname),
                "text": text.strip(),
                "duration_s": round(len(audio) / args.sample_rate, 3),
                "source": wav_path.parent.name,
            }
        )

    random.Random(args.seed).shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_frac))
    n_test = max(1, int(len(rows) * args.test_frac))
    test, val, train = rows[:n_test], rows[n_test : n_test + n_val], rows[n_test + n_val :]

    fields = ["audio", "text", "duration_s", "source"]
    def write(name, subset):
        with open(out / name, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(subset)

    write("train.csv", train)
    write("val.csv", val)
    write("test.csv", test)

    ref_rows = [
        {"audio": r["audio"], "text": r["text"]}
        for r in random.Random(args.seed + 1).sample(train, min(200, len(train)))
    ]
    with open(out / "ref_pool.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["audio", "text"])
        w.writeheader()
        w.writerows(ref_rows)

    total_min = sum(r["duration_s"] for r in train) / 60
    print(f"train={len(train)} val={len(val)} test={len(test)} | {total_min:.1f} min of training audio")
    print(f"manifests written to {out}")


if __name__ == "__main__":
    main()
