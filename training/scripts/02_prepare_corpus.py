"""02 — Prepare the fine-tune corpus.

- Resample everything to 24 kHz mono WAV (IndicF5 native rate)
- Trim leading/trailing silence
- Filter by duration + SNR + optional Rasa style filter
- Emit manifest CSV: audio_path|transcript|duration_s|speaker

Inputs:
  --raw-dir     directory tree from 01_download_data.py:
                  rasa/Tamil/*.parquet          (ai4bharat/Rasa, studio quality)
                  indicvoices_r/**/*.parquet    (ai4bharat/indicvoices_r, crowd-sourced)
                  and/or your own Tanglish recordings:
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


def find_transcript_pairs(raw_dir: Path, delete_parquets: bool = False,
                          rasa_styles: list | None = None):
    """Yield (audio_path_or_data, transcript) pairs from known dataset layouts.

    For parquet-embedded audio: yields (dict with 'array' and 'sampling_rate', transcript).
    For standalone wavs: yields (Path, transcript).

    rasa_styles: if set, only yield rows whose 'style' column matches one of these values.
                 Default None = keep all styles (but neutral/read is best for TTS assistants).
    """
    # Layout A: HuggingFace parquet files with embedded audio columns
    parquets = sorted(raw_dir.rglob("*.parquet"))
    if parquets:
        import pandas as pd

        print(f"  extracting audio from {len(parquets)} parquet files...")
        style_skipped = 0
        for pq_path in parquets:
            try:
                df = pd.read_parquet(pq_path)
            except Exception as exc:
                print(f"  skip unreadable parquet {pq_path.name}: {exc}")
                continue
            # Detect audio and text columns
            audio_col = None
            for col in ("audio", "Audio", "speech", "input_values"):
                if col in df.columns:
                    audio_col = col
                    break
            text_col = None
            for col in ("transcript", "text", "sentence", "transcription", "raw_text"):
                if col in df.columns:
                    text_col = col
                    break
            if audio_col is None or text_col is None:
                print(f"  skip {pq_path.name}: missing audio/text columns "
                      f"(found: {list(df.columns)[:10]})")
                continue
            # Rasa style filter — keep only natural/neutral styles for TTS assistant
            if rasa_styles and "style" in df.columns:
                before = len(df)
                df = df[df["style"].str.lower().isin([s.lower() for s in rasa_styles])]
                style_skipped += before - len(df)
            for idx, row in df.iterrows():
                transcript = row[text_col]
                if not transcript or not isinstance(transcript, str):
                    continue
                audio_data = row[audio_col]
                if isinstance(audio_data, dict):
                    yield audio_data, transcript.strip()
                elif isinstance(audio_data, bytes):
                    yield {"bytes": audio_data}, transcript.strip()
            del df
            if delete_parquets:
                pq_path.unlink(missing_ok=True)
                print(f"  freed {pq_path.name}")
        if style_skipped:
            print(f"  Rasa style filter: skipped {style_skipped} non-neutral rows")
        return

    # Layout B: transcripts.tsv sidecar files (custom recordings)
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
    # Layout C: indicvoices_r style — wav next to same-stem .json/.txt
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


def quality_score(audio: np.ndarray, sr: int) -> float:
    """Combined audio quality proxy (higher = better). Range ~0-1.

    Combines three fast signals (no neural net required):
      1. SNR  — speech-to-noise ratio (high = clean recording)
      2. Clipping ratio — fraction of near-saturated samples (low = no distortion)
      3. Spectral flatness — how tonal vs noisy the signal is (low = speech-like)

    All three are normalised then averaged so each contributes equally.
    """
    # 1. SNR score (capped at 40 dB = score 1.0)
    snr = snr_proxy(audio)
    snr_score = min(snr / 40.0, 1.0)

    # 2. Clipping score — penalise clips with >1% saturated samples
    clip_ratio = float(np.mean(np.abs(audio) > 0.98))
    clip_score = max(0.0, 1.0 - clip_ratio * 50)  # 2% clipping → score 0

    # 3. Spectral flatness score (geometric mean / arithmetic mean of spectrum)
    #    Low flatness = tonal/speech-like; high = noise. We want low flatness.
    try:
        frame = min(2048, len(audio))
        spectrum = np.abs(np.fft.rfft(audio[:frame]))
        spectrum = np.maximum(spectrum, 1e-10)
        geom = np.exp(np.mean(np.log(spectrum)))
        arith = np.mean(spectrum)
        flatness = float(geom / arith)  # 0 (pure tone) … 1 (white noise)
        flatness_score = max(0.0, 1.0 - flatness)
    except Exception:
        flatness_score = 0.5

    return (snr_score + clip_score + flatness_score) / 3.0


def extract_ref_audio(
    speaker_id: str,
    raw_dir: Path,
    voice_out: Path,
    sample_rate: int = 24000,
    target_dur_min: float = 8.0,
    target_dur_max: float = 15.0,
) -> None:
    """Find speaker_id in parquet files, pick the best clip, save as voice profile.

    Scans parquets in raw_dir until it finds a clip from speaker_id that is
    between target_dur_min and target_dur_max seconds and has the best quality
    score. Saves ref.wav + voice.yaml into voice_out/.
    """
    import soundfile as sf
    import librosa

    print(f"Searching for speaker '{speaker_id}' in {raw_dir}...")

    best_audio = None
    best_text = None
    best_score = -1.0
    found_count = 0

    parquets = sorted(raw_dir.rglob("*.parquet"))
    if not parquets:
        raise SystemExit(f"No parquet files found in {raw_dir}. Run 01_download_data.py first.")

    import pandas as pd

    for pq_path in parquets:
        try:
            df = pd.read_parquet(pq_path)
        except Exception as e:
            print(f"  skip {pq_path.name}: {e}")
            continue

        # Find speaker/filename column
        spk_col = None
        for col in ("filename", "speaker_id", "speaker", "spk_id", "spk", "id"):
            if col in df.columns:
                spk_col = col
                break
        if spk_col is None:
            continue

        # Filter to this speaker
        mask = df[spk_col].astype(str).str.contains(speaker_id, regex=False)
        hits = df[mask]
        if hits.empty:
            continue

        found_count += len(hits)
        print(f"  found {len(hits)} clips in {pq_path.name}")

        # Audio + text columns
        audio_col = next((c for c in ("audio", "Audio", "speech") if c in hits.columns), None)
        text_col = next((c for c in ("transcript", "text", "sentence", "transcription") if c in hits.columns), None)
        if audio_col is None or text_col is None:
            continue

        for _, row in hits.iterrows():
            text = row[text_col]
            if not text or not isinstance(text, str):
                continue
            audio_data = row[audio_col]
            try:
                if isinstance(audio_data, dict):
                    if "array" in audio_data:
                        audio = np.array(audio_data["array"], dtype=np.float32)
                        src_sr = audio_data.get("sampling_rate", 16000)
                    elif "bytes" in audio_data:
                        import io
                        audio, src_sr = sf.read(io.BytesIO(audio_data["bytes"]))
                        audio = audio.astype(np.float32)
                    else:
                        continue
                elif isinstance(audio_data, bytes):
                    import io
                    audio, src_sr = sf.read(io.BytesIO(audio_data))
                    audio = audio.astype(np.float32)
                else:
                    continue
            except Exception:
                continue

            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if src_sr != sample_rate:
                audio = librosa.resample(audio, orig_sr=src_sr, target_sr=sample_rate)

            dur = len(audio) / sample_rate
            if not (target_dur_min <= dur <= target_dur_max):
                continue

            score = quality_score(audio, sample_rate)
            if score > best_score:
                best_score = score
                best_audio = trim_silence(audio, sample_rate)
                best_text = text.strip()

        # Stop after finding 5+ candidates (avoid scanning all 67 shards)
        if found_count >= 5:
            break

    if best_audio is None:
        raise SystemExit(
            f"No clips found for speaker '{speaker_id}' with duration "
            f"{target_dur_min}-{target_dur_max}s.\n"
            f"Total clips found: {found_count} (may have wrong duration range)."
        )

    voice_out.mkdir(parents=True, exist_ok=True)
    wav_path = voice_out / "ref.wav"
    sf.write(wav_path, best_audio, sample_rate, subtype="PCM_16")
    dur = len(best_audio) / sample_rate

    yaml_path = voice_out / "voice.yaml"
    yaml_path.write_text(
        f"name: {voice_out.name}\n"
        f"ref_audio: ref.wav\n"
        f'ref_text: "{best_text}"\n'
        f"speaker_id: {speaker_id}\n"
        f"duration_s: {dur:.1f}\n"
        f"quality_score: {best_score:.3f}\n",
        encoding="utf-8",
    )

    print(f"\nExtracted ref audio for '{speaker_id}':")
    print(f"  wav:     {wav_path}  ({dur:.1f}s, quality={best_score:.3f})")
    print(f"  text:    {best_text}")
    print(f"  config:  {yaml_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out", default="/kaggle/working/data")
    ap.add_argument("--sample-rate", type=int, default=24000)
    ap.add_argument("--min-dur", type=float, default=1.0)
    ap.add_argument("--max-dur", type=float, default=12.0)
    ap.add_argument("--min-snr", type=float, default=12.0)
    ap.add_argument("--quality-top-pct", type=float, default=30.0,
                    help="After basic SNR filter, keep only the top N%% of clips by "
                         "combined quality score (SNR + no-clipping + spectral flatness). "
                         "Use 100 to disable. Default 30 keeps the cleanest ~3-4h from "
                         "indicvoices_r, which reduces catastrophic forgetting.")
    ap.add_argument("--rasa-styles", default="all",
                    help="Comma-separated Rasa 'style' values to keep (default: all). "
                         "All Rasa styles are studio-quality — safe to keep everything. "
                         "Set to specific values (e.g. 'neutral,read') only if you want "
                         "to exclude expressive styles.")
    ap.add_argument("--delete-parquets", action="store_true",
                    help="Delete parquet files after extraction to free disk space.")
    ap.add_argument("--val-frac", type=float, default=0.03)
    ap.add_argument("--test-frac", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--extract-ref", metavar="SPEAKER_ID",
        help="Extract a ref audio clip for this speaker ID from the parquets and "
             "save it as a voice profile. Use with --voice-out. "
             "Example: --extract-ref TAM_F_WIKI_01124"
    )
    ap.add_argument(
        "--voice-out", default=None,
        help="Output directory for the voice profile (default: config/voices/<speaker_id>). "
             "Only used with --extract-ref."
    )
    ap.add_argument(
        "--ref-min-dur", type=float, default=3.0,
        help="Min clip duration in seconds for ref audio extraction (default: 3)."
    )
    ap.add_argument(
        "--ref-max-dur", type=float, default=15.0,
        help="Max clip duration in seconds for ref audio extraction (default: 15)."
    )
    args = ap.parse_args()

    # --extract-ref mode: just extract one speaker's best clip and exit
    if args.extract_ref:
        voice_out = Path(args.voice_out) if args.voice_out else \
            Path("config/voices") / args.extract_ref.lower().replace(" ", "_")
        extract_ref_audio(
            speaker_id=args.extract_ref,
            raw_dir=Path(args.raw_dir),
            voice_out=voice_out,
            sample_rate=args.sample_rate,
            target_dur_min=args.ref_min_dur,
            target_dur_max=args.ref_max_dur,
        )
        return

    import soundfile as sf
    import librosa

    raw_dir = Path(args.raw_dir)
    out = Path(args.out)
    audio_out = out / "audio"
    audio_out.mkdir(parents=True, exist_ok=True)

    rasa_styles = None if args.rasa_styles.lower() == "all" else [
        s.strip() for s in args.rasa_styles.split(",") if s.strip()
    ]
    if rasa_styles:
        print(f"Rasa style filter: keeping styles {rasa_styles}")

    rows = []
    seen_hashes = set()
    skipped = {"unreadable": 0, "duration": 0, "snr": 0, "dup": 0}
    for audio_src, text in find_transcript_pairs(raw_dir, delete_parquets=args.delete_parquets,
                                                  rasa_styles=rasa_styles):
        try:
            if isinstance(audio_src, dict):
                # Parquet-embedded audio
                if "array" in audio_src:
                    audio = np.array(audio_src["array"], dtype=np.float32)
                    src_sr = audio_src.get("sampling_rate", 16000)
                elif "bytes" in audio_src:
                    import io
                    audio, src_sr = sf.read(io.BytesIO(audio_src["bytes"]))
                    audio = audio.astype(np.float32)
                elif "path" in audio_src and audio_src.get("path"):
                    audio, src_sr = librosa.load(audio_src["path"], sr=None, mono=True)
                else:
                    skipped["unreadable"] += 1
                    continue
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                if src_sr != args.sample_rate:
                    audio = librosa.resample(audio, orig_sr=src_sr, target_sr=args.sample_rate)
                source_name = "indicvoices_r"
            else:
                # Standalone wav file
                audio, _ = librosa.load(str(audio_src), sr=args.sample_rate, mono=True)
                source_name = audio_src.parent.name
        except Exception as exc:
            skipped["unreadable"] += 1
            if skipped["unreadable"] <= 5:
                print(f"skip unreadable: {exc}")
            continue

        dur = len(audio) / args.sample_rate
        if not (args.min_dur <= dur <= args.max_dur):
            skipped["duration"] += 1
            continue
        if snr_proxy(audio) < args.min_snr:
            skipped["snr"] += 1
            continue
        audio = trim_silence(audio, args.sample_rate)
        h = hash(audio.tobytes())
        if h in seen_hashes:
            skipped["dup"] += 1
            continue
        seen_hashes.add(h)

        q = quality_score(audio, args.sample_rate)
        fname = f"utt_{len(rows):07d}.wav"
        sf.write(audio_out / fname, audio, args.sample_rate, subtype="PCM_16")
        rows.append(
            {
                "audio": str(audio_out / fname),
                "text": text.strip(),
                "duration_s": round(len(audio) / args.sample_rate, 3),
                "source": source_name,
                "_quality": q,
            }
        )
        if len(rows) % 1000 == 0:
            print(f"  processed {len(rows)} utterances...")

    print(f"Skipped: {skipped}")
    print(f"Kept after basic filters: {len(rows)} utterances")

    # Quality filter — keep top N% by combined score, delete the rest from disk
    if args.quality_top_pct < 100.0 and len(rows) > 10:
        threshold = np.percentile([r["_quality"] for r in rows],
                                  100.0 - args.quality_top_pct)
        kept, dropped = [], []
        for r in rows:
            (kept if r["_quality"] >= threshold else dropped).append(r)
        for r in dropped:
            Path(r["audio"]).unlink(missing_ok=True)
        rows = kept
        print(f"Quality filter (top {args.quality_top_pct:.0f}%): "
              f"kept {len(rows)}, dropped {len(dropped)} "
              f"(threshold score={threshold:.3f})")

    # Strip internal quality field before writing CSVs
    for r in rows:
        r.pop("_quality", None)

    print(f"Final training set: {len(rows)} utterances")

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
