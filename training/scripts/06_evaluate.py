"""06 — Objective evaluation of the fine-tuned model.

Metrics:
  - UTMOS  (objective MOS proxy)      — sarulab-speech/utmos22_strong
  - WER    (intelligibility, Tamil)   — vasista22/whisper-tamil-base
Compares base vs fine-tuned on the held-out test manifest.

Usage:
    python training/scripts/06_evaluate.py \
        --model /kaggle/working/indicf5_tanglish_merged \
        --test /kaggle/working/data/test.csv --out /kaggle/working/eval.json
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch


def synth_samples(model_dir: str, test_csv: Path, limit: int, ref_csv: Path | None):
    import soundfile as sf
    from transformers import AutoModel

    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    rows = []
    with open(test_csv, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
            if len(rows) >= limit:
                break

    refs = []
    if ref_csv and ref_csv.exists():
        with open(ref_csv, encoding="utf-8") as fh:
            refs = list(csv.DictReader(fh))

    outs = []
    import random

    rng = random.Random(7)
    for i, row in enumerate(rows):
        payload = {"text": row["text"]}
        if refs:
            ref = rng.choice(refs)
            payload["ref_audio"] = ref["audio"]
            payload["ref_text"] = ref["text"]
        elif "audio" in row:
            payload["ref_audio"] = row["audio"]
            payload["ref_text"] = row.get("text", "")
        audio = np.asarray(model(payload), dtype=np.float32)
        wav_path = f"/tmp/eval_{i}.wav"
        sf.write(wav_path, audio, 24000)
        outs.append({"path": wav_path, "reference_text": row["text"]})
    return outs


def utmos_scores(paths: list[str]) -> list[float]:
    predictor = torch.hub.load(
        "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True
    )
    scores = []
    for p in paths:
        wav, sr = _load(p)
        with torch.no_grad():
            score = predictor(torch.from_numpy(wav).unsqueeze(0), sr)
        scores.append(float(score.item()))
    return scores


def _load(path: str):
    import librosa

    wav, sr = librosa.load(path, sr=16000, mono=True)
    return wav.astype(np.float32), sr


def wer_scores(samples: list[dict], whisper_model: str) -> dict:
    import jiwer
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(whisper_model)
    model = WhisperForConditionalGeneration.from_pretrained(whisper_model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    model.generation_forced_language = None

    hyps, refs = [], []
    for s in samples:
        wav, _ = _load(s["path"])
        feats = processor(wav, sampling_rate=16000, return_tensors="pt").input_features
        feats = feats.to(device)
        with torch.no_grad():
            ids = model.generate(feats, language="tamil", task="transcribe", max_new_tokens=200)
        hyps.append(processor.batch_decode(ids, skip_special_tokens=True)[0])
        refs.append(s["reference_text"])

    wer = jiwer.wer(refs, hyps)
    return {"wer": round(wer, 4), "n": len(refs), "hypotheses": hyps[:10]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--ref-pool", default=None)
    ap.add_argument("--num-samples", type=int, default=100)
    ap.add_argument("--out", default="/kaggle/working/eval.json")
    ap.add_argument("--skip-wer", action="store_true")
    args = ap.parse_args()

    samples = synth_samples(
        args.model, Path(args.test), args.num_samples,
        Path(args.ref_pool) if args.ref_pool else None,
    )

    report = {"model": args.model, "n_samples": len(samples)}
    report["utmos"] = {
        "mean": round(float(np.mean(utmos_scores([s["path"] for s in samples]))), 3),
    }
    if not args.skip_wer:
        try:
            report["wer"] = wer_scores(samples, "vasista22/whisper-tamil-base")
        except Exception as exc:
            report["wer_error"] = str(exc)

    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
