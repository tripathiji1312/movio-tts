"""Quality evaluation — problem-statement deliverable 7.

Objective metrics over a synthesized test set:
  - WER / CER  (intelligibility, Tamil ASR via vasista22/whisper-tamil-base)
  - UTMOS      (objective MOS proxy)
  - Speaker similarity (cosine of ECAPA/resemblyzer embeddings vs reference)

Usage (GPU machine, e.g. the Kaggle session or a GPU box):
  python eval/run_quality_eval.py --model ai4bharat/IndicF5 \
      --testset eval/testsets/tanglish_transport_200.tsv \
      --ref-audio config/voices/ta_female_neutral/ref.wav \
      --num-samples 100 --out eval/quality_base.json
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np
import torch


def synthesize(model_id: str, texts: list[str], ref_audio: str | None,
               ref_text: str | None) -> list[str]:
    import soundfile as sf
    from transformers import AutoModel

    model = AutoModel.from_pretrained(model_id, trust_remote_code=True).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    out_dir = Path("/tmp/movio_eval_audio")
    out_dir.mkdir(exist_ok=True)
    paths = []
    for i, text in enumerate(texts):
        payload = {"text": text}
        if ref_audio:
            payload["ref_audio"] = ref_audio
            payload["ref_text"] = ref_text or ""
        audio = np.asarray(model(payload), dtype=np.float32)
        p = str(out_dir / f"utt_{i:04d}.wav")
        sf.write(p, audio, 24000)
        paths.append(p)
    return paths


def asr_transcribe(paths: list[str]) -> list[str]:
    import librosa
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    model_name = "vasista22/whisper-tamil-base"
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    hyps = []
    for p in paths:
        wav, _ = librosa.load(p, sr=16000, mono=True)
        feats = processor(wav, sampling_rate=16000, return_tensors="pt").input_features.to(device)
        with torch.no_grad():
            ids = model.generate(feats, language="tamil", task="transcribe", max_new_tokens=200)
        hyps.append(processor.batch_decode(ids, skip_special_tokens=True)[0])
    return hyps


def wer_cer(refs: list[str], hyps: list[str]) -> dict:
    import jiwer

    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
    ])
    refs_t = [transformation(r) for r in refs]
    hyps_t = [transformation(h) for h in hyps]
    return {
        "wer": round(jiwer.wer(refs_t, hyps_t), 4),
        "cer": round(jiwer.cer(refs_t, hyps_t), 4),
        "n": len(refs_t),
    }


def utmos_scores(paths: list[str]) -> list[float]:
    predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    import librosa

    scores = []
    for p in paths:
        wav, sr = librosa.load(p, sr=16000, mono=True)
        with torch.no_grad():
            scores.append(float(predictor(torch.from_numpy(wav).unsqueeze(0), sr).item()))
    return scores


def speaker_similarity(paths: list[str], ref_audio: str) -> list[float]:
    from resemblyzer import VoiceEncoder, preprocess_wav

    encoder = VoiceEncoder()
    ref_emb = encoder.embed_utterance(preprocess_wav(ref_audio))
    sims = []
    for p in paths:
        emb = encoder.embed_utterance(preprocess_wav(p))
        sims.append(float(np.dot(ref_emb, emb) / (np.linalg.norm(ref_emb) * np.linalg.norm(emb))))
    return sims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ai4bharat/IndicF5")
    ap.add_argument("--testset", required=True)
    ap.add_argument("--ref-audio", default=None)
    ap.add_argument("--ref-text", default=None)
    ap.add_argument("--num-samples", type=int, default=100)
    ap.add_argument("--skip-asr", action="store_true")
    ap.add_argument("--skip-sim", action="store_true")
    ap.add_argument("--out", default="eval/quality.json")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.testset, encoding="utf-8"), delimiter="\t"))
    rows = rows[: args.num_samples]
    texts = [r["text"] for r in rows]

    print(f"Synthesizing {len(texts)} utterances with {args.model} ...")
    paths = synthesize(args.model, texts, args.ref_audio, args.ref_text)

    report = {"model": args.model, "n": len(texts)}

    if not args.skip_asr:
        print("ASR round-trip (WER/CER)...")
        hyps = asr_transcribe(paths)
        report["asr"] = wer_cer(texts, hyps)
        report["asr"]["samples"] = [
            {"ref": a, "hyp": b} for a, b in list(zip(texts, hyps))[:10]
        ]

    print("UTMOS...")
    u = utmos_scores(paths)
    report["utmos"] = {"mean": round(statistics.mean(u), 3),
                       "std": round(statistics.stdev(u), 3)}

    if args.ref_audio and not args.skip_sim:
        print("Speaker similarity...")
        s = speaker_similarity(paths, args.ref_audio)
        report["speaker_similarity"] = {"mean": round(statistics.mean(s), 3)}

    by_cat: dict[str, list[float]] = {}
    if not args.skip_asr and "asr" in report:
        per_cat_hyp = {}
        for r, h in zip(rows, hyps):
            per_cat_hyp.setdefault(r["category"], ([], []))
            per_cat_hyp[r["category"]][0].append(r["text"])
            per_cat_hyp[r["category"]][1].append(h)
        report["per_category_wer"] = {
            cat: round(__import__("jiwer").wer(v[0], v[1]), 4)
            for cat, v in per_cat_hyp.items()
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in report.items() if k != "asr"}, indent=2, ensure_ascii=False))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
