"""06 — Objective evaluation of the fine-tuned model.

Metrics:
  - UTMOS  (objective MOS proxy)      — tarepan/SpeechMOS
  - WER    (intelligibility, Tamil)   — vasista22/whisper-tamil-base

Loads models via F5-TTS's own inference stack so it works with both the
fine-tuned .pt bundle and the base SWivid/F5-TTS weights.

Usage:
    # Evaluate fine-tuned model:
    python training/scripts/06_evaluate.py \
        --model /kaggle/working/indicf5_tanglish_merged \
        --f5tts-dir /kaggle/working/f5tts_out/F5-TTS \
        --test /kaggle/working/data/test.csv \
        --ref-pool /kaggle/working/data/ref_pool.csv \
        --out /kaggle/working/eval_finetuned.json

    # Evaluate base model (for comparison):
    python training/scripts/06_evaluate.py \
        --model base \
        --f5tts-dir /kaggle/working/f5tts_out/F5-TTS \
        --test /kaggle/working/data/test.csv \
        --ref-pool /kaggle/working/data/ref_pool.csv \
        --out /kaggle/working/eval_base.json
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch


def _patch_f5tts_init(f5tts_src: Path):
    """Make Trainer import optional in __init__.py to avoid dataset→hub crash."""
    init_py = f5tts_src / "src" / "f5_tts" / "model" / "__init__.py"
    if init_py.exists():
        src = init_py.read_text()
        if "try:" not in src and "from f5_tts.model.trainer import Trainer" in src:
            src = src.replace(
                "from f5_tts.model.trainer import Trainer",
                "try:\n    from f5_tts.model.trainer import Trainer\nexcept ImportError:\n    Trainer = None",
            )
            init_py.write_text(src)


def _resize_text_embed(sd: dict, model) -> dict:
    """Pad text_embed.weight in checkpoint to match model's vocab size (mean-init new rows)."""
    import torch as _torch
    model_sd = model.state_dict()
    for key in list(sd.keys()):
        if "text_embed" in key and key.endswith(".weight") and key in model_sd:
            ckpt_w, model_w = sd[key], model_sd[key]
            if ckpt_w.shape != model_w.shape and ckpt_w.shape[1] == model_w.shape[1]:
                new_w = model_w.clone()
                new_w[: ckpt_w.shape[0]] = ckpt_w
                new_w[ckpt_w.shape[0] :] = ckpt_w.mean(dim=0, keepdim=True)
                sd[key] = new_w
                print(f"Resized {key}: {list(ckpt_w.shape)} → {list(model_w.shape)}")
    return sd


def load_f5tts_model(model_dir: str, f5tts_src: Path, device: str):
    """Load a fine-tuned or base F5-TTS model using the F5-TTS inference stack."""
    f5tts_src_str = str(f5tts_src / "src")
    if f5tts_src_str not in sys.path:
        sys.path.insert(0, f5tts_src_str)

    _patch_f5tts_init(f5tts_src)
    from f5_tts.model.backbones.dit import DiT
    from f5_tts.infer.utils_infer import load_vocoder, load_checkpoint

    model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)

    if model_dir == "base":
        # Use ai4bharat/IndicF5 (MIT) as the baseline — same model we fine-tune from.
        # IndicF5 safetensors has _orig_mod prefix + vocoder keys that must be stripped.
        indicf5_base_dir = f5tts_src.parent / "indicf5_base"
        converted_path = indicf5_base_dir / "model_converted.safetensors"
        raw_path = indicf5_base_dir / "model.safetensors"
        vocab_path = str(f5tts_src / "src" / "f5_tts" / "infer" / "examples" / "vocab.txt")
        indicf5_vocab = indicf5_base_dir / "vocab.txt"
        if indicf5_vocab.exists():
            vocab_path = str(indicf5_vocab)
        use_ema = True  # converted keys still have "ema_model." prefix

        if converted_path.exists():
            ckpt_path = str(converted_path)
        else:
            if not raw_path.exists():
                from huggingface_hub import hf_hub_download
                indicf5_base_dir.mkdir(parents=True, exist_ok=True)
                hf_hub_download(
                    "ai4bharat/IndicF5", "model.safetensors",
                    local_dir=str(indicf5_base_dir),
                )
            # Convert: strip _orig_mod prefix + vocoder keys
            print("Converting IndicF5 base weights for evaluation...")
            from safetensors.torch import load_file as _st_load, save_file as _st_save
            raw = _st_load(str(raw_path))
            converted = {}
            for k, v in raw.items():
                if k.startswith("vocoder."):
                    continue
                converted[k.replace("ema_model._orig_mod.", "ema_model.")] = v
            _st_save(converted, str(converted_path))
            print(f"Converted: {len(raw)} → {len(converted)} keys → {converted_path}")
            ckpt_path = str(converted_path)
    else:
        model_path = Path(model_dir)
        ckpt_path = str(model_path / "model.pt")
        vocab_path = str(model_path / "vocab.txt")
        use_ema = False  # 05_merge_export.py already stripped ema prefix; saved as model_state_dict
        if not Path(ckpt_path).exists():
            raise SystemExit(f"model.pt not found in {model_dir}. Run 05_merge_export.py first.")

    from f5_tts.infer.utils_infer import get_tokenizer
    vocab_char_map, vocab_size = get_tokenizer(vocab_path, "custom")

    from f5_tts.model.cfm import CFM  # bypass __init__ to avoid Trainer→dataset→hub chain

    model = CFM(
        transformer=DiT(**model_cfg, text_num_embeds=vocab_size, mel_dim=100),
        mel_spec_kwargs=dict(
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mel_channels=100,
            target_sample_rate=24000,
            mel_spec_type="vocos",
        ),
        vocab_char_map=vocab_char_map,
    )

    if ckpt_path.endswith(".safetensors"):
        # load_checkpoint uses torch.load which can't read safetensors — load manually
        from safetensors.torch import load_file as _st_load
        sd = _st_load(ckpt_path)
        # Strip "ema_model." prefix — keys are already cleaned of _orig_mod
        sd = {k.replace("ema_model.", "", 1) if k.startswith("ema_model.") else k: v
              for k, v in sd.items()}
        sd = _resize_text_embed(sd, model)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        benign = {"initted", "step"}
        real_missing = [k for k in missing if k not in benign]
        if real_missing:
            print(f"WARNING: {len(real_missing)} missing keys (first 5: {real_missing[:5]})")
    else:
        import torch as _torch
        ckpt = _torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict") or ckpt.get("ema_model_state_dict") or ckpt
        sd = _resize_text_embed(sd, model)
        model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()
    vocoder = load_vocoder(vocoder_name="vocos", is_local=False, device=device)
    return model, vocoder, vocab_char_map


def synth_samples(model_dir: str, f5tts_src: Path, test_csv: Path, limit: int,
                  ref_csv: Path | None, device: str):
    import soundfile as sf
    from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text

    model, vocoder, vocab_char_map = load_f5tts_model(model_dir, f5tts_src, device)

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

    import random
    rng = random.Random(7)

    outs = []
    for i, row in enumerate(rows):
        if refs:
            ref = rng.choice(refs)
            ref_audio_path = ref["audio"]
            ref_text = ref["text"]
        elif "audio" in row:
            ref_audio_path = row["audio"]
            ref_text = row.get("text", "")
        else:
            print(f"Skipping row {i}: no reference audio")
            continue

        if not Path(ref_audio_path).exists():
            print(f"Skipping row {i}: ref audio not found: {ref_audio_path}")
            continue

        ref_audio, ref_text_processed = preprocess_ref_audio_text(ref_audio_path, ref_text)
        audio, _, _ = infer_process(
            ref_audio, ref_text_processed, row["text"],
            model, vocoder,
            device=device,
        )
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
        wav, sr = _load_wav(p)
        with torch.no_grad():
            score = predictor(torch.from_numpy(wav).unsqueeze(0), sr)
        scores.append(float(score.item()))
    return scores


def _load_wav(path: str):
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

    hyps, refs = [], []
    for s in samples:
        wav, _ = _load_wav(s["path"])
        feats = processor(wav, sampling_rate=16000, return_tensors="pt").input_features.to(device)
        with torch.no_grad():
            ids = model.generate(feats, language="tamil", task="transcribe", max_new_tokens=200)
        hyps.append(processor.batch_decode(ids, skip_special_tokens=True)[0])
        refs.append(s["reference_text"])

    return {"wer": round(jiwer.wer(refs, hyps), 4), "n": len(refs), "hypotheses": hyps[:10]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="Path to exported model dir (from 05_merge_export.py), or 'base'")
    ap.add_argument("--f5tts-dir", default="/kaggle/working/f5tts_out/F5-TTS",
                    help="Root of cloned SWivid/F5-TTS repo")
    ap.add_argument("--test", required=True)
    ap.add_argument("--ref-pool", default=None)
    ap.add_argument("--num-samples", type=int, default=50)
    ap.add_argument("--out", default="/kaggle/working/eval.json")
    ap.add_argument("--skip-wer", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    f5tts_src = Path(args.f5tts_dir)

    samples = synth_samples(
        args.model, f5tts_src, Path(args.test), args.num_samples,
        Path(args.ref_pool) if args.ref_pool else None, device,
    )

    if not samples:
        raise SystemExit("No samples synthesized — check ref audio paths in test/ref CSVs.")

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
