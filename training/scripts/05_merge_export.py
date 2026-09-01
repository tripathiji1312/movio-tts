"""05 — Export fine-tuned F5-TTS checkpoint into a self-contained bundle.

The F5-TTS trainer produces .pt files containing F5-TTS CFM weights (keys like
transformer.*, mel_spec.*) — NOT IndicF5 HuggingFace weights. These two models
share architecture but NOT the same HF wrapper, so we cannot naively copy weights
into AutoModel.from_pretrained().

This script instead packages the fine-tuned .pt alongside the vocab and F5-TTS
config so inference can load it directly with F5-TTS's own load_checkpoint().

Output layout:
    <out>/
        model.pt          — fine-tuned weights (EMA, stripped of ema_model. prefix)
        vocab.txt         — the extended vocabulary used during training
        config.json       — F5-TTS model config for reconstruction
        README.md         — how to load for serving

Usage:
    python training/scripts/05_merge_export.py \
        --ckpt /kaggle/working/f5tts_out/model_last.pt \
        --f5tts-dir /kaggle/working/f5tts_out/F5-TTS \
        --out /kaggle/working/indicf5_tanglish_merged
"""

import argparse
import json
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="F5-TTS .pt checkpoint (model_last.pt)")
    ap.add_argument("--f5tts-dir", default="/kaggle/working/f5tts_out/F5-TTS",
                    help="Root of cloned SWivid/F5-TTS repo")
    ap.add_argument("--out", default="/kaggle/working/indicf5_tanglish_merged")
    # backwards compat
    ap.add_argument("--adapter-dir", default=None, help="(deprecated, ignored — use --ckpt)")
    ap.add_argument("--base-model", default=None, help="(deprecated, ignored)")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    f5tts_dir = Path(args.f5tts_dir)
    out = Path(args.out)

    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    import torch

    print(f"Loading checkpoint: {ckpt_path} ({ckpt_path.stat().st_size / 1e9:.2f} GB)")

    if ckpt_path.suffix == ".safetensors":
        from safetensors.torch import load_file as _st_load
        flat_sd = _st_load(str(ckpt_path))
        # IndicF5 safetensors: keys are "ema_model._orig_mod.transformer.…" + vocoder
        # Strip _orig_mod prefix, drop vocoder, strip ema_model prefix.
        checkpoint = {}
        for k, v in flat_sd.items():
            if k.startswith("vocoder."):
                continue
            new_k = k.replace("ema_model._orig_mod.", "ema_model.")
            checkpoint[new_k] = v
        # Wrap into the same format as a .pt checkpoint
        checkpoint = {"ema_model_state_dict": checkpoint}
        print(f"Loaded safetensors: {len(flat_sd)} keys → {len(checkpoint['ema_model_state_dict'])} (stripped vocoder + _orig_mod)")
    else:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "ema_model_state_dict" in checkpoint:
        raw_sd = checkpoint["ema_model_state_dict"]
        print("Using EMA weights")
    elif "model_state_dict" in checkpoint:
        raw_sd = checkpoint["model_state_dict"]
        print("Using model weights (no EMA found)")
    else:
        raise SystemExit(f"Unrecognized checkpoint format. Keys: {list(checkpoint.keys())[:20]}")

    # Strip the "ema_model." prefix — this is how F5-TTS inference expects the keys
    skip_keys = {"initted", "step", "update",
                 "ema_model.mel_spec.mel_stft.mel_scale.fb",
                 "ema_model.mel_spec.mel_stft.spectrogram.window"}
    model_sd = {
        k.replace("ema_model.", ""): v
        for k, v in raw_sd.items()
        if k not in skip_keys
    }
    print(f"State dict: {len(model_sd)} keys")

    out.mkdir(parents=True, exist_ok=True)

    # Save wrapped as {"model_state_dict": ...} so F5-TTS load_checkpoint(use_ema=False) works
    out_ckpt = out / "model.pt"
    torch.save({"model_state_dict": model_sd}, out_ckpt)
    print(f"Saved weights: {out_ckpt} ({out_ckpt.stat().st_size / 1e9:.2f} GB)")

    # Copy vocab.txt from the training data directory
    dataset_name = "movio_tanglish"
    tokenizer = "char"
    vocab_src = f5tts_dir / "data" / f"{dataset_name}_{tokenizer}" / "vocab.txt"
    if vocab_src.exists():
        shutil.copy(vocab_src, out / "vocab.txt")
        print(f"Copied vocab: {vocab_src} ({sum(1 for _ in open(vocab_src))} chars)")
    else:
        # Fall back to downloading IndicF5's vocab directly — NEVER use F5-TTS's
        # English-only vocab (infer/examples/vocab.txt) which has zero Tamil chars.
        print("Training vocab not found — downloading IndicF5 vocab from HF...")
        from huggingface_hub import hf_hub_download
        vocab_dl = hf_hub_download("ai4bharat/IndicF5", "checkpoints/vocab.txt")
        shutil.copy(vocab_dl, out / "vocab.txt")
        print(f"Copied IndicF5 vocab from HF ({sum(1 for _ in open(out / 'vocab.txt'))} chars)")

    # Write a config.json so the inference code knows model architecture
    config = {
        "model_type": "F5TTS_v1",
        "exp_name": "F5TTS_v1_Base",
        "tokenizer": "char",
        "sample_rate": 24000,
        "n_mel_channels": 100,
        "finetune_base": "ai4bharat/IndicF5",
    }
    with open(out / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Wrote config.json")

    # Write a README with loading instructions
    readme = """# movio Tanglish Fine-tuned F5-TTS Model

## Loading for inference (F5-TTS)

```python
import sys
sys.path.insert(0, "/path/to/F5-TTS/src")

import torch
from f5_tts.model import DiT
from f5_tts.infer.utils_infer import load_vocoder, infer_process, preprocess_ref_audio_text

model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
model = DiT(**model_cfg, text_num_embeds=YOUR_VOCAB_SIZE, mel_dim=100)
ckpt = torch.load("model.pt", map_location="cpu")
model.load_state_dict(ckpt)
```

See 06_evaluate.py for a complete inference example using the F5-TTS stack.
"""
    (out / "README.md").write_text(readme)

    print(f"\nExport complete -> {out}")
    print("Contents:", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
