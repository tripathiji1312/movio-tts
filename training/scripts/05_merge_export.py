"""05 — Merge LoRA adapter into a full checkpoint and export for serving.

Produces a self-contained model directory loadable by the inference stack:
    movio.acoustic.indicf5_engine.IndicF5Engine (point stage_c.model_id at it)
"""

import argparse
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-dir", required=True, help="LoRA adapter dir from step 04")
    ap.add_argument("--base-model", default="ai4bharat/IndicF5")
    ap.add_argument("--out", default="/kaggle/working/indicf5_tanglish_merged")
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModel

    print("Loading base...")
    # transformers uses init_empty_weights() (meta tensors) during __init__ when
    # low_cpu_mem_usage=True (the default). IndicF5's __init__ builds a Vocos
    # vocoder which calls torchaudio.transforms.MelSpectrogram → .any() → crash
    # on meta tensors. low_cpu_mem_usage=False uses real CPU tensors throughout.
    base = AutoModel.from_pretrained(
        args.base_model, trust_remote_code=True, low_cpu_mem_usage=False
    )

    print("Attaching adapter...")
    model = PeftModel.from_pretrained(base, args.adapter_dir)

    print("Merging weights...")
    merged = model.merge_and_unload()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out), safe_serialization=True)
    print(f"Merged model -> {out}")

    src_code = Path(args.base_model)
    print(
        "NOTE: if the base was loaded via trust_remote_code from the Hub, also copy\n"
        "the .py remote-code files next to the export so serving works offline."
    )


if __name__ == "__main__":
    main()
