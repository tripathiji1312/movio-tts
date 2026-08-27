"""01 — Download IndicVoices-R Tamil subset (+ optional extras) on Kaggle.

Usage (Kaggle notebook cell or script):
    python training/scripts/01_download_data.py --out /kaggle/working/raw

Downloads:
  - ai4bharat/IndicVoices-R  (CC-BY-4.0) Tamil split only
Requires: HF_TOKEN env var for gated datasets/models.
"""

import argparse
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/kaggle/working/raw")
    ap.add_argument("--language", default="Tamil")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise SystemExit(
            "HF_TOKEN missing. On Kaggle: Add-ons → Secrets → add HF_TOKEN, "
            "then attach it to the notebook."
        )
    from huggingface_hub import snapshot_download

    print("Downloading IndicVoices-R (this can take a while)...")
    path = snapshot_download(
        repo_id="ai4bharat/IndicVoices-R",
        repo_type="dataset",
        token=token,
        allow_patterns=[f"*{args.language}*", "*.json", "README.md"],
        local_dir=str(out / "indicvoices_r"),
    )
    print(f"Saved to: {path}")


if __name__ == "__main__":
    main()
