"""01 — Download indicvoices_r Tamil subset (+ optional extras) on Kaggle.

Usage (Kaggle notebook cell or script):
    python training/scripts/01_download_data.py --out /kaggle/working/raw

Downloads:
  - ai4bharat/indicvoices_r  (CC-BY-4.0) Tamil split only
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

    print("Downloading indicvoices_r (this can take a while)...")
    # HF allow_patterns are fnmatch against repo paths (case-sensitive).
    # Dataset uses lower-case folder names like "tamil" for some releases,
    # capitalised "Tamil" for others, and sometimes code "ta".
    # We include all variants so wavs like "tamil/audio_001.wav" are matched.
    lang_lower = args.language.lower()
    lang_code = {"tamil": "ta", "hindi": "hi", "bengali": "bn", "telugu": "te",
                 "kannada": "kn", "malayalam": "ml", "gujarati": "gu",
                 "marathi": "mr", "punjabi": "pa"}.get(lang_lower, lang_lower[:2])
    allow = [
        f"*{args.language}*",      # Tamil
        f"*{lang_lower}*",         # tamil
        f"*{args.language.capitalize()}*",
        f"*/{lang_code}/*",        # ta/  (repo uses lang-code folders in some versions)
        f"{lang_code}_*",
        "*.json", "README.md",
        "*.tsv", "*.txt", "*.csv",
    ]
    try:
        path = snapshot_download(
            repo_id="ai4bharat/indicvoices_r",
            repo_type="dataset",
            token=token,
            allow_patterns=allow,
            local_dir=str(out / "indicvoices_r"),
        )
    except Exception as exc:
        print(f"snapshot_download failed: {exc}")
        print("Hint: check HF_TOKEN has gated access to ai4bharat/indicvoices_r "
              "(request on https://huggingface.co/datasets/ai4bharat/indicvoices_r )")
        raise
    print(f"Saved to: {path}")
    # quick diagnostics for next cell
    import glob
    top = list((Path(path)).rglob("*"))[:20]
    print("Top-level sample:", [str(p.relative_to(path)) for p in top[:10]])
    n_wav = len(list((Path(path)).rglob("*.wav"))) + len(list((Path(path)).rglob("*.mp3"))) + len(list((Path(path)).rglob("*.flac")))
    print(f"Diagnostic wav/mp3/flac count: {n_wav}")
    if n_wav == 0:
        print("WARNING: 0 audio found. Listing all files matched by allow patterns:")
        for p in sorted((Path(path)).rglob("*"))[:50]:
            print(" ", p.relative_to(path))


if __name__ == "__main__":
    main()
