"""01 — Download indicvoices_r Tamil subset (+ optional extras) on Kaggle.

Usage (Kaggle notebook cell or script):
    python training/scripts/01_download_data.py --out /kaggle/working/raw

Downloads:
  - ai4bharat/indicvoices_r  (CC-BY-4.0) Tamil split only
Requires: HF_TOKEN env var for gated datasets/models.

Disk-budget aware: on Kaggle (20 GB working space) the full Tamil subset
(~24 GB) doesn't fit. We list files first and download only up to --max-gb.
"""

import argparse
import fnmatch
import os
import shutil
from pathlib import Path


def _get_free_gb(path: str) -> float:
    """Return free disk space in GB at *path*."""
    st = shutil.disk_usage(path)
    return st.free / (1024 ** 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/kaggle/working/raw")
    ap.add_argument("--language", default="Tamil")
    ap.add_argument(
        "--max-gb", type=float, default=None,
        help="Max GB to download. Default: (free disk - 5) or 15, whichever is smaller.",
    )
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise SystemExit(
            "HF_TOKEN missing. On Kaggle: Add-ons → Secrets → add HF_TOKEN, "
            "then attach it to the notebook."
        )
    from huggingface_hub import HfApi, hf_hub_download

    repo_id = "ai4bharat/indicvoices_r"
    local_dir = out / "indicvoices_r"
    local_dir.mkdir(parents=True, exist_ok=True)

    lang_lower = args.language.lower()
    lang_code = {"tamil": "ta", "hindi": "hi", "bengali": "bn", "telugu": "te",
                 "kannada": "kn", "malayalam": "ml", "gujarati": "gu",
                 "marathi": "mr", "punjabi": "pa"}.get(lang_lower, lang_lower[:2])
    allow = [
        f"*{args.language}*",
        f"*{lang_lower}*",
        f"*{args.language.capitalize()}*",
        f"*/{lang_code}/*",
        f"{lang_code}_*",
        "*.json", "README.md",
        "*.tsv", "*.txt", "*.csv",
    ]

    # Compute disk budget
    free_gb = _get_free_gb(str(out))
    if args.max_gb is not None:
        budget_bytes = int(args.max_gb * 1024 ** 3)
    else:
        budget_gb = min(free_gb - 5.0, 15.0)
        budget_bytes = int(max(budget_gb, 2.0) * 1024 ** 3)
    print(f"Disk free: {free_gb:.1f} GB | download budget: {budget_bytes / 1024**3:.1f} GB")

    # List repo files matching our patterns
    print("Listing repo files...")
    api = HfApi()
    all_files = api.list_repo_tree(
        repo_id=repo_id, repo_type="dataset", token=token, recursive=True,
    )

    # Filter to matching files, collect (path, size)
    matched = []
    for info in all_files:
        if not hasattr(info, "size") or info.size is None:
            continue
        rpath = info.path
        if any(fnmatch.fnmatch(rpath, pat) for pat in allow):
            matched.append((rpath, info.size))

    # Sort: metadata files first (small), then audio files by name
    def _sort_key(item):
        rp, sz = item
        ext = rp.rsplit(".", 1)[-1].lower() if "." in rp else ""
        if ext in ("json", "tsv", "csv", "txt", "md"):
            return (0, rp)
        return (1, rp)

    matched.sort(key=_sort_key)
    total_size = sum(s for _, s in matched)
    print(f"Matched {len(matched)} files totalling {total_size / 1024**3:.1f} GB")

    # Select files within budget
    selected = []
    cumulative = 0
    for rpath, size in matched:
        if cumulative + size > budget_bytes:
            print(f"  budget reached at {cumulative / 1024**3:.1f} GB — "
                  f"skipping remaining {len(matched) - len(selected)} files")
            break
        selected.append(rpath)
        cumulative += size

    if not selected:
        raise SystemExit("No files fit within disk budget — free up space or increase --max-gb")

    print(f"Downloading {len(selected)} files ({cumulative / 1024**3:.1f} GB)...")

    # Download selected files one-by-one (with progress)
    for i, rpath in enumerate(selected):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1}/{len(selected)}] {rpath}")
        try:
            hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=rpath,
                token=token,
                local_dir=str(local_dir),
            )
        except OSError as e:
            if "No space left" in str(e):
                print(f"  disk full at file {i+1}/{len(selected)} — stopping download")
                break
            raise

    path = str(local_dir)
    print(f"Saved to: {path}")

    # Diagnostics
    top = list(Path(path).rglob("*"))[:20]
    print("Top-level sample:", [str(p.relative_to(path)) for p in top[:10]])
    n_wav = (len(list(Path(path).rglob("*.wav")))
             + len(list(Path(path).rglob("*.mp3")))
             + len(list(Path(path).rglob("*.flac"))))
    print(f"Diagnostic wav/mp3/flac count: {n_wav}")
    if n_wav == 0:
        print("WARNING: 0 audio found. Listing all files matched by allow patterns:")
        for p in sorted(Path(path).rglob("*"))[:50]:
            print(" ", p.relative_to(path))


if __name__ == "__main__":
    main()
