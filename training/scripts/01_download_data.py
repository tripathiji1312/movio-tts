"""01 — Download training data for Tamil TTS fine-tuning on Kaggle.

Supported datasets (choose via --dataset):
  rasa          ai4bharat/Rasa Tamil split — RECOMMENDED
                Studio-recorded by professional voice actors. Clean, controlled
                acoustics. ~20 GB Tamil train split. This is one of the two
                datasets IndicF5 itself was trained on.
                License: CC BY 4.0

  indicvoices_r ai4bharat/indicvoices_r Tamil split
                Crowd-sourced field recordings. Lower audio quality (UTMOS ~2.1
                vs base model's ~2.27) — fine-tuning on this data alone causes
                quality degradation. Use with --quality-top-pct 30 in step 02.
                License: CC BY 4.0

  both          Download Rasa first (within budget), then indicvoices_r with
                whatever disk space remains. Gives speaker diversity while
                keeping Rasa's studio quality as the majority of the corpus.

Usage (Kaggle):
    python training/scripts/01_download_data.py --out /kaggle/working/raw

Both datasets are gated on Hugging Face — request access before running:
  https://huggingface.co/datasets/ai4bharat/Rasa
  https://huggingface.co/datasets/ai4bharat/indicvoices_r
"""

import argparse
import fnmatch
import os
import shutil
from pathlib import Path


def _get_free_gb(path: str) -> float:
    return shutil.disk_usage(path).free / (1024 ** 3)


def _download_files(
    repo_id: str,
    patterns: list,
    local_dir: Path,
    token: str,
    budget_bytes: int,
    label: str,
) -> int:
    """Download files matching *patterns* from *repo_id* within *budget_bytes*.

    Returns bytes actually downloaded.
    """
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    print(f"\n[{label}] Listing repo files...")
    all_files = api.list_repo_tree(
        repo_id=repo_id, repo_type="dataset", token=token, recursive=True,
    )

    matched = []
    for info in all_files:
        if not hasattr(info, "size") or info.size is None:
            continue
        rpath = info.path
        if any(fnmatch.fnmatch(rpath, pat) for pat in patterns):
            matched.append((rpath, info.size))

    # Metadata first, then audio parquets
    def _sort_key(item):
        rp, sz = item
        ext = rp.rsplit(".", 1)[-1].lower() if "." in rp else ""
        return (0, rp) if ext in ("json", "tsv", "csv", "txt", "md") else (1, rp)

    matched.sort(key=_sort_key)
    total_size = sum(s for _, s in matched)
    print(f"[{label}] Matched {len(matched)} files — {total_size / 1024**3:.1f} GB total")

    # Select within budget
    selected, cumulative = [], 0
    for rpath, size in matched:
        if cumulative + size > budget_bytes:
            print(f"[{label}]   budget reached at {cumulative / 1024**3:.1f} GB — "
                  f"skipping {len(matched) - len(selected)} remaining files")
            break
        selected.append((rpath, size))
        cumulative += size

    if not selected:
        print(f"[{label}] WARNING: no files fit within budget ({budget_bytes/1024**3:.1f} GB)")
        return 0

    # Remove corrupted partial files from prior runs
    size_map = {rp: sz for rp, sz in matched}
    corrupted = 0
    for rpath, _ in selected:
        local_file = local_dir / rpath
        if local_file.exists():
            actual = local_file.stat().st_size
            if actual != size_map.get(rpath, actual):
                local_file.unlink()
                corrupted += 1
    if corrupted:
        cache_dir = local_dir / ".cache"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        print(f"[{label}]   removed {corrupted} corrupted/partial files from prior run")

    print(f"[{label}] Downloading {len(selected)} files ({cumulative / 1024**3:.1f} GB)...")
    downloaded = 0
    for i, (rpath, size) in enumerate(selected):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"[{label}]   [{i+1}/{len(selected)}] {rpath}")
        try:
            hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=rpath,
                token=token,
                local_dir=str(local_dir),
            )
            downloaded += size
        except OSError as e:
            if "No space left" in str(e):
                print(f"[{label}]   disk full at file {i+1}/{len(selected)} — stopping")
                break
            raise

    cache_dir = local_dir / ".cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)

    return downloaded


def download_rasa(out: Path, token: str, budget_bytes: int) -> None:
    """Download ai4bharat/Rasa Tamil train split (studio quality, CC BY 4.0)."""
    local_dir = out / "rasa"
    local_dir.mkdir(parents=True, exist_ok=True)

    # Rasa Tamil train split: Tamil/train-*.parquet
    # We skip the test split (3.6k examples) — we already have our own test split.
    patterns = [
        "Tamil/train-*.parquet",
        "README.md",
    ]

    downloaded = _download_files(
        repo_id="ai4bharat/Rasa",
        patterns=patterns,
        local_dir=local_dir,
        token=token,
        budget_bytes=budget_bytes,
        label="Rasa",
    )

    _print_diagnostics(local_dir, "Rasa")
    print(f"[Rasa] Saved to: {local_dir}")


def download_indicvoices_r(out: Path, token: str, budget_bytes: int, language: str = "Tamil") -> None:
    """Download ai4bharat/indicvoices_r Tamil subset (crowd-sourced, CC BY 4.0)."""
    local_dir = out / "indicvoices_r"
    local_dir.mkdir(parents=True, exist_ok=True)

    lang_lower = language.lower()
    lang_code = {
        "tamil": "ta", "hindi": "hi", "bengali": "bn", "telugu": "te",
        "kannada": "kn", "malayalam": "ml", "gujarati": "gu",
        "marathi": "mr", "punjabi": "pa",
    }.get(lang_lower, lang_lower[:2])

    patterns = [
        f"*{language}*",
        f"*{lang_lower}*",
        f"*{language.capitalize()}*",
        f"*/{lang_code}/*",
        f"{lang_code}_*",
        "*.json", "README.md", "*.tsv", "*.txt", "*.csv",
    ]

    downloaded = _download_files(
        repo_id="ai4bharat/indicvoices_r",
        patterns=patterns,
        local_dir=local_dir,
        token=token,
        budget_bytes=budget_bytes,
        label="indicvoices_r",
    )

    _print_diagnostics(local_dir, "indicvoices_r")
    print(f"[indicvoices_r] Saved to: {local_dir}")


def _print_diagnostics(path: Path, label: str) -> None:
    n_parquet = len(list(path.rglob("*.parquet")))
    n_wav = (len(list(path.rglob("*.wav")))
             + len(list(path.rglob("*.mp3")))
             + len(list(path.rglob("*.flac"))))
    print(f"[{label}] parquet files: {n_parquet} | standalone audio files: {n_wav}")
    if n_parquet > 0:
        print(f"[{label}] Audio is embedded in parquets — 02_prepare_corpus.py will extract it.")
    elif n_wav == 0:
        print(f"[{label}] WARNING: 0 audio found. Listing files:")
        for p in sorted(path.rglob("*"))[:30]:
            print("  ", p.relative_to(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/kaggle/working/raw")
    ap.add_argument(
        "--dataset", default="rasa",
        choices=["rasa", "indicvoices_r", "both"],
        help=(
            "rasa: studio-quality Tamil (recommended, default). "
            "indicvoices_r: crowd-sourced Tamil. "
            "both: Rasa first (primary budget), then indicvoices_r with remaining space."
        ),
    )
    ap.add_argument("--language", default="Tamil",
                    help="Language name (for indicvoices_r only; Rasa is always Tamil config)")
    ap.add_argument(
        "--max-gb", type=float, default=None,
        help="Total download budget in GB. Default: (free disk − 5 GB) capped at 15 GB.",
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

    free_gb = _get_free_gb(str(out))
    if args.max_gb is not None:
        total_budget = int(args.max_gb * 1024 ** 3)
    else:
        budget_gb = min(free_gb - 5.0, 15.0)
        total_budget = int(max(budget_gb, 2.0) * 1024 ** 3)
    print(f"Disk free: {free_gb:.1f} GB | download budget: {total_budget / 1024**3:.1f} GB")
    print(f"Dataset(s): {args.dataset}")

    if args.dataset == "rasa":
        download_rasa(out, token, total_budget)

    elif args.dataset == "indicvoices_r":
        download_indicvoices_r(out, token, total_budget, args.language)

    elif args.dataset == "both":
        # Give Rasa ~60% of the budget (it's the quality anchor), indicvoices_r the rest.
        # If Rasa uses less than its share, the remainder rolls to indicvoices_r.
        rasa_budget = int(total_budget * 0.6)
        indicvoices_budget = total_budget - rasa_budget

        # Track actual disk use to be precise about remaining budget
        disk_before = shutil.disk_usage(str(out)).used
        download_rasa(out, token, rasa_budget)
        disk_after_rasa = shutil.disk_usage(str(out)).used
        rasa_used = max(0, disk_after_rasa - disk_before)

        remaining = max(0, total_budget - rasa_used)
        remaining = min(remaining, indicvoices_budget)
        free_now = _get_free_gb(str(out))
        remaining = min(remaining, int((free_now - 3.0) * 1024 ** 3))

        if remaining > 512 * 1024 * 1024:  # at least 512 MB
            print(f"\nRasa used {rasa_used/1024**3:.1f} GB — "
                  f"{remaining/1024**3:.1f} GB remaining for indicvoices_r")
            download_indicvoices_r(out, token, remaining, args.language)
        else:
            print(f"\nNot enough space for indicvoices_r "
                  f"({remaining/1024**3:.1f} GB left < 0.5 GB minimum) — skipping")

    print("\nDownload complete.")
    total_parquet = len(list(out.rglob("*.parquet")))
    total_wav = (len(list(out.rglob("*.wav")))
                 + len(list(out.rglob("*.mp3")))
                 + len(list(out.rglob("*.flac"))))
    print(f"Total: {total_parquet} parquet files | {total_wav} standalone audio files")
    if total_parquet == 0 and total_wav == 0:
        raise SystemExit(
            "Download produced no data — check HF_TOKEN and gated dataset access:\n"
            "  Rasa: https://huggingface.co/datasets/ai4bharat/Rasa\n"
            "  indicvoices_r: https://huggingface.co/datasets/ai4bharat/indicvoices_r"
        )


if __name__ == "__main__":
    main()
