"""Download pre-trained Tamil TTS models for Solution 4.

Downloads:
  1. samprabin/tamil_vits — VITS Tamil (MIT license, Coqui TTS format)
  2. smtiitm/FastSpeech2_HS — FastSpeech2 + HiFi-GAN (MIT, IIT Madras)

Usage:
    python scripts/download_models.py
    python scripts/download_models.py --models vits
    python scripts/download_models.py --models fastspeech2
"""

import argparse
import shutil
from pathlib import Path

MODELS_DIR = Path("models")

VITS_REPO = "samprabin/tamil_vits"
VITS_DIR = MODELS_DIR / "vits" / "tamil_vits"

FS2_REPO = "smtiitm/FastSpeech2_HS"
FS2_DIR = MODELS_DIR / "fastspeech2" / "tamil_fs2"
HIFIGAN_DIR = MODELS_DIR / "fastspeech2" / "hifigan"


def download_vits():
    """Download VITS Tamil model from HuggingFace."""
    print(f"\n{'='*60}")
    print(f"Downloading VITS Tamil model ({VITS_REPO})")
    print(f"{'='*60}")

    VITS_DIR.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=VITS_REPO,
        local_dir=str(VITS_DIR),
        ignore_patterns=["*.md", ".gitattributes"],
    )
    print(f"Downloaded to: {path}")

    # Normalize: Coqui expects model_file.pth + config.json
    best = VITS_DIR / "best_model.pth"
    model = VITS_DIR / "model.pth"
    if best.exists() and not model.exists():
        shutil.copy2(best, model)
        print(f"  Copied best_model.pth → model.pth")

    _verify_dir(VITS_DIR, "VITS")


def download_fastspeech2():
    """Download FastSpeech2 + HiFi-GAN Tamil models."""
    print(f"\n{'='*60}")
    print(f"Downloading FastSpeech2+HiFi-GAN ({FS2_REPO})")
    print(f"{'='*60}")

    FS2_DIR.mkdir(parents=True, exist_ok=True)
    HIFIGAN_DIR.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=FS2_REPO,
        local_dir=str(FS2_DIR),
        ignore_patterns=["*.md", ".gitattributes"],
    )
    print(f"Downloaded to: {path}")

    # Organize: move vocoder files to hifigan dir
    for pattern in ["*hifigan*", "*vocoder*", "*generator*"]:
        for f in FS2_DIR.rglob(pattern):
            if f.is_file() and HIFIGAN_DIR not in f.parents:
                dest = HIFIGAN_DIR / f.name
                if not dest.exists():
                    shutil.copy2(f, dest)
                    print(f"  Vocoder: {f.name} → hifigan/")

    # If there's a Tamil/ subdirectory, lift its contents
    tamil_dir = FS2_DIR / "Tamil"
    if tamil_dir.exists() and tamil_dir.is_dir():
        for f in tamil_dir.iterdir():
            if f.is_file():
                dest = FS2_DIR / f.name
                if not dest.exists():
                    shutil.copy2(f, dest)
        print(f"  Lifted Tamil/ subdirectory contents")

    _verify_dir(FS2_DIR, "FastSpeech2")
    _verify_dir(HIFIGAN_DIR, "HiFi-GAN")


def _verify_dir(d: Path, name: str):
    if not d.exists():
        print(f"  WARNING: {name} directory missing!")
        return
    files = list(d.iterdir())
    total_mb = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e6
    pth_files = [f.name for f in files if f.suffix == ".pth"]
    json_files = [f.name for f in files if f.suffix == ".json"]
    print(f"  {name}: {total_mb:.1f} MB | .pth={pth_files} | .json={json_files}")


def print_summary():
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    vits_ok = any(VITS_DIR.glob("*.pth")) if VITS_DIR.exists() else False
    fs2_ok = FS2_DIR.exists() and any(FS2_DIR.rglob("*.pth"))

    print(f"  VITS (T1):       {'OK' if vits_ok else 'MISSING'}")
    print(f"  FastSpeech2 (T2): {'OK' if fs2_ok else 'MISSING'}")

    if vits_ok:
        print("\nReady! Start server with: python -m movio")
    else:
        print("\nCheck errors above.")


def main():
    ap = argparse.ArgumentParser(description="Download pre-trained Tamil TTS models")
    ap.add_argument(
        "--models", choices=["all", "vits", "fastspeech2"], default="all",
    )
    args = ap.parse_args()

    if args.models in ("all", "vits"):
        download_vits()
    if args.models in ("all", "fastspeech2"):
        download_fastspeech2()

    print_summary()


if __name__ == "__main__":
    main()
