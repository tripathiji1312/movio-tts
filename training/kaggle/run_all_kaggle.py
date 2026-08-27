"""run_all_kaggle.py — paste each CELL block into a separate Kaggle notebook cell,
or run the whole file top-to-bottom in a single cell.

Kaggle setup checklist (do these in the UI before running):
  1. Settings → Accelerator → GPU T4 x2 (or P100)
  2. Settings → Internet → ON
  3. Add-ons → Secrets → add `HF_TOKEN` (your Hugging Face token, must have
     been granted access to ai4bharat/IndicF5 — approval takes 24-48h)
  4. File → Import Notebook, or upload this repo as a Kaggle Dataset and
     clone it into /kaggle/working.
"""

import os
import subprocess
import sys

REPO_DIR = "/kaggle/working/movio"
DATA_RAW = "/kaggle/input"          # attach your raw audio as a Kaggle Dataset
WORK = "/kaggle/working"

# ───────────────────────────── CELL 1: env setup ────────────────────────────
CELL_1 = r"""
import os, sys, subprocess
if not os.environ.get("HF_TOKEN"):
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers", "accelerate", "peft", "datasets", "soundfile",
                "librosa", "jiwer", "pyyaml"], check=True)

import torch
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
"""

# ───────────────────────── CELL 2: get repo + data ──────────────────────────
CELL_2 = r"""
import shutil, subprocess, os
REPO_DIR = "/kaggle/working/movio"
if not os.path.exists(REPO_DIR):
    # Option A: your repo is attached as a Kaggle dataset input
    src = "/kaggle/input/movio"
    if os.path.exists(src):
        shutil.copytree(src, REPO_DIR)
    else:
        # Option B: private/public GitHub clone
        subprocess.run(["git", "clone",
                        "https://github.com/<you>/movio.git", REPO_DIR], check=True)
%cd {REPO_DIR}
"""

# ────────────────────────── CELL 3: download corpus ─────────────────────────
CELL_3 = f"""
!python training/scripts/01_download_data.py --out {WORK}/raw
"""

# ────────────────────────── CELL 4: prepare corpus ──────────────────────────
CELL_4 = f"""
!python training/scripts/02_prepare_corpus.py \
    --raw-dir {WORK}/raw \
    --out {WORK}/data \
    --min-dur 1.0 --max-dur 12.0 --min-snr 12.0
"""

# ────────────────────────── CELL 5: build dataset ───────────────────────────
CELL_5 = f"""
!python training/scripts/03_build_dataset.py --data-dir {WORK}/data
"""

# ────────────────────────── CELL 6: fine-tune (F5-TTS path) ─────────────────
CELL_6 = f"""
!python training/scripts/04_train_lora.py --path f5tts \
    --data-dir {WORK}/data --out {WORK}/f5tts_out --epochs 6 --batch-size 2
"""

# ──────────────────────── CELL 7: export merged model ───────────────────────
CELL_7 = f"""
!python training/scripts/05_merge_export.py \
    --adapter-dir {WORK}/f5tts_out/F5-TTS/ckpts/latest \
    --base-model ai4bharat/IndicF5 \
    --out {WORK}/indicf5_tanglish_merged
"""

# ──────────────────────────── CELL 8: evaluate ──────────────────────────────
CELL_8 = f"""
!python training/scripts/06_evaluate.py \
    --model {WORK}/indicf5_tanglish_merged \
    --test {WORK}/data/test.csv \
    --ref-pool {WORK}/data/ref_pool.csv \
    --num-samples 100 --out {WORK}/eval.json
"""

# ─────────────────────── CELL 9: save as Kaggle output ──────────────────────
CELL_9 = f"""
import shutil
# Anything in /kaggle/working is saved when you click "Save Version".
shutil.make_archive("/kaggle/working/movio_ft_model", "zip",
                    root_dir="{WORK}/indicf5_tanglish_merged")
print("Done. Save the notebook version; download movio_ft_model.zip")
"""


def main():
    for i, cell in enumerate(
        [CELL_1, CELL_2, CELL_3, CELL_4, CELL_5, CELL_6, CELL_7, CELL_8, CELL_9], 1
    ):
        print(f"\n{'=' * 70}\nCELL {i}\n{'=' * 70}")
        print(cell.strip())


if __name__ == "__main__":
    main()
