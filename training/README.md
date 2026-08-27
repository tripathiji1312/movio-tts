# Training Pipeline — Tanglish LoRA Fine-Tune of IndicF5 (v2 Upgrade)

> **Run this on Kaggle, never locally.** Everything here is a complete,
> cell-by-cell pipeline: corpus download → preparation → dataset build →
> fine-tune → merge/export → evaluation.

## What you are training

The inference stack in `movio/` works out-of-the-box with the stock
`ai4bharat/IndicF5` checkpoint (Apache-2.0) using the Tanglish router for
script unification. This training pipeline produces the **v2 model**: a
LoRA-adapted IndicF5 with production-grade prosody at code-switch
boundaries (blueprint §3.5), trained on:

| Source | License | Hours | Role |
|---|---|---|---|
| **indicvoices_r** Tamil subset | CC-BY-4.0 | ~100–300 usable | Base adaptation |
| **Custom Chennai-region recordings** (4–6 bilingual talents) | Yours | 50–100 | Tanglish switch-point prosody |
| Optional TTS-distillation audio | — | — | Augmentation only |

Target: **50–100 h total**, ~2,000 transport-domain dialogues covering all
three Tanglish forms:
1. Tamil Unicode + Latin English (`உங்கள் pickup location எங்கே?`)
2. Romanized Tamil + English (`Unga pickup location enga?`)
3. Mixed proper nouns (`Chennai Central-ல இருக்கா?`)

## Kaggle setup (one time)

1. Request access to the gated models/datasets on Hugging Face
   (**24–48 h lead time**):
   - `ai4bharat/IndicF5`
   - `ai4bharat/indicvoices_r`
2. Create an HF token (read scope): https://huggingface.co/settings/tokens
3. On Kaggle: **Add-ons → Secrets → Add secret** → name it `HF_TOKEN`.
4. Upload your custom Tanglish recordings as a private Kaggle Dataset
   shaped like: `audio/*.wav` + `transcripts.tsv` (`filename<TAB>text`).

## Run the pipeline

### Option A — all-in-one Kaggle notebook (recommended)

**[`kaggle/movio_kaggle_training.ipynb`](kaggle/movio_kaggle_training.ipynb)**
contains everything: environment checks, HF auth from secrets, git clone /
dataset-attach of this repo, indicvoices_r download, corpus prep, dataset
build, fine-tune, merge/export, evaluation gate, and an in-notebook
base-vs-fine-tuned listening demo.

1. Kaggle → **File → Import Notebook** → upload the `.ipynb`.
2. Settings: Accelerator **GPU T4 x2 / P100**, Internet **ON**, secret `HF_TOKEN`.
3. Edit the config cell (`REPO_URL`, optional `CUSTOM_DATA_INPUT`).
4. **Run All**, then **Save Version → Save & Run All** for a reproducible artifact.

### Option B — script-by-script

```bash
# attach your repo as /kaggle/input/movio first
cd /kaggle/working/movio

python training/scripts/01_download_data.py --out /kaggle/working/raw
python training/scripts/02_prepare_corpus.py \
    --raw-dir /kaggle/working/raw --out /kaggle/working/data \
    --min-dur 1.0 --max-dur 12.0 --min-snr 12.0
python training/scripts/03_build_dataset.py --data-dir /kaggle/working/data
python training/scripts/04_train_lora.py --path f5tts \
    --data-dir /kaggle/working/data --out /kaggle/working/f5tts_out \
    --epochs 6 --batch-size 2
python training/scripts/05_merge_export.py \
    --adapter-dir /kaggle/working/f5tts_out/F5-TTS/ckpts/latest \
    --base-model ai4bharat/IndicF5 \
    --out /kaggle/working/indicf5_tanglish_merged
python training/scripts/06_evaluate.py \
    --model /kaggle/working/indicf5_tanglish_merged \
    --test /kaggle/working/data/test.csv \
    --ref-pool /kaggle/working/data/ref_pool.csv
```

## Stage details

### 01_download_data.py
Pulls only the Tamil split of indicvoices_r via `snapshot_download` allow-
patterns (keeps the download small). Requires `HF_TOKEN`.

### 02_prepare_corpus.py
- Resamples everything to 24 kHz mono PCM16 (IndicF5's native rate)
- Trims silence (`librosa.effects.split`, 40 dB)
- Filters: duration 1–12 s, spectral-flux SNR proxy ≥ 12 dB
- Deduplicates by waveform hash
- Splits train/val/test (94/3/3) and writes `ref_pool.csv` — 200 clips used
  to sample `(ref_audio, ref_text)` zero-shot prompts during training

### 03_build_dataset.py
Builds a Hugging Face Arrow dataset where each item carries text + target
audio + sampled reference pair — exactly the conditioning IndicF5 uses at
inference. Also writes F5-TTS-format `metadata.csv`.

### 04_train_lora.py
Two paths:

| Path | When | How |
|---|---|---|
| `--path f5tts` (**recommended**) | Always | Clones SWivid/F5-TTS (the native trainer family for F5-style CFM models), symlinks your audio into their layout, generates a finetune config, and runs `f5-tts finetune`. Battle-tested; handles mel extraction, batching, EMA. |
| `--path peft` | Experimental | PEFT LoRA directly on the trust_remote_code HF model (rank 32, α 64). Introspects the remote code for a loss entry point and fails loudly rather than silently mis-training if none exists. Use `--list-modules` to inspect module names before editing `target_modules_patterns`. |

Kaggle budget: 6 epochs × 20 h data ≈ **12–20 GPU-hours** on a single T4.
T4 has no bf16 → config uses fp16. If you get A100 (rare on Kaggle),
switch to bf16.

### 05_merge_export.py
`merge_and_unload()` → full safetensors checkpoint saved to
`indicf5_tanglish_merged`. Copy this directory out of `/kaggle/working`
(Save Version) and serve it by pointing `stage_c.model_id` at its local
path in `config/settings.yaml`. Also copy any `*.py` remote-code files from
the base repo next to it so serving works offline.

### 06_evaluate.py
- **UTMOS** objective MOS (SpeechMOS v1.2.0, `utmos22_strong`)
- **WER** intelligibility with `vasista22/whisper-tamil-base` + jiwer

Gate for promotion to production: UTMOS ≥ base-model −0.05 AND WER ≤ base
+2% absolute on the test split, plus human spot-check of 30 Tanglish
utterances across all three script forms.

## After training

```bash
# local: drop the merged model next to the repo and update settings.yaml
stage_c:
  model_id: "/models/indicf5_tanglish_merged"
```

No code changes needed — the engine loads it through the same
`AutoModel.from_pretrained(..., trust_remote_code=True)` path.
