"""04 — LoRA fine-tune of IndicF5 on Tanglish corpus (Kaggle GPU).

Two supported paths:
  --path f5tts   (RECOMMENDED) drives the SWivid/F5-TTS finetuning framework,
                 which is the native trainer family for IndicF5. Uses the
                 metadata.csv produced by 03_build_dataset.py.
  --path peft    experimental custom PEFT LoRA loop against the HF
                 trust_remote_code model. Run with --list-modules first to
                 confirm target-module names in your pinned revision.

Examples (Kaggle):
    python training/scripts/04_train_lora.py --path f5tts \
        --data-dir /kaggle/working/data --out /kaggle/working/f5tts_out

    python training/scripts/04_train_lora.py --path peft \
        --config training/configs/lora_tanglish.yaml --list-modules
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def check_gpu():
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device. Kaggle: Settings → Accelerator → GPU T4/P100.")
    print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")


def ensure_hf_token():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        from kaggle_secrets import UserSecretsClient  # noqa

        try:
            os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            raise SystemExit("Set the HF_TOKEN Kaggle secret first.")


def path_f5tts(data_dir: Path, out_dir: Path, epochs: int, batch_size: int):
    workdir = out_dir / "F5-TTS"
    if not workdir.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/SWivid/F5-TTS.git", str(workdir)],
            check=True,
        )
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(workdir)], check=True)

    import csv as csv_mod

    dataset_name = "movio_tanglish"
    tokenizer = "char"

    # F5-TTS expects data at: data/{dataset_name}_{tokenizer}/raw/ (Arrow) + duration.json + vocab.txt
    ds_dir = workdir / "data" / f"{dataset_name}_{tokenizer}"
    ds_dir.mkdir(parents=True, exist_ok=True)

    # Read train.csv which already has audio paths, text, and duration
    train_csv = data_dir / "train.csv"
    if not train_csv.exists():
        raise SystemExit("train.csv not found. Run 02_prepare_corpus.py first.")

    durations = []
    texts = []
    audio_paths = []
    with open(train_csv, encoding="utf-8") as f:
        for row in csv_mod.DictReader(f):
            audio_path = row["audio"]
            if Path(audio_path).exists():
                durations.append(float(row["duration_s"]))
                texts.append(row["text"])
                audio_paths.append(audio_path)

    print(f"Building Arrow dataset: {len(texts)} utterances, "
          f"{sum(durations)/3600:.1f}h total")

    # Use the pretrained model's vocab and extend with any new Tamil characters.
    # The pretrained F5TTS_v1_Base uses a 2546-char vocab; we must keep it intact
    # and append new chars so the pretrained embeddings remain aligned.
    pretrained_vocab_path = workdir / "src" / "f5_tts" / "infer" / "examples" / "vocab.txt"
    with open(pretrained_vocab_path, "r", encoding="utf-8") as f:
        pretrained_vocab = [line.rstrip("\n") for line in f]
    # First char should be space (idx 0)
    if pretrained_vocab and pretrained_vocab[0] != " ":
        pretrained_vocab = [" "] + pretrained_vocab

    existing_chars = set(pretrained_vocab)
    new_chars = set()
    for t in texts:
        for ch in t:
            if ch not in existing_chars:
                new_chars.add(ch)

    vocab = pretrained_vocab + sorted(new_chars)
    vocab_path = ds_dir / "vocab.txt"
    with open(vocab_path, "w", encoding="utf-8") as f:
        for ch in vocab:
            f.write(ch + "\n")
    print(f"Vocab: {len(vocab)} characters ({len(new_chars)} new Tamil/Tanglish chars added) -> {vocab_path}")

    # Write duration.json
    with open(ds_dir / "duration.json", "w") as f:
        json.dump({"duration": durations}, f)

    # Build and save Arrow dataset
    # F5-TTS CustomDataset expects columns: audio_path, text, duration
    from datasets import Dataset
    ds = Dataset.from_dict({
        "audio_path": audio_paths,
        "text": texts,
        "duration": durations,
    })
    ds.save_to_disk(str(ds_dir / "raw"))
    print(f"Arrow dataset saved to: {ds_dir / 'raw'}")

    # Patch F5-TTS trainer to handle embedding size mismatch.
    # The pretrained checkpoint has text_embed of shape [2546, 512] but our extended
    # vocab creates [2546+N, 512]. We patch the checkpoint loading to resize the
    # embedding tensor by padding with the mean of existing embeddings.
    trainer_py = workdir / "src" / "f5_tts" / "model" / "trainer.py"
    trainer_src = trainer_py.read_text()
    patch_marker = "# MOVIO_PATCHED"
    if patch_marker not in trainer_src:
        # Add a helper function at the top (after imports)
        patch_fn = '''
# MOVIO_PATCHED
def _resize_state_dict_embeddings(state_dict, model_state_dict):
    """Resize text embedding weights in checkpoint to match the model's vocab size."""
    import torch as _torch
    for key in list(state_dict.keys()):
        if "text_embed" in key and "weight" in key and key in model_state_dict:
            ckpt_shape = state_dict[key].shape
            model_shape = model_state_dict[key].shape
            if ckpt_shape != model_shape and ckpt_shape[1] == model_shape[1]:
                old = state_dict[key]
                new = _torch.zeros(model_shape, dtype=old.dtype, device=old.device)
                new[:ckpt_shape[0]] = old
                # Init new embeddings with mean of existing
                new[ckpt_shape[0]:] = old.mean(dim=0, keepdim=True)
                state_dict[key] = new
    return state_dict

'''
        # Insert after the last top-level import
        import_end = trainer_src.rfind("\nimport ")
        if import_end == -1:
            import_end = trainer_src.rfind("\nfrom ")
        insert_pos = trainer_src.index("\n", import_end + 1) + 1
        trainer_src = trainer_src[:insert_pos] + patch_fn + trainer_src[insert_pos:]

        # Patch the two load_state_dict calls to resize first
        trainer_src = trainer_src.replace(
            'self.ema_model.load_state_dict(checkpoint["ema_model_state_dict"])',
            '_resize_state_dict_embeddings(checkpoint["ema_model_state_dict"], self.ema_model.state_dict())\n'
            '            self.ema_model.load_state_dict(checkpoint["ema_model_state_dict"])',
        )
        trainer_src = trainer_src.replace(
            'self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"])',
            '_resize_state_dict_embeddings(checkpoint["model_state_dict"], self.accelerator.unwrap_model(self.model).state_dict())\n'
            '            self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"])',
        )
        trainer_py.write_text(trainer_src)
        print("Patched trainer.py for text embedding resize")

    # Aggressively free disk before training
    import shutil as _shutil
    for cleanup_dir in [
        Path("/kaggle/working/raw"),
        Path("/root/.cache/huggingface/hub"),
        Path("/root/.cache/pip"),
    ]:
        if cleanup_dir.exists():
            _shutil.rmtree(cleanup_dir, ignore_errors=True)
            print(f"Freed disk: removed {cleanup_dir}")
    free_gb = _shutil.disk_usage("/kaggle/working").free / (1024**3)
    print(f"Disk free before training: {free_gb:.1f} GB")
    if free_gb < 12:
        print(f"WARNING: only {free_gb:.1f} GB free — checkpoints are ~5 GB each, training may fail")

    # Run finetune CLI with correct args
    finetune_script = workdir / "src" / "f5_tts" / "train" / "finetune_cli.py"
    cmd = [
        sys.executable, str(finetune_script),
        "--exp_name", "F5TTS_v1_Base",
        "--dataset_name", dataset_name,
        "--tokenizer", tokenizer,
        "--epochs", str(epochs),
        "--batch_size_per_gpu", str(batch_size * 24000 * 10),  # frames: batch * sr * ~10s
        "--batch_size_type", "frame",
        "--max_samples", str(batch_size),
        "--learning_rate", "7e-6",
        "--num_warmup_updates", "200",
        "--save_per_updates", "50000",
        "--last_per_updates", "2000",
        "--keep_last_n_checkpoints", "1",
        "--finetune",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(workdir))

    # Keep only the final checkpoint (model_last.pt) to save disk for export
    ckpt_dir = workdir / "ckpts" / dataset_name
    if ckpt_dir.exists():
        import shutil
        last_pt = ckpt_dir / "model_last.pt"
        if last_pt.exists():
            shutil.copy2(last_pt, out_dir / last_pt.name)
            print(f"Copied checkpoint: {last_pt.name}")
        # Remove all intermediate checkpoints to free disk for merge/export
        for pt in list(ckpt_dir.glob("model_*.pt")):
            if pt.name != "model_last.pt":
                pt.unlink(missing_ok=True)
                print(f"Removed intermediate: {pt.name}")
        safetensors = ckpt_dir / "pretrained_model_1250000.safetensors"
        if safetensors.exists():
            safetensors.unlink(missing_ok=True)
            print("Removed pretrained safetensors (no longer needed)")
    free_gb = _shutil.disk_usage("/kaggle/working").free / (1024**3)
    print(f"Disk free after training: {free_gb:.1f} GB")
    print(f"Checkpoints in: {out_dir}")


def list_modules(model):
    names = [n for n, _ in model.named_modules()]
    interesting = [
        n for n in names
        if re.search(r"(attn|attention|qkv|proj|out|feed|ff|mlp|linear)", n, re.I)
    ]
    print(json.dumps(interesting[:400], indent=1))
    print(f"... total modules: {len(names)}")


def path_peft(config_path: Path, list_only: bool):
    import torch
    import yaml
    from transformers import AutoModel

    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    ensure_hf_token()
    mcfg = cfg["model"]
    model = AutoModel.from_pretrained(mcfg["base_model_id"], trust_remote_code=True)
    model.eval()

    if list_only:
        list_modules(model)
        return

    lcfg = cfg["lora"]
    tcfg = cfg["train"]
    dcfg = cfg["data"]

    all_named = dict(model.named_modules())
    targets = sorted(
        {
            full_name.split(".")[-1]
            for pattern in lcfg["target_modules_patterns"]
            for full_name in all_named
            if re.search(pattern, full_name)
        }
    )
    if not targets:
        print("No modules matched. Re-run with --list-modules and fix "
              "lora.target_modules_patterns in the config.")
        list_modules(model)
        return
    print("LoRA target leaf modules:", targets)

    from peft import LoraConfig, get_peft_model

    lora_cfg = LoraConfig(
        r=lcfg["rank"],
        lora_alpha=lcfg["alpha"],
        lora_dropout=lcfg["dropout"],
        target_modules=targets,
        bias=lcfg.get("bias", "none"),
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    from datasets import load_from_disk

    ds = load_from_disk(dcfg["manifest_train"].replace("/train.csv", "/hf_dataset")) \
        if False else load_from_disk(str(Path(dcfg["manifest_train"]).parent / "hf_dataset"))
    print(f"dataset: {len(ds)} items")

    # NOTE: The exact forward/loss signature lives in IndicF5's remote code.
    # We introspect for a documented training entry point; when present we
    # drive it directly, else we fail loudly with guidance instead of
    # silently producing garbage.
    loss_fn = None
    for attr in ("forward_with_loss", "training_forward", "compute_loss"):
        if hasattr(model, attr):
            loss_fn = getattr(model, attr)
            break
    if loss_fn is None:
        print(
            "\nIndicF5 remote code exposes no obvious single-call training loss.\n"
            "Use --path f5tts (recommended): it trains via the official F5-TTS\n"
            "trainer, which is the native codebase for this architecture."
        )
        sys.exit(2)

    device = "cuda"
    model.to(device)
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=tcfg["learning_rate"],
        weight_decay=tcfg["weight_decay"],
    )
    accum = tcfg["gradient_accumulation_steps"]
    step = 0
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=tcfg["fp16"])
    for epoch in range(tcfg["num_epochs"]):
        for i, batch in enumerate(ds.shuffle(seed=tcfg["seed"]).iter(batch_size=tcfg["batch_size_per_device"])):
            with torch.cuda.amp.autocast(enabled=tcfg["fp16"]):
                loss_dict = loss_fn(batch)
                loss = loss_dict["loss"] / accum
            scaler.scale(loss).backward()
            if (i + 1) % accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["max_grad_norm"])
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % tcfg["logging_steps"] == 0:
                    print(f"epoch={epoch} step={step} loss={loss.item() * accum:.4f}")
                if step % tcfg["save_steps"] == 0:
                    ckpt = Path(tcfg["output_dir"]) / f"step_{step}"
                    model.save_pretrained(str(ckpt))
                    print(f"saved {ckpt}")
    final = Path(tcfg["output_dir"]) / "final"
    model.save_pretrained(str(final))
    print(f"LoRA adapter saved -> {final}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", choices=["f5tts", "peft"], default="f5tts")
    ap.add_argument("--config", default="training/configs/lora_tanglish.yaml")
    ap.add_argument("--data-dir", default="/kaggle/working/data")
    ap.add_argument("--out", default="/kaggle/working/f5tts_out")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--list-modules", action="store_true")
    args = ap.parse_args()

    check_gpu()
    ensure_hf_token()
    Path(args.out).mkdir(parents=True, exist_ok=True)

    if args.path == "f5tts":
        path_f5tts(Path(args.data_dir), Path(args.out), args.epochs, args.batch_size)
    else:
        path_peft(Path(args.config), args.list_modules)


if __name__ == "__main__":
    main()
