"""04 — Fine-tune ai4bharat/IndicF5 on Tanglish corpus (Kaggle GPU).

Two supported paths:
  --path f5tts   (RECOMMENDED) drives the SWivid/F5-TTS finetuning framework
                 starting from ai4bharat/IndicF5 weights (MIT licensed).
  --path peft    experimental custom PEFT LoRA loop against the HF
                 trust_remote_code model. Run with --list-modules first to
                 confirm target-module names in your pinned revision.

License note:
  Base model: ai4bharat/IndicF5 (MIT) — commercially usable fine-tuned weights.
  Fine-tuning framework code: SWivid/F5-TTS (CC BY-NC 4.0) — training-time only.

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
                texts.append(row["text"].replace("\n", " ").replace("\r", " ").strip())
                audio_paths.append(audio_path)

    print(f"Building Arrow dataset: {len(texts)} utterances, "
          f"{sum(durations)/3600:.1f}h total")

    # Download ai4bharat/IndicF5 base weights (MIT — commercially usable).
    # IndicF5 is the better starting point: already knows Tamil phonemes + script,
    # same CFM-DiT architecture as F5-TTS, and MIT-licensed for commercial use.
    from huggingface_hub import hf_hub_download, snapshot_download
    import shutil as _shutil

    indicf5_ckpt_dir = out_dir / "indicf5_base"
    indicf5_ckpt_dir.mkdir(parents=True, exist_ok=True)

    # IndicF5 on HF is stored as a safetensors model — download model + vocab
    print("Downloading ai4bharat/IndicF5 weights (MIT) ...")
    try:
        indicf5_model_file = hf_hub_download(
            "ai4bharat/IndicF5", filename="model.safetensors",
            local_dir=str(indicf5_ckpt_dir),
        )
    except Exception:
        # Fallback: full snapshot (includes vocab.txt and config)
        snapshot_download("ai4bharat/IndicF5", local_dir=str(indicf5_ckpt_dir))
        indicf5_model_file = str(indicf5_ckpt_dir / "model.safetensors")

    # IndicF5 was saved with torch.compile() — keys have "._orig_mod." prefix
    # and the vocoder is bundled in the same file. Strip both so F5-TTS trainer
    # can load it with load_checkpoint().
    converted_file = indicf5_ckpt_dir / "model_converted.safetensors"
    _needs_conversion = not converted_file.exists()
    if not _needs_conversion:
        # Re-convert if the file is missing the EMA metadata buffers added in a later fix
        from safetensors.torch import load_file as _st_check
        _keys = set(_st_check(str(converted_file)).keys())
        if "initted" not in _keys or "step" not in _keys:
            print("Re-converting: existing file missing EMA metadata buffers (initted/step)")
            _needs_conversion = True
    if _needs_conversion:
        print("Converting IndicF5 weights (strip _orig_mod prefix + vocoder keys)...")
        from safetensors.torch import load_file as _st_load, save_file as _st_save
        raw = _st_load(indicf5_model_file)
        converted = {}
        for k, v in raw.items():
            # Skip vocoder weights entirely (not needed for the DiT backbone)
            if k.startswith("vocoder."):
                continue
            # Strip torch.compile prefix from EMA model keys
            new_k = k.replace("ema_model._orig_mod.", "ema_model.")
            converted[new_k] = v
        # Add EMA metadata buffers — the trainer's EMA wrapper (ema_pytorch) requires
        # these in the state dict. IndicF5 was saved without them.
        import torch as _torch
        converted["initted"] = _torch.tensor(True)
        converted["step"] = _torch.tensor(0, dtype=_torch.long)
        _st_save(converted, str(converted_file))
        print(f"Converted: {len(raw)} keys → {len(converted)} keys → {converted_file}")
    indicf5_model_file = str(converted_file)

    # Free the original (1.4 GB) now that we have the converted copy.
    orig_safetensors = indicf5_ckpt_dir / "model.safetensors"
    if orig_safetensors.exists():
        orig_safetensors.unlink()
        print(f"Freed {orig_safetensors.name} (~1.4 GB) — replaced by converted copy")

    # Clean HF download cache immediately — may contain another 1.4 GB copy
    for _cache in [Path("/root/.cache/huggingface/hub"),
                   Path.home() / ".cache" / "huggingface" / "hub"]:
        if _cache.exists():
            _shutil.rmtree(_cache, ignore_errors=True)
            print(f"Freed HF cache: {_cache}")

    free_gb = _shutil.disk_usage("/kaggle/working").free / (1024**3)
    print(f"Disk free after weight prep: {free_gb:.1f} GB")

    # IndicF5 ships its own vocab — use it as the base vocab so Tamil chars are
    # already present without needing to resize embeddings.
    indicf5_vocab_candidates = [
        indicf5_ckpt_dir / "vocab.txt",
        indicf5_ckpt_dir / "src" / "f5_tts" / "infer" / "examples" / "vocab.txt",
    ]
    indicf5_vocab_file = None
    for cand in indicf5_vocab_candidates:
        if cand.exists():
            indicf5_vocab_file = cand
            break
    if indicf5_vocab_file is None:
        # IndicF5 doesn't ship a standalone vocab.txt — fall back to F5-TTS default
        # (same 2546-char vocab; Tamil chars will be appended as new_chars)
        indicf5_vocab_file = workdir / "src" / "f5_tts" / "infer" / "examples" / "vocab.txt"
    print(f"Using vocab: {indicf5_vocab_file}")

    # Use the base vocab and extend with any new Tamil characters.
    # IndicF5's vocab already covers Tamil Unicode; new_chars will likely be 0.
    pretrained_vocab_path = indicf5_vocab_file
    with open(pretrained_vocab_path, "r", encoding="utf-8") as f:
        pretrained_vocab = [line.rstrip("\n") for line in f]
    # Ensure space is at index 0 (F5-TTS asserts vocab_char_map[" "] == 0).
    # Deduplicate while preserving order so indices stay stable.
    seen = set()
    deduped = []
    for ch in pretrained_vocab:
        if ch not in seen:
            seen.add(ch)
            deduped.append(ch)
    pretrained_vocab = deduped
    if not pretrained_vocab or pretrained_vocab[0] != " ":
        pretrained_vocab = [" "] + [ch for ch in pretrained_vocab if ch != " "]

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

    # Pre-resize the text embedding in the converted safetensors to match the final
    # vocab size. The --pretrain loading path in finetune_cli.py loads directly from
    # safetensors before model instantiation, so the trainer patch (which only covers
    # the checkpoint-resume path) does not help here. Writing the right shape up front
    # avoids the CUDA gather OOB when the first batch containing new Tamil chars is hit.
    if new_chars:
        from safetensors.torch import load_file as _st_load, save_file as _st_save
        import torch as _torch
        sd = _st_load(str(converted_file))
        # Target specifically the nn.Embedding weight (2D), not conv weights (3D) that
        # also live under the text_embed module. The Embedding key ends with
        # ".text_embed.weight" while conv weights end with ".convs.N.weight".
        embed_key = next(
            (k for k in sd if k.endswith(".text_embed.weight") and len(sd[k].shape) == 2),
            None,
        )
        if embed_key is None:
            print(f"WARNING: no text_embed.weight (2D) found in {converted_file.name} — resize skipped")
        if embed_key:
            old_emb = sd[embed_key]
            old_n, dim = old_emb.shape
            # DiT's TextEmbedding uses nn.Embedding(text_num_embeds + 1, ...) — the +1
            # is a null/padding slot. get_tokenizer returns len(vocab_txt_lines) as
            # vocab_size, and DiT adds +1, so the actual embedding rows = len(vocab)+1.
            new_n = len(vocab) + 1
            if old_n < new_n:
                new_emb = _torch.zeros(new_n, dim, dtype=old_emb.dtype)
                new_emb[:old_n] = old_emb
                new_emb[old_n:] = old_emb.mean(dim=0, keepdim=True)
                sd[embed_key] = new_emb
                _st_save(sd, str(converted_file))
                print(f"Resized text_embed in converted checkpoint: [{old_n}, {dim}] → [{new_n}, {dim}]")

    # Pre-place the converted file at the exact path finetune_cli.py would copy it to.
    # finetune_cli computes: workdir/ckpts/{dataset_name}/pretrained_{basename(pretrain)}.
    # If that file already exists, it skips shutil.copy2, saving ~1.3 GB of I/O.
    # Must be done AFTER any resize above so the link points to the final file content.
    ckpts_dir = workdir / "ckpts" / dataset_name
    ckpts_dir.mkdir(parents=True, exist_ok=True)
    pretrained_dest = ckpts_dir / f"pretrained_{converted_file.name}"
    # Remove stale link/file if the embedding was resized (inode unchanged for hard-link,
    # but we unconditionally recreate so the dest always reflects the current source).
    if pretrained_dest.exists() or pretrained_dest.is_symlink():
        pretrained_dest.unlink()
    try:
        os.link(str(converted_file), str(pretrained_dest))
        print(f"Hard-linked converted model → {pretrained_dest} (zero extra disk)")
    except OSError:
        pretrained_dest.symlink_to(converted_file.resolve())
        print(f"Symlinked converted model → {pretrained_dest}")

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

    # Patch __init__.py to make Trainer import optional — it triggers
    # dataset.py → datasets 5.0.0 → huggingface_hub version mismatch on Kaggle
    init_py = workdir / "src" / "f5_tts" / "model" / "__init__.py"
    init_src = init_py.read_text()
    if "try:" not in init_src and "from f5_tts.model.trainer import Trainer" in init_src:
        init_src = init_src.replace(
            "from f5_tts.model.trainer import Trainer",
            "try:\n    from f5_tts.model.trainer import Trainer\nexcept ImportError:\n    Trainer = None",
        )
        init_py.write_text(init_src)
        print("Patched model/__init__.py for optional Trainer import")

    # Aggressively free disk before training — audio WAVs are still needed by the
    # Arrow dataset (audio_path references), so we cannot delete data/audio/.
    import shutil as _shutil
    for cleanup_dir in [
        Path("/kaggle/working/raw"),
        Path("/root/.cache/huggingface/hub"),
        Path("/root/.cache/pip"),
        Path.home() / ".cache" / "huggingface" / "hub",
    ]:
        if cleanup_dir.exists():
            _shutil.rmtree(cleanup_dir, ignore_errors=True)
            print(f"Freed disk: removed {cleanup_dir}")
    free_gb = _shutil.disk_usage("/kaggle/working").free / (1024**3)
    print(f"Disk free before training: {free_gb:.1f} GB")
    if free_gb < 3:
        raise SystemExit(f"Only {free_gb:.1f} GB free — need ≥3 GB for training checkpoints. Abort.")
    elif free_gb < 5:
        print(f"WARNING: only {free_gb:.1f} GB free — checkpoints are ~1.5 GB each, may be tight")

    # Run finetune CLI starting from ai4bharat/IndicF5 weights (MIT).
    # --pretrain overrides the default SWivid pretrained download so
    # the derived checkpoint is MIT-compatible for commercial use.
    finetune_script = workdir / "src" / "f5_tts" / "train" / "finetune_cli.py"
    cmd = [
        sys.executable, str(finetune_script),
        "--exp_name", "F5TTS_v1_Base",
        "--dataset_name", dataset_name,
        "--tokenizer", tokenizer,
        "--pretrain", str(indicf5_model_file),   # ← IndicF5 base (MIT, not SWivid CC BY-NC)
        "--epochs", str(epochs),
        # batch_size_per_gpu is a mel-frame budget (hop_length=256, sr=24000 → 93.75 fps).
        # batch_size arg here controls utterances-per-batch; 8s average × fps × utts.
        "--batch_size_per_gpu", str(int(batch_size * 8 * 24000 / 256)),
        "--batch_size_type", "frame",
        # max_samples caps sequences per batch (not a length filter). Default 64 is fine;
        # match to batch_size so one batch = exactly batch_size utterances.
        "--max_samples", str(max(batch_size, 8)),
        "--learning_rate", "5e-6",
        "--num_warmup_updates", "200",
        "--save_per_updates", "50000",
        "--last_per_updates", "2000",
        "--keep_last_n_checkpoints", "1",
        "--finetune",
    ]
    # Validate that every character in every utterance maps to a valid vocab index.
    # list_str_to_idx uses vocab_char_map.get(c, 0) — unknowns silently become 0 (space).
    # After +1 shift in TextEmbedding, max index = vocab_size = len(vocab).
    # Embedding table size = vocab_size + 1, so valid lookup range is [0, vocab_size].
    with open(vocab_path, "r", encoding="utf-8") as _vf:
        _vocab_char_map = {line.rstrip("\n"): i for i, line in enumerate(_vf)}
    _vocab_size = len(_vocab_char_map)
    _max_idx = max((_vocab_char_map.get(c, 0) for t in texts for c in t), default=0)
    _missing = set(c for t in texts for c in t if c not in _vocab_char_map)
    # After TextEmbedding's +1 shift, valid lookup range is [0, vocab_size] (table has vocab_size+1 rows).
    # Vocab indices are 0-based so the max valid index is vocab_size-1; unknowns use 0.
    print(f"Vocab validation: size={_vocab_size}, max_token_idx={_max_idx} (embed table={_vocab_size+1} rows)")
    if _missing:
        print(f"WARNING: {len(_missing)} chars not in vocab (will map to idx 0/space): {sorted(_missing)[:20]}")
    assert _max_idx <= _vocab_size - 1, f"Token index {_max_idx} >= vocab_size {_vocab_size} — OOB!"

    env = {**os.environ, "CUDA_LAUNCH_BLOCKING": "1"}
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(workdir), env=env)

    # Keep only the final checkpoint (model_last.pt) to save disk for export.
    # F5-TTS finetune_cli saves under ckpts/{exp_name}/, not ckpts/{dataset_name}/.
    exp_name = "F5TTS_v1_Base"
    ckpt_dir = workdir / "ckpts" / exp_name
    if not ckpt_dir.exists():
        # Fallback: older F5-TTS versions key by dataset_name
        ckpt_dir_fallback = workdir / "ckpts" / dataset_name
        if ckpt_dir_fallback.exists():
            ckpt_dir = ckpt_dir_fallback
        else:
            raise SystemExit(
                f"No checkpoint directory found after training.\n"
                f"Checked: {workdir / 'ckpts' / exp_name}\n"
                f"     and: {workdir / 'ckpts' / dataset_name}"
            )
    import shutil
    last_pt = ckpt_dir / "model_last.pt"
    if not last_pt.exists():
        # model_last.pt is only written every last_per_updates steps; if training ended
        # before the first save, fall back to the most recent numbered checkpoint.
        candidates = sorted(ckpt_dir.glob("model_*.pt"))
        if candidates:
            last_pt = candidates[-1]
            print(f"model_last.pt not found; using most recent: {last_pt.name}")
        else:
            raise SystemExit(f"No checkpoint .pt files found in {ckpt_dir} after training.")
    shutil.copy2(last_pt, out_dir / "model_last.pt")
    print(f"Copied checkpoint: {last_pt.name}")
    # Remove intermediates to free disk for merge/export
    for pt in list(ckpt_dir.glob("model_*.pt")):
        if pt != last_pt:
            pt.unlink(missing_ok=True)
            print(f"Removed intermediate: {pt.name}")
    for safetensors in list(ckpt_dir.glob("pretrained_*.safetensors")):
        safetensors.unlink(missing_ok=True)
        print(f"Removed pretrained safetensors: {safetensors.name}")
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
