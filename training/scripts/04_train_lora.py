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

    meta_src = data_dir / "metadata.csv"
    ds_root = data_dir / "wavs"
    ds_root.mkdir(exist_ok=True)
    import shutil

    audio_dir = data_dir / "audio"
    for wav in audio_dir.glob("*.wav"):
        link = ds_root / wav.name
        if not link.exists():
            os.symlink(wav.resolve(), link)

    cfg_path = out_dir / "finetune_config.yaml"
    cfg = f"""\
data:
  train: {meta_src}
  root: {ds_root}
model:
  name: F5TTS_v1_Base
  tokenizer: pinyin
  ckpt: null
train:
  epochs: {epochs}
  batch_size: {batch_size}
  learning_rate: 1e-05
  num_warmup_updates: 200
  save_per_updates: 1000
  mixed_precision: fp16
  finetune: true
"""
    cfg_path.write_text(cfg)
    print(f"F5-TTS finetune config:\n{cfg}")
    cmd = [
        "f5-tts", "finetune",
        "--config", str(cfg_path),
        "--dataset-name", "movio_tanglish",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(workdir))
    print(f"Checkpoints under: {workdir}/ckpts (copy final .pt into {out_dir})")


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
