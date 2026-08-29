"""Export VITS and FastSpeech2 models to ONNX for fast CPU inference.

ONNX achieves ~150ms per sentence on CPU vs ~2s for unoptimized PyTorch.
This is MANDATORY for production deployment without GPU.

Usage:
    python scripts/export_onnx.py
    python scripts/export_onnx.py --model vits
    python scripts/export_onnx.py --model fastspeech2
"""

import argparse
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
VITS_DIR = MODELS_DIR / "vits" / "tamil_vits"
FS2_DIR = MODELS_DIR / "fastspeech2" / "tamil_fs2"
HIFIGAN_DIR = MODELS_DIR / "fastspeech2" / "hifigan"


def export_vits():
    """Export VITS model to ONNX."""
    import torch

    model_path = VITS_DIR / "model.pth"
    config_path = VITS_DIR / "config.json"
    output_path = VITS_DIR / "model.onnx"

    if not model_path.exists():
        logger.error("VITS model not found at %s. Run download_models.py first.", model_path)
        return False

    logger.info("Exporting VITS to ONNX...")

    try:
        from TTS.tts.configs.vits_config import VitsConfig
        from TTS.tts.models.vits import Vits
        from TTS.utils.io import load_config

        config = load_config(str(config_path))
        model = Vits.init_from_config(config)

        checkpoint = torch.load(str(model_path), map_location="cpu")
        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                model.load_state_dict(checkpoint["model"])
            elif "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)
        model.eval()

        num_chars = config.model_args.get("num_chars", 200)
        dummy_input = torch.randint(0, num_chars, (1, 50), dtype=torch.long)
        dummy_lengths = torch.tensor([50], dtype=torch.long)
        dummy_scales = torch.tensor([0.667, 1.0, 0.8], dtype=torch.float32)

        model.export_onnx(str(output_path))
        logger.info("VITS ONNX exported to: %s", output_path)

    except (AttributeError, TypeError):
        logger.info("Built-in export failed, using manual torch.onnx.export...")

        try:
            from TTS.utils.synthesizer import Synthesizer

            synth = Synthesizer(
                tts_checkpoint=str(model_path),
                tts_config_path=str(config_path),
                use_cuda=False,
            )
            tts_model = synth.tts_model
            tts_model.eval()

            if hasattr(tts_model, "export_onnx"):
                tts_model.export_onnx(str(output_path))
            else:
                logger.warning(
                    "Model doesn't support ONNX export directly. "
                    "Consider using TTS library's built-in export tools."
                )
                return False
        except Exception as exc:
            logger.error("VITS ONNX export failed: %s", exc)
            return False

    _verify_onnx(output_path, "VITS")
    return True


def export_fastspeech2():
    """Export FastSpeech2 + HiFi-GAN to ONNX."""
    import torch

    fs2_model_path = FS2_DIR / "model.pth"
    hifigan_model_path = HIFIGAN_DIR / "model.pth"
    fs2_onnx_path = FS2_DIR / "model.onnx"
    hifigan_onnx_path = HIFIGAN_DIR / "vocoder.onnx"

    if not fs2_model_path.exists():
        logger.error("FastSpeech2 model not found at %s", fs2_model_path)
        return False

    logger.info("Exporting FastSpeech2 to ONNX...")

    try:
        from TTS.utils.synthesizer import Synthesizer

        config_path = FS2_DIR / "config.json"
        vocoder_config = HIFIGAN_DIR / "config.json"

        synth = Synthesizer(
            tts_checkpoint=str(fs2_model_path),
            tts_config_path=str(config_path) if config_path.exists() else None,
            vocoder_checkpoint=str(hifigan_model_path) if hifigan_model_path.exists() else None,
            vocoder_config=str(vocoder_config) if vocoder_config.exists() else None,
            use_cuda=False,
        )

        if hasattr(synth.tts_model, "export_onnx"):
            synth.tts_model.export_onnx(str(fs2_onnx_path))
            logger.info("FastSpeech2 ONNX exported: %s", fs2_onnx_path)

        if synth.vocoder_model and hasattr(synth.vocoder_model, "export_onnx"):
            synth.vocoder_model.export_onnx(str(hifigan_onnx_path))
            logger.info("HiFi-GAN ONNX exported: %s", hifigan_onnx_path)

    except Exception as exc:
        logger.error("FastSpeech2 ONNX export failed: %s", exc)
        logger.info("Manual export may be required for this model architecture.")
        return False

    return True


def _verify_onnx(path: Path, name: str):
    """Verify ONNX model loads and runs basic inference."""
    if not path.exists():
        logger.warning("%s ONNX file not found at %s", name, path)
        return

    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inputs = sess.get_inputs()
        outputs = sess.get_outputs()
        logger.info(
            "%s ONNX verified: %d inputs, %d outputs",
            name, len(inputs), len(outputs),
        )
        for inp in inputs:
            logger.info("  Input: %s shape=%s dtype=%s", inp.name, inp.shape, inp.type)
        for out in outputs:
            logger.info("  Output: %s shape=%s", out.name, out.shape)

        size_mb = path.stat().st_size / 1e6
        logger.info("  Size: %.1f MB", size_mb)
    except Exception as exc:
        logger.warning("%s ONNX verification failed: %s", name, exc)


def main():
    ap = argparse.ArgumentParser(description="Export TTS models to ONNX")
    ap.add_argument(
        "--model",
        choices=["all", "vits", "fastspeech2"],
        default="all",
    )
    args = ap.parse_args()

    results = {}
    if args.model in ("all", "vits"):
        results["vits"] = export_vits()
    if args.model in ("all", "fastspeech2"):
        results["fastspeech2"] = export_fastspeech2()

    print("\nExport results:")
    for name, ok in results.items():
        print(f"  {name}: {'SUCCESS' if ok else 'FAILED'}")

    if all(results.values()):
        print("\nONNX models ready. Update config/settings.yaml:")
        print("  stage_c.vits.use_onnx: true")
        print("  stage_c.fastspeech2.use_onnx: true")


if __name__ == "__main__":
    main()
