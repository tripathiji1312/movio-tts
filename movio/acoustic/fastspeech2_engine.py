"""FastSpeech2 + HiFi-GAN acoustic engine — Solution 4 T2 (fallback tier).

Non-autoregressive, deterministic, robust. Used when VITS is unavailable or
fails on a given input.

Checkpoint: smtiitm/FastSpeech2_HS (IIT Madras, MIT license, ~15M + 14M params)
Latency: ~120ms total (FastSpeech2 ~60ms + HiFi-GAN ~60ms)
"""

import logging
from pathlib import Path
from typing import Iterator

import numpy as np

from movio.acoustic.engine_base import AcousticEngine

logger = logging.getLogger(__name__)


class FastSpeech2Engine(AcousticEngine):
    """FastSpeech2 + HiFi-GAN vocoder.

    Two-stage: text → mel spectrogram (FastSpeech2) → waveform (HiFi-GAN).
    Deterministic and robust — never fails on valid text input.
    """

    def __init__(self, config: dict):
        cfg = config.get("stage_c", {}).get("fastspeech2", {})
        self.model_path = cfg.get("model_path", "")
        self.vocoder_path = cfg.get("vocoder_path", "")
        self.config_path = cfg.get("config_path", "")
        self.vocoder_config_path = cfg.get("vocoder_config_path", "")
        self.onnx_path = cfg.get("onnx_path", "")
        self.vocoder_onnx_path = cfg.get("vocoder_onnx_path", "")
        self.use_onnx = bool(cfg.get("use_onnx", False))
        self.device = cfg.get("device", "cpu")
        self._sample_rate = int(cfg.get("sample_rate", 22050))
        self._synthesizer = None
        self._onnx_session = None
        self._vocoder_onnx = None

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def is_ready(self) -> bool:
        if self.use_onnx:
            return self._onnx_session is not None and self._vocoder_onnx is not None
        return self._synthesizer is not None

    def load(self):
        if self.use_onnx:
            self._load_onnx()
        else:
            self._load_pytorch()

    def _load_pytorch(self):
        if self._synthesizer is not None:
            return

        model_path = Path(self.model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"FastSpeech2 model not found at {model_path}. "
                "Run: python scripts/download_models.py"
            )

        from TTS.utils.synthesizer import Synthesizer

        use_cuda = self.device == "cuda"
        kwargs = {
            "tts_checkpoint": str(model_path),
            "use_cuda": use_cuda,
        }

        config_path = Path(self.config_path)
        if config_path.exists():
            kwargs["tts_config_path"] = str(config_path)

        vocoder_path = Path(self.vocoder_path)
        if vocoder_path.exists():
            kwargs["vocoder_checkpoint"] = str(vocoder_path)
            vocoder_cfg = Path(self.vocoder_config_path)
            if vocoder_cfg.exists():
                kwargs["vocoder_config"] = str(vocoder_cfg)

        logger.info("Loading FastSpeech2 from %s (cuda=%s)", model_path, use_cuda)
        self._synthesizer = Synthesizer(**kwargs)
        self._sample_rate = self._synthesizer.output_sample_rate
        logger.info("FastSpeech2 loaded: sample_rate=%d", self._sample_rate)

    def _load_onnx(self):
        if self._onnx_session is not None:
            return
        import onnxruntime as ort

        providers = ["CPUExecutionProvider"]
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads = 4

        onnx_path = Path(self.onnx_path)
        vocoder_onnx = Path(self.vocoder_onnx_path)

        if not onnx_path.exists():
            raise FileNotFoundError(f"FastSpeech2 ONNX not found: {onnx_path}")
        if not vocoder_onnx.exists():
            raise FileNotFoundError(f"HiFi-GAN ONNX not found: {vocoder_onnx}")

        logger.info("Loading FastSpeech2 ONNX from %s", onnx_path)
        self._onnx_session = ort.InferenceSession(
            str(onnx_path), sess_opts, providers=providers
        )
        logger.info("Loading HiFi-GAN ONNX from %s", vocoder_onnx)
        self._vocoder_onnx = ort.InferenceSession(
            str(vocoder_onnx), sess_opts, providers=providers
        )

    def _ensure_loaded(self):
        if self.use_onnx:
            if self._onnx_session is None:
                self._load_onnx()
        else:
            if self._synthesizer is None:
                self._load_pytorch()

    def synthesize(self, text: str) -> np.ndarray:
        self._ensure_loaded()
        if self.use_onnx:
            return self._synthesize_onnx(text)
        return self._synthesize_pytorch(text)

    def _synthesize_pytorch(self, text: str) -> np.ndarray:
        wav = self._synthesizer.tts(text)
        return np.array(wav, dtype=np.float32)

    def _synthesize_onnx(self, text: str) -> np.ndarray:
        from movio.acoustic.vits_text_processor import text_to_sequence

        input_ids = text_to_sequence(text, config_path=self.config_path)
        input_ids_np = np.array([input_ids], dtype=np.int64)
        input_lengths = np.array([len(input_ids)], dtype=np.int64)

        mel_outputs = self._onnx_session.run(
            None, {"input": input_ids_np, "input_lengths": input_lengths}
        )
        mel = mel_outputs[0]

        audio_outputs = self._vocoder_onnx.run(None, {"mel": mel})
        audio = audio_outputs[0].squeeze()
        return audio.astype(np.float32)

    def synthesize_stream(self, text: str, chunk_words: int = 8) -> Iterator[np.ndarray]:
        from movio.acoustic.chunking import chunk_text

        chunks = chunk_text(text, min_syl=6, max_syl=12)
        if not chunks:
            chunks = [text]
        for chunk in chunks:
            audio = self.synthesize(chunk)
            if audio is not None and len(audio) > 0:
                yield audio
