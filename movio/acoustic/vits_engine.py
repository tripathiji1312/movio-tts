"""VITS Tamil acoustic engine — Solution 4 T1 (hot tier).

Uses Coqui TTS library to load a pre-trained Tamil VITS model.
Supports both PyTorch (via Synthesizer) and ONNX inference paths.

Checkpoint: samprabin/tamil_vits (MIT license, ~80M params)
Latency: ~80ms GPU, ~150ms ONNX CPU per sentence
"""

import logging
from pathlib import Path
from typing import Iterator

import numpy as np

from movio.acoustic.engine_base import AcousticEngine

logger = logging.getLogger(__name__)


class VITSEngine(AcousticEngine):
    """VITS end-to-end TTS (GlowTTS encoder + HiFi-GAN decoder).

    Single forward pass — no separate vocoder needed.
    """

    def __init__(self, config: dict):
        cfg = config.get("stage_c", {}).get("vits", {})
        self.model_path = cfg.get("model_path", "")
        self.config_path = cfg.get("config_path", "")
        self.onnx_path = cfg.get("onnx_path", "")
        self.use_onnx = bool(cfg.get("use_onnx", False))
        self.device = cfg.get("device", "cpu")
        self.speaker_id = cfg.get("speaker_id", None)
        self.length_scale = float(cfg.get("length_scale", 1.0))
        self.noise_scale = float(cfg.get("noise_scale", 0.667))
        self.noise_scale_w = float(cfg.get("noise_scale_w", 0.8))
        self._sample_rate = int(cfg.get("sample_rate", 22050))
        self._synthesizer = None
        self._onnx_session = None

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def is_ready(self) -> bool:
        if self.use_onnx:
            return self._onnx_session is not None
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
        config_path = Path(self.config_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"VITS model not found at {model_path}. "
                "Run: python scripts/download_models.py"
            )
        if not config_path.exists():
            raise FileNotFoundError(
                f"VITS config not found at {config_path}. "
                "Run: python scripts/download_models.py"
            )

        from TTS.utils.synthesizer import Synthesizer

        use_cuda = self.device == "cuda"
        logger.info("Loading VITS from %s (cuda=%s)", model_path, use_cuda)

        self._synthesizer = Synthesizer(
            tts_checkpoint=str(model_path),
            tts_config_path=str(config_path),
            use_cuda=use_cuda,
        )
        self._sample_rate = self._synthesizer.output_sample_rate
        logger.info("VITS loaded: sample_rate=%d", self._sample_rate)

    def _load_onnx(self):
        if self._onnx_session is not None:
            return
        import os

        import onnxruntime as ort

        onnx_file = Path(self.onnx_path)
        if not onnx_file.exists():
            raise FileNotFoundError(
                f"ONNX model not found at {onnx_file}. "
                "Run: python scripts/export_onnx.py"
            )

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.device == "cuda"
            else ["CPUExecutionProvider"]
        )

        num_physical = max(1, (os.cpu_count() or 4) // 2)
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads = num_physical
        sess_opts.inter_op_num_threads = 1
        sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_opts.enable_mem_pattern = True
        sess_opts.enable_cpu_mem_arena = True

        logger.info(
            "Loading VITS ONNX from %s (threads: intra=%d, sequential)",
            onnx_file, num_physical,
        )
        self._onnx_session = ort.InferenceSession(
            str(onnx_file), sess_opts, providers=providers
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

        # Build feed dict from actual model inputs (scales may be baked in during export)
        input_names = [i.name for i in self._onnx_session.get_inputs()]
        feeds = {}
        if "input" in input_names:
            feeds["input"] = input_ids_np
        if "input_lengths" in input_names:
            feeds["input_lengths"] = input_lengths
        if "scales" in input_names:
            feeds["scales"] = np.array(
                [self.noise_scale, self.length_scale, self.noise_scale_w],
                dtype=np.float32,
            )
        if "sid" in input_names and self.speaker_id is not None:
            feeds["sid"] = np.array([self.speaker_id], dtype=np.int64)

        outputs = self._onnx_session.run(None, feeds)
        audio = outputs[0].squeeze()
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
