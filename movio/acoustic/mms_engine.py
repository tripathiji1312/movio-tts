"""MMS-TTS engine — Facebook MMS-TTS Tamil (single-pass VITS, ~50-150ms CPU).

Used as the real-time fallback for uncached dynamic phrases (OTPs, names,
addresses). Pre-cached phrases are served from disk at ~5ms.

Model: facebook/mms-tts-tam — 80 MB, Apache 2.0, no GPU required.
Sample rate: 16000 Hz (upsampled to server rate on output).
"""

import logging
from pathlib import Path
from typing import Iterator

import numpy as np

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16000


class MMSEngine:
    """Facebook MMS-TTS Tamil — fast CPU inference for dynamic phrases.

    Load is lazy: first synthesis call triggers model download (~80 MB).
    Subsequent calls are ~50-150 ms on a modern CPU core.
    """

    SAMPLE_RATE = 16000

    def __init__(self, config: dict):
        cfg = config.get("stage_c", {}).get("mms", {})
        self.model_id = cfg.get("model_id", "facebook/mms-tts-tam")
        self.device = "cpu"   # MMS is for CPU serving — never use CUDA here
        self.target_sr = int(config.get("server", {}).get("sample_rate", 16000))
        self._processor = None
        self._model = None

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self.is_ready:
            return
        logger.info("Loading MMS-TTS model %s ...", self.model_id)
        from transformers import VitsModel, AutoTokenizer
        import torch

        self._processor = AutoTokenizer.from_pretrained(self.model_id)
        self._model = VitsModel.from_pretrained(self.model_id)
        self._model.eval()
        logger.info("MMS-TTS ready (sample_rate=%d)", self.SAMPLE_RATE)

    def synthesize(self, text: str) -> np.ndarray:
        self.load()
        import torch

        inputs = self._processor(text, return_tensors="pt")
        with torch.no_grad():
            output = self._model(**inputs).waveform
        audio = output.squeeze().numpy().astype(np.float32)

        if self.target_sr != self.SAMPLE_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=self.SAMPLE_RATE,
                                     target_sr=self.target_sr)
        return audio

    def synthesize_stream(self, text: str) -> Iterator[np.ndarray]:
        # MMS is single-pass — yield the whole thing as one chunk
        yield self.synthesize(text)
