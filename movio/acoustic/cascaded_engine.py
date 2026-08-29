"""Cascaded dual-engine: VITS (T1) + FastSpeech2 (T2).

The cascaded engine tries VITS first — if it fails, produces silence,
or produces an artifact (clipping, excessive duration), it falls through
to FastSpeech2+HiFi-GAN. This gives us VITS's naturalness when it
works, plus FastSpeech2's robustness as a safety net.
"""

import logging
import time
from typing import Iterator

import numpy as np

from movio.acoustic.engine_base import AcousticEngine
from movio.acoustic.fastspeech2_engine import FastSpeech2Engine
from movio.acoustic.vits_engine import VITSEngine
from movio.utils.audio import crossfade, resample, trim_silence

logger = logging.getLogger(__name__)

MAX_AUDIO_DURATION_S = 30.0
MIN_AUDIO_SAMPLES = 100
CLIP_THRESHOLD = 0.99
CLIP_RATIO_LIMIT = 0.1


TARGET_PEAK = 0.85


def _normalize_volume(audio: np.ndarray) -> np.ndarray:
    """Normalize audio to consistent volume level."""
    peak = np.max(np.abs(audio))
    if peak > 0.01:
        audio = audio * (TARGET_PEAK / peak)
    return audio


def _is_valid_audio(audio: np.ndarray, sample_rate: int) -> bool:
    if audio is None or len(audio) < MIN_AUDIO_SAMPLES:
        return False
    duration = len(audio) / sample_rate
    if duration > MAX_AUDIO_DURATION_S:
        return False
    clip_ratio = np.mean(np.abs(audio) > CLIP_THRESHOLD)
    if clip_ratio > CLIP_RATIO_LIMIT:
        return False
    if np.max(np.abs(audio)) < 0.001:
        return False
    return True


class CascadedEngine(AcousticEngine):
    """Dual-tier engine: VITS → FastSpeech2 fallback.

    - T1 (VITS): Natural, fast, single-pass. Handles most Tamil text.
    - T2 (FastSpeech2+HiFi-GAN): Deterministic, never fails. Catches
      OOV, unusual phoneme sequences, and VITS edge cases.

    Output sample rate is unified to the configured pipeline rate.
    """

    def __init__(self, config: dict):
        cfg = config.get("stage_c", {})
        self._target_sr = int(config.get("server", {}).get("sample_rate", 22050))
        self.vits = VITSEngine(config)
        self.fs2 = FastSpeech2Engine(config)
        self.enable_fallback = bool(cfg.get("enable_fallback", True))
        self.vits_timeout_ms = float(cfg.get("vits_timeout_ms", 5000))
        self._t1_failures = 0
        self._t1_total = 0

    @property
    def sample_rate(self) -> int:
        return self._target_sr

    @property
    def is_ready(self) -> bool:
        return self.vits.is_ready or self.fs2.is_ready

    def load(self):
        try:
            self.vits.load()
            logger.info("VITS engine loaded (T1 hot tier)")
        except Exception as exc:
            logger.warning("VITS load failed, will use FastSpeech2 only: %s", exc)

        if self.enable_fallback:
            try:
                self.fs2.load()
                logger.info("FastSpeech2 engine loaded (T2 fallback)")
            except Exception as exc:
                logger.warning("FastSpeech2 load failed: %s", exc)

        if not self.is_ready:
            raise RuntimeError(
                "Neither VITS nor FastSpeech2 engine could be loaded. "
                "Check model paths in config/settings.yaml"
            )

    def _prepare_text(self, text: str) -> str:
        """Transliterate English segments to Tamil for the Tamil-only VITS model."""
        from movio.acoustic.transliterate import transliterate_english_segments
        return transliterate_english_segments(text)

    def synthesize(self, text: str) -> np.ndarray:
        self._t1_total += 1
        synth_text = self._prepare_text(text)

        if self.vits.is_ready:
            try:
                t0 = time.perf_counter()
                audio = self.vits.synthesize(synth_text)
                elapsed_ms = (time.perf_counter() - t0) * 1000

                if _is_valid_audio(audio, self.vits.sample_rate):
                    audio = trim_silence(audio, threshold_db=-40.0,
                                         sample_rate=self.vits.sample_rate)
                    audio = _normalize_volume(audio)
                    audio = resample(audio, self.vits.sample_rate, self._target_sr)
                    logger.debug("VITS T1 synth %.1fms: %s", elapsed_ms, text[:40])
                    return audio
                else:
                    logger.warning("VITS produced invalid audio for: %s", text[:60])
                    self._t1_failures += 1
            except Exception as exc:
                logger.warning("VITS synthesis failed (%s), falling back to FastSpeech2", exc)
                self._t1_failures += 1

        if self.enable_fallback and self.fs2.is_ready:
            try:
                t0 = time.perf_counter()
                audio = self.fs2.synthesize(text)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                audio = trim_silence(audio, threshold_db=-40.0,
                                     sample_rate=self.fs2.sample_rate)
                audio = _normalize_volume(audio)
                audio = resample(audio, self.fs2.sample_rate, self._target_sr)
                logger.debug("FastSpeech2 T2 synth %.1fms: %s", elapsed_ms, text[:40])
                return audio
            except Exception as exc:
                logger.error("FastSpeech2 also failed: %s", exc)
                raise

        raise RuntimeError("No engine available for synthesis")

    def synthesize_stream(self, text: str, chunk_words: int = 8) -> Iterator[np.ndarray]:
        from movio.acoustic.chunking import chunk_text

        chunks = chunk_text(text, min_syl=8, max_syl=16)
        if not chunks:
            chunks = [text]

        xfade_samples = int(0.01 * self._target_sr)  # 10ms crossfade
        prev_tail: np.ndarray | None = None

        for chunk in chunks:
            audio = self.synthesize(chunk)
            if audio is None or len(audio) == 0:
                continue
            if prev_tail is not None and len(prev_tail) > 0:
                audio = crossfade(prev_tail, audio, xfade_samples)
                prev_tail = None
            if len(audio) > xfade_samples:
                prev_tail = audio[-xfade_samples:]
                yield audio[:-xfade_samples]
            else:
                yield audio

        if prev_tail is not None:
            yield prev_tail

    @property
    def stats(self) -> dict:
        return {
            "t1_total": self._t1_total,
            "t1_failures": self._t1_failures,
            "t1_success_rate": (
                (self._t1_total - self._t1_failures) / max(1, self._t1_total)
            ),
            "vits_ready": self.vits.is_ready,
            "fs2_ready": self.fs2.is_ready,
        }
