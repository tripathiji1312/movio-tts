"""HybridEngine — disk cache (5ms) + MMS-TTS fallback (50-150ms CPU).

Architecture for CPU-only real-time serving:

  Request
    │
    ├─ cache hit?  ──yes──▶  load PCM from disk  (~5ms)   ✓ p99 ≤500ms
    │
    └─ cache miss ──────▶  MMS-TTS synthesis    (~100ms)  ✓ p99 ≤500ms
                           │
                           └─▶  write to disk cache (async)

Pre-warm: run `python -m movio.acoustic.phrase_cache --config config/settings.yaml`
once after deployment to pre-synthesize all ~200 common taxi phrases. After that,
~90% of real-time traffic is served from disk at ~5ms.

The IndicF5 fine-tuned model is NOT loaded at runtime — it is used only in the
offline phrase_cache pre-synthesis step on a machine with a GPU/enough time.
"""

import asyncio
import hashlib
import logging
import time
from pathlib import Path
from typing import Iterator

import numpy as np

from movio.acoustic.mms_engine import MMSEngine
from movio.utils.audio import float_to_pcm16, pcm16_to_float

logger = logging.getLogger(__name__)


def _cache_key(text: str, voice: str) -> str:
    h = hashlib.sha256(f"{voice}|{text}".encode()).hexdigest()[:16]
    return h


class HybridEngine:
    """Disk-cache-first engine with MMS-TTS CPU fallback.

    model_path config options (stage_c.hybrid):
      cache_dir   — directory of pre-synthesized PCM16 files (built by phrase_cache.py)
      mms.*       — MMS-TTS config (passed through to MMSEngine)
    """

    SAMPLE_RATE = 16000   # MMS native; override in settings if needed

    def __init__(self, config: dict):
        cfg = config.get("stage_c", {}).get("hybrid", {})
        sr_cfg = int(config.get("server", {}).get("sample_rate", 16000))
        self.SAMPLE_RATE = sr_cfg

        self.cache_dir = Path(cfg.get("cache_dir", "models/phrase_cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.default_voice = cfg.get("default_voice", "default")
        self._mms = MMSEngine(config)
        self._hits = 0
        self._misses = 0

    @property
    def is_ready(self) -> bool:
        return self._mms.is_ready

    @property
    def model_path(self) -> str:
        return f"hybrid:cache={self.cache_dir}+mms={self._mms.model_id}"

    def load(self) -> None:
        self._mms.load()
        n = len(list(self.cache_dir.glob("*.pcm")))
        logger.info(
            "HybridEngine ready: %d pre-cached phrases, MMS fallback=%s",
            n, self._mms.model_id,
        )

    # ── cache helpers ────────────────────────────────────────────────────────

    def _pcm_path(self, text: str, voice: str) -> Path:
        key = _cache_key(text, voice)
        return self.cache_dir / f"{key}.pcm"

    def _load_from_cache(self, text: str, voice: str) -> np.ndarray | None:
        p = self._pcm_path(text, voice)
        if p.exists():
            try:
                audio = pcm16_to_float(p.read_bytes())
                self._hits += 1
                return audio
            except Exception as exc:
                logger.warning("corrupt cache file %s: %s — regenerating", p, exc)
                p.unlink(missing_ok=True)
        return None

    def _save_to_cache(self, text: str, voice: str, audio: np.ndarray) -> None:
        p = self._pcm_path(text, voice)
        try:
            p.write_bytes(float_to_pcm16(audio))
        except Exception as exc:
            logger.debug("cache write failed: %s", exc)

    # ── synthesis ────────────────────────────────────────────────────────────

    def synthesize(self, text: str, voice_name: str | None = None) -> np.ndarray:
        voice = voice_name or self.default_voice
        t0 = time.perf_counter()

        cached = self._load_from_cache(text, voice)
        if cached is not None:
            logger.debug("cache HIT %.1f ms | %s", (time.perf_counter() - t0) * 1000, text[:40])
            return cached

        self._misses += 1
        audio = self._mms.synthesize(text)
        elapsed = (time.perf_counter() - t0) * 1000
        logger.debug("MMS synth %.1f ms | %s", elapsed, text[:40])

        # Write to cache so the next identical request is instant
        self._save_to_cache(text, voice, audio)
        return audio

    def synthesize_stream(self, text: str, voice_name: str | None = None) -> Iterator[np.ndarray]:
        yield self.synthesize(text, voice_name)

    def cache_stats(self) -> dict:
        n_files = len(list(self.cache_dir.glob("*.pcm")))
        total = self._hits + self._misses
        return {
            "cached_phrases": n_files,
            "runtime_hits": self._hits,
            "runtime_misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total else 0.0,
        }
