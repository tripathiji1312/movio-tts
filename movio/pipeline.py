"""movio TTS pipeline — Stage A → B → C with CPU-optimised serving.

Serving architecture for CPU-only deployment:

  ┌─────────────────────────────────────────────────────┐
  │  Stage A: text normalisation  (<1 ms, thread pool)  │
  │  Stage B: Tanglish router     (<1 ms, thread pool)  │
  │  Stage C: IndicF5Engine                             │
  │    └─ CFM-DiT zero-shot TTS  →  ~250-400 ms (GPU)  │
  └─────────────────────────────────────────────────────┘

Concurrency model:
  A single asyncio.Queue feeds ONE synthesis worker thread. This prevents
  N concurrent requests from thrashing the CPU simultaneously (which would
  push all of them over 500 ms). With the queue, each request waits its
  turn but gets the CPU's full attention → p99 ≤500 ms as long as
  queue depth stays low (which it does for cache-heavy traffic).
"""

import asyncio
import io
import logging
import time
from dataclasses import dataclass, field

import numpy as np

from movio.acoustic.indicf5_engine import IndicF5Engine
from movio.cache.audio_cache import AudioCache
from movio.router.en2ta import transliterate_english_to_tamil
from movio.router.tanglish_router import TanglishRouter
from movio.textnorm.normalizer import TextNormalizer
from movio.utils.audio import float_to_pcm16

logger = logging.getLogger(__name__)


@dataclass
class SynthesisRequest:
    text: str
    voice: str | None = None
    language_hint: str | None = None


@dataclass
class StageTiming:
    stage_a_ms: float = 0.0
    stage_b_ms: float = 0.0
    stage_c_first_chunk_ms: float = 0.0
    total_ms: float = 0.0
    cache_hit: bool = False
    engine_used: str = ""


@dataclass
class SynthesisResult:
    audio: np.ndarray
    sample_rate: int
    normalized_text: str
    routed_text: str
    timings: StageTiming = field(default_factory=StageTiming)


class TTSPipeline:
    """Stage A→B→C TTS pipeline optimised for CPU-only p99 ≤500 ms.

    A: context-aware text normalisation (OTP, phone, booking ID, time, dates)
    B: Tanglish router (LID + xlit + <cs> code-switch boundaries)
    C: IndicF5Engine — zero-shot CFM-DiT TTS (Tamil + English + Tanglish)
    """

    def __init__(self, config: dict):
        self.config = config
        self.normalizer = TextNormalizer(config)
        self.router = TanglishRouter(config)
        self.engine = IndicF5Engine(config)
        self.audio_cache = AudioCache(config)
        self.sample_rate = self.engine.SAMPLE_RATE

        srv = config.get("server", {})
        self.ws_chunk_ms = int(srv.get("ws_chunk_ms", 50))

        timeouts = config.get("pipeline", {}).get("stage_timeout_ms", {})
        self.timeout_a = timeouts.get("stage_a", 200) / 1000
        self.timeout_b = timeouts.get("stage_b", 150) / 1000

        # Single synthesis worker — prevents CPU thrashing under concurrency
        self._synth_queue: asyncio.Queue | None = None
        self._worker_task: asyncio.Task | None = None

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def warmup(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.engine.load)

        # Start the background synthesis worker
        self._synth_queue = asyncio.Queue()
        self._worker_task = asyncio.create_task(self._synthesis_worker())

        logger.info(
            "TTSPipeline ready | engine=%s | sample_rate=%d",
            self.engine.model_path, self.sample_rate,
        )

    async def shutdown(self) -> None:
        if self._synth_queue is not None:
            await self._synth_queue.put(None)   # sentinel
        if self._worker_task is not None:
            await self._worker_task

    # ── synthesis worker (single thread, serialises GPU/CPU access) ──────────

    async def _synthesis_worker(self) -> None:
        """Drains the synthesis queue one request at a time.

        Serialising synthesis prevents N threads fighting over the CPU/GPU,
        which keeps p50 and p99 lower than parallel execution for short texts.
        """
        loop = asyncio.get_running_loop()
        while True:
            item = await self._synth_queue.get()
            if item is None:
                break
            text, voice_name, fut = item
            try:
                audio = await loop.run_in_executor(
                    None, self.engine.synthesize, text, voice_name
                )
                fut.set_result(audio)
            except Exception as exc:
                fut.set_exception(exc)
            finally:
                self._synth_queue.task_done()

    async def _synthesize_queued(self, text: str, voice_name: str | None) -> np.ndarray:
        """Submit to the worker queue and await the result."""
        if self._synth_queue is None:
            # Warmup was deferred — load now on first request
            await self.warmup()
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        await self._synth_queue.put((text, voice_name, fut))
        return await fut

    # ── public API ───────────────────────────────────────────────────────────

    async def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        timings = StageTiming()
        t0 = time.perf_counter()

        # Stage A — text normalisation
        loop = asyncio.get_running_loop()
        norm = await asyncio.wait_for(
            loop.run_in_executor(None, self.normalizer.normalize, request.text),
            timeout=self.timeout_a,
        )
        timings.stage_a_ms = (time.perf_counter() - t0) * 1000

        # Stage B — Tanglish router
        t1 = time.perf_counter()
        route = await asyncio.wait_for(
            loop.run_in_executor(None, self.router.route, norm.text),
            timeout=self.timeout_b,
        )
        timings.stage_b_ms = (time.perf_counter() - t1) * 1000

        voice_name = request.voice or None
        # Strip <cs> boundary tokens — they are internal router markers and
        # confuse the acoustic model if passed through as literal text.
        cs_token = getattr(self.router, "cs_token", "<cs>")
        synth_text = route.normalized_text.replace(cs_token, " ").strip()
        import re as _re
        synth_text = _re.sub(r" {2,}", " ", synth_text)

        # Transliterate any remaining Latin-script words to Tamil phonetics
        # so IndicF5 can pronounce them (it only speaks Tamil script).
        synth_text = transliterate_english_to_tamil(synth_text)

        # AudioCache check (in-memory / Redis — separate from disk phrase cache)
        cached_pcm = await self.audio_cache.get(synth_text, voice_name or "default", 0)
        if cached_pcm:
            timings.cache_hit = True
            audio = np.frombuffer(cached_pcm, dtype="<i2").astype(np.float32) / 32768.0
        else:
            t2 = time.perf_counter()
            audio = await self._synthesize_queued(synth_text, voice_name)
            timings.stage_c_first_chunk_ms = (time.perf_counter() - t2) * 1000
            timings.engine_used = "indicf5"
            asyncio.create_task(
                self.audio_cache.set(synth_text, voice_name or "default", 0,
                                     float_to_pcm16(audio))
            )

        timings.total_ms = (time.perf_counter() - t0) * 1000
        return SynthesisResult(
            audio=audio,
            sample_rate=self.sample_rate,
            normalized_text=norm.text,
            routed_text=synth_text,
            timings=timings,
        )

    async def stream_pcm_chunks(self, request: SynthesisRequest):
        """Async generator yielding PCM16 chunks as they arrive.

        Yields the full audio in one chunk. Streaming still reduces TTFA
        vs waiting for the full HTTP response.
        """
        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()

        norm = self.normalizer.normalize(request.text)
        route = self.router.route(norm.text)
        synth_text = route.normalized_text
        voice_name = request.voice or None

        audio = await self._synthesize_queued(synth_text, voice_name)
        logger.info("TTFA %.1f ms | %s", (time.perf_counter() - t0) * 1000, synth_text[:40])

        samples_per_chunk = max(1, self.sample_rate * self.ws_chunk_ms // 1000)
        for i in range(0, len(audio), samples_per_chunk):
            yield float_to_pcm16(audio[i: i + samples_per_chunk])
