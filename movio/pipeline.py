import asyncio
import logging
import time
from dataclasses import dataclass, field

import numpy as np

from movio.acoustic.cascaded_engine import CascadedEngine
from movio.cache.audio_cache import AudioCache
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
    """Orchestrator — Solution 4: VITS + FastSpeech2 cascaded stack.

    A: context-aware normalization (WFST → NeMo → domain rules)
    B: Tanglish router (LID + xlit + <cs> boundaries)
    C: Cascaded synthesis — VITS (T1) → FastSpeech2+HiFi-GAN (T2 fallback)
    """

    def __init__(self, config: dict):
        self.config = config
        self.normalizer = TextNormalizer(config)
        self.router = TanglishRouter(config)
        self.engine = CascadedEngine(config)
        self.cache = AudioCache(config)
        sr_cfg = config.get("server", {})
        self.sample_rate = int(sr_cfg.get("sample_rate", 22050))
        self.ws_chunk_ms = int(sr_cfg.get("ws_chunk_ms", 50))
        timeouts = config.get("pipeline", {}).get("stage_timeout_ms", {})
        self.timeout_a = timeouts.get("stage_a", 200) / 1000
        self.timeout_b = timeouts.get("stage_b", 150) / 1000

    async def warmup(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.engine.load)
        logger.info("Cascaded engine ready: %s", self.engine.stats)

    async def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        timings = StageTiming()
        t0 = time.perf_counter()

        norm = await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(
                None, self.normalizer.normalize, request.text
            ),
            timeout=self.timeout_a,
        )
        timings.stage_a_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        route = await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(
                None, self.router.route, norm.text
            ),
            timeout=self.timeout_b,
        )
        timings.stage_b_ms = (time.perf_counter() - t1) * 1000

        cache_key = route.normalized_text
        cached = await self.cache.get(cache_key, "default", 0)
        if cached:
            timings.cache_hit = True
            audio = np.frombuffer(cached, dtype="<i2").astype(np.float32) / 32768.0
        else:
            t2 = time.perf_counter()
            audio = await asyncio.get_running_loop().run_in_executor(
                None, self.engine.synthesize, route.normalized_text,
            )
            timings.stage_c_first_chunk_ms = (time.perf_counter() - t2) * 1000
            pcm = float_to_pcm16(audio)
            asyncio.create_task(self.cache.set(cache_key, "default", 0, pcm))

        timings.total_ms = (time.perf_counter() - t0) * 1000
        return SynthesisResult(
            audio=audio,
            sample_rate=self.sample_rate,
            normalized_text=norm.text,
            routed_text=route.normalized_text,
            timings=timings,
        )

    async def stream_pcm_chunks(self, request: SynthesisRequest):
        """Async generator yielding ~ws_chunk_ms PCM16 chunks as they're ready."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        started = time.perf_counter()

        norm = self.normalizer.normalize(request.text)
        route = self.router.route(norm.text)
        synth_text = route.normalized_text

        def produce():
            for seg in self.engine.synthesize_stream(synth_text):
                yield seg

        async def producer():
            gen = produce()
            while True:
                try:
                    seg = await loop.run_in_executor(None, next, gen)
                except StopIteration:
                    break
                await queue.put(seg)
            await queue.put(None)

        task = asyncio.create_task(producer())
        samples_per_chunk = max(1, self.sample_rate * self.ws_chunk_ms // 1000)

        first = True
        while True:
            seg = await queue.get()
            if seg is None:
                break
            for i in range(0, len(seg), samples_per_chunk):
                yield float_to_pcm16(seg[i : i + samples_per_chunk])
            if first:
                logger.info("TTFA %.1f ms", (time.perf_counter() - started) * 1000)
                first = False
        await task
