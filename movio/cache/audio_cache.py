import hashlib
import logging

logger = logging.getLogger(__name__)

try:
    import redis.asyncio as aioredis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False


class MemoryCache:
    def __init__(self, ttl_seconds: int, max_entries: int = 2048):
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self._store: dict[str, tuple[float, bytes]] = {}

    async def get(self, key: str) -> bytes | None:
        import time

        item = self._store.get(key)
        if not item:
            return None
        ts, value = item
        if time.time() - ts > self.ttl:
            self._store.pop(key, None)
            return None
        return value

    async def set(self, key: str, value: bytes) -> None:
        import time

        if len(self._store) >= self.max_entries:
            oldest = min(self._store, key=lambda k: self._store[k][0])
            self._store.pop(oldest, None)
        self._store[key] = (time.time(), value)


class AudioCache:
    """Sub-sentence audio cache keyed on normalized text + voice + flow steps."""

    def __init__(self, config: dict):
        cfg = config.get("cache", {})
        backend = cfg.get("backend", "memory")
        self.enabled = bool(config.get("pipeline", {}).get("enable_cache", True))
        self.ttl = int(cfg.get("ttl_seconds", 604800))
        self.max_bytes = int(cfg.get("max_audio_bytes_mb", 8)) * 1024 * 1024
        self._redis = None
        if self.enabled and backend == "redis" and HAS_REDIS:
            try:
                url = cfg.get("redis_url", "redis://localhost:6379/0")
                self._client = aioredis.from_url(url, decode_responses=False)
                logger.info("Audio cache: redis (%s)", url)
            except Exception as exc:
                logger.warning("Redis unavailable (%s); in-memory cache", exc)
                self._client = None
        else:
            self._client = None
        self._fallback = MemoryCache(self.ttl)

    @staticmethod
    def make_key(text: str, voice: str, steps: int) -> str:
        h = hashlib.sha256(f"{voice}|{steps}|{text}".encode()).hexdigest()
        return f"movio:tts:{h}"

    async def get(self, text: str, voice: str, steps: int) -> bytes | None:
        if not self.enabled:
            return None
        key = self.make_key(text, voice, steps)
        try:
            if self._client is not None:
                data = await self._client.get(key)
                return data if data and len(data) <= self.max_bytes else None
            return await self._fallback.get(key)
        except Exception as exc:
            logger.debug("cache get failed: %s", exc)
            return await self._fallback.get(key)

    async def set(self, text: str, voice: str, steps: int, pcm: bytes) -> None:
        if not self.enabled or len(pcm) > self.max_bytes:
            return
        key = self.make_key(text, voice, steps)
        try:
            if self._client is not None:
                await self._client.setex(key, self.ttl, pcm)
            else:
                await self._fallback.set(key, pcm)
        except Exception as exc:
            logger.debug("cache set failed: %s", exc)
            await self._fallback.set(key, pcm)
