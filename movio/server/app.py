import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from movio.pipeline import SynthesisRequest, TTSPipeline
from movio.utils.audio import wav_bytes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("movio.server")


async def _background_warmup(p: "TTSPipeline") -> None:
    try:
        await p.warmup()
    except Exception as exc:
        logger.warning("Pipeline warmup deferred: %s", exc)

pipeline: TTSPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    from movio.textnorm.normalizer import load_settings

    settings = load_settings()
    # Wire REDIS_URL env (docker-compose) into AudioCache config — previously
    # the env var was set but never read (only config.cache.redis_url was).
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        settings.setdefault("cache", {})["backend"] = "redis"
        settings["cache"]["redis_url"] = redis_url
    pipeline = TTSPipeline(settings)
    # Warmup runs in background — server starts immediately, readiness is
    # explicit via /healthz {"ready":true} (model loads on first request fallback).
    asyncio.create_task(_background_warmup(pipeline))
    yield
    if pipeline is not None:
        await pipeline.shutdown()
    pipeline = None


app = FastAPI(
    title="movio TTS — IndicF5 streaming (24 kHz, CPU-first)",
    version="0.5.0",
    lifespan=lifespan,
)

STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def root():
    from fastapi.responses import FileResponse
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "movio TTS API — see /docs"}


class TTSRequest(BaseModel):
    text: str
    voice: str | None = None
    language_hint: str | None = None
    speed: float | None = None


@app.get("/healthz")
async def healthz():
    # Explicit readiness: background warmup means the socket is open before
    # the 337M DiT + Vocos + ref-audio are loaded. Clients must wait for
    # ready:true instead of timing a silent 2-min first synthesis as TTFA.
    if pipeline is None:
        return {"status": "starting", "ready": False}
    try:
        ready = bool(pipeline.engine.is_ready)
    except Exception:
        ready = False
    if ready:
        return {"status": "ok", "ready": True}
    return {"status": "starting", "ready": False}


@app.get("/voices")
async def voices():
    if pipeline and hasattr(pipeline.engine, "voices") and pipeline.engine.voices:
        return {"voices": list(pipeline.engine.voices.keys())}
    return {"voices": ["ta_female_neutral", "ta_male_neutral"]}


@app.get("/engine/stats")
async def engine_stats():
    if pipeline is None:
        raise HTTPException(503, "warming")
    engine = pipeline.engine
    stats = {
        "engine": "indicf5",
        "model_path": engine.model_path,
        "sample_rate": engine.SAMPLE_RATE,
        "is_ready": engine.is_ready,
        "flow_steps": getattr(engine, "num_flow_steps", None),
        "ode_method": getattr(engine, "ode_method", None),
        "device": getattr(engine, "_device", getattr(engine, "device", None)),
        "ws_chunk_ms": getattr(pipeline, "ws_chunk_ms", 50),
    }
    if hasattr(engine, "cache_stats"):
        stats.update(engine.cache_stats())
    return stats


@app.post("/tts")
async def tts(req: TTSRequest):
    if not req.text.strip():
        raise HTTPException(400, "text is required")
    result = await pipeline.synthesize(
        SynthesisRequest(text=req.text, voice=req.voice, language_hint=req.language_hint, speed=req.speed)
    )
    audio = wav_bytes(result.audio, result.sample_rate)
    return {
        "audio_wav_base64": __import__("base64").b64encode(audio).decode(),
        "sample_rate": result.sample_rate,
        "normalized_text": result.normalized_text,
        "routed_text": result.routed_text,
        "timings": {
            "stage_a_ms": round(result.timings.stage_a_ms, 2),
            "stage_b_ms": round(result.timings.stage_b_ms, 2),
            "stage_c_first_chunk_ms": round(result.timings.stage_c_first_chunk_ms, 2),
            "total_ms": round(result.timings.total_ms, 2),
            "cache_hit": result.timings.cache_hit,
        },
    }


@app.post("/tts/wav")
async def tts_wav(req: TTSRequest):
    if not req.text.strip():
        raise HTTPException(400, "text is required")
    result = await pipeline.synthesize(
        SynthesisRequest(text=req.text, voice=req.voice, language_hint=req.language_hint, speed=req.speed)
    )
    from fastapi.responses import Response
    return Response(
        content=wav_bytes(result.audio, result.sample_rate),
        media_type="audio/wav",
        headers={
            "X-Normalized-Text": result.normalized_text[:500],
            "X-Cache-Hit": str(result.timings.cache_hit),
            "X-Total-Ms": f"{result.timings.total_ms:.1f}",
        },
    )


@app.websocket("/tts/stream")
async def tts_stream(ws: WebSocket):
    from starlette.websockets import WebSocketState

    await ws.accept()
    try:
        while True:
            msg = await ws.receive_json()
            text = (msg.get("text") or "").strip()
            if not text:
                await ws.send_json({"type": "error", "message": "empty text"})
                continue
            speed_val = msg.get("speed")
            speed = float(speed_val) if speed_val is not None else None
            req = SynthesisRequest(text=text, voice=msg.get("voice"), speed=speed)
            await ws.send_json({"type": "start"})
            # Live progress feed so demos/clients can show something is
            # happening during the long CPU synthesis (stages + chunk count).
            # Binary PCM16 frames keep flowing between these JSON messages;
            # order is preserved per connection.
            await ws.send_json({"type": "progress", "phase": "AB",
                                "message": "Stage A/B: normalizing + routing..."})
            gen = pipeline.stream_pcm_chunks(req)
            n_chunks = 0
            try:
                async for chunk in gen:
                    # Stop burning CPU/GPU the moment the client goes away.
                    if ws.client_state != WebSocketState.CONNECTED:
                        break
                    if n_chunks == 0:
                        await ws.send_json({"type": "progress", "phase": "C",
                                            "message": "Stage C: first audio chunk — playing live"})
                    n_chunks += 1
                    await ws.send_bytes(chunk)
                    if n_chunks % 20 == 0:
                        await ws.send_json({"type": "progress", "phase": "C",
                                            "message": f"Stage C: streaming... {n_chunks} chunks sent"})
            except asyncio.TimeoutError:
                logger.warning("stream normalize timeout")
                try:
                    await ws.send_json({"type": "error", "message": "normalize timeout"})
                except Exception:
                    pass
                continue
            finally:
                try:
                    await gen.aclose()
                except Exception:
                    pass
            if ws.client_state != WebSocketState.CONNECTED:
                break
            await ws.send_json({"type": "end", "chunks": n_chunks})
    except WebSocketDisconnect:
        logger.info("client disconnected")
    except Exception:
        logger.exception("stream error")
        try:
            await ws.send_json({"type": "error", "message": "synthesis failed"})
        except Exception:
            pass


def main():
    import uvicorn
    from movio.textnorm.normalizer import load_settings
    settings = load_settings()
    srv = settings.get("server", {})
    uvicorn.run(
        "movio.server.app:app",
        host=srv.get("host", "0.0.0.0"),
        port=int(srv.get("port", 8000)),
        log_level="info",
    )


if __name__ == "__main__":
    main()
