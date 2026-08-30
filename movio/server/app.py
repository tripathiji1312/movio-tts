import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from movio.pipeline import SynthesisRequest, TTSPipeline
from movio.utils.audio import wav_bytes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("movio.server")

pipeline: TTSPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    from movio.textnorm.normalizer import load_settings

    settings = load_settings()
    pipeline = TTSPipeline(settings)
    try:
        await pipeline.warmup()
    except Exception as exc:
        logger.warning("Pipeline warmup deferred: %s", exc)
    yield
    if pipeline is not None:
        await pipeline.shutdown()
    pipeline = None


app = FastAPI(
    title="movio TTS — Hybrid CPU (disk cache + MMS-TTS, 16 kHz)",
    version="0.4.0",
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


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/voices")
async def voices():
    return {"voices": ["default"]}


@app.get("/engine/stats")
async def engine_stats():
    engine = pipeline.engine
    stats = {
        "engine": "hybrid",
        "model_path": engine.model_path,
        "sample_rate": engine.SAMPLE_RATE,
        "is_ready": engine.is_ready,
    }
    stats.update(engine.cache_stats())
    return stats


@app.post("/tts")
async def tts(req: TTSRequest):
    if not req.text.strip():
        raise HTTPException(400, "text is required")
    result = await pipeline.synthesize(
        SynthesisRequest(text=req.text, voice=req.voice, language_hint=req.language_hint)
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
        SynthesisRequest(text=req.text, voice=req.voice, language_hint=req.language_hint)
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
    await ws.accept()
    try:
        while True:
            msg = await ws.receive_json()
            text = (msg.get("text") or "").strip()
            if not text:
                await ws.send_json({"type": "error", "message": "empty text"})
                continue
            req = SynthesisRequest(text=text, voice=msg.get("voice"))
            await ws.send_json({"type": "start"})
            async for chunk in pipeline.stream_pcm_chunks(req):
                await ws.send_bytes(chunk)
            await ws.send_json({"type": "end"})
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
