import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
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
        logger.warning("Model warmup deferred: %s", exc)
    yield
    pipeline = None


app = FastAPI(title="movio TTS — Solution 1", version="0.1.0", lifespan=lifespan)


class TTSRequest(BaseModel):
    text: str
    voice: str | None = None
    language_hint: str | None = None


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/voices")
async def voices():
    return {"voices": list(pipeline.engine.voices.keys())}


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
    except Exception as exc:
        logger.exception("stream error")
        try:
            await ws.send_json({"type": "error", "message": str(exc)})
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
