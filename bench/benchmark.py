"""Performance benchmark — problem-statement deliverable 5.

Measures, under a reproducible protocol:
  - TTFA  (request sent -> first playable audio chunk)
  - E2E   (request sent -> last audio byte)
  - p50 / p95 / p99 at concurrency 1, 5, 10, 15, 20
  - requests/sec, audio-minutes/hour
  - CPU %, RAM, GPU memory (sampled during the run)

Usage:
  # against a running server:
  python bench/benchmark.py --url ws://localhost:8000/tts/stream \
      --out bench/results.json

  # in-process (no server; same process as the pipeline):
  python bench/benchmark.py --in-process --out bench/results.json
"""

import argparse
import asyncio
import json
import statistics
import threading
import time
from pathlib import Path

DEFAULT_SENTENCES = [
    "Your cab will arrive in 10 minutes.",
    "உங்கள் pickup location எங்கே?",
    "Unga OTP enna? Please share it.",
    "Your booking ID is TN45AB1234.",
    "உங்கள் ஓட்டுநர் Chennai Central-ல இருக்கார், ஐந்து நிமிடத்தில் வருவார்.",
    "Your driver is arriving now. Please share your OTP.",
    "உங்கள் கேப் 7:30-க்கு வரும்.",
    "The trip fare is Rs. 250 and the distance is 4.5 km.",
]


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(q / 100 * (len(s) - 1)))))
    return s[k]


class ResourceSampler:
    def __init__(self, interval=1.0):
        self.interval = interval
        self.samples = {"cpu_pct": [], "ram_gb": [], "gpu_mem_gb": []}
        self._stop = threading.Event()

    def start(self):
        t = threading.Thread(target=self._run, daemon=True)
        t.start()
        return self

    def _run(self):
        import psutil

        proc = psutil.Process()
        while not self._stop.is_set():
            self.samples["cpu_pct"].append(proc.cpu_percent(interval=None))
            self.samples["ram_gb"].append(proc.memory_info().rss / 1e9)
            try:
                import torch

                if torch.cuda.is_available():
                    self.samples["gpu_mem_gb"].append(
                        torch.cuda.max_memory_allocated() / 1e9
                    )
            except Exception:
                pass
            time.sleep(self.interval)

    def stop(self):
        self._stop.set()
        return {
            k: {
                "mean": round(statistics.mean(v), 3) if v else None,
                "max": round(max(v), 3) if v else None,
            }
            for k, v in self.samples.items()
        }


async def one_ws_request(url: str, text: str, voice: str | None):
    import websockets

    ttfa = None
    audio_bytes = 0
    t0 = time.perf_counter()
    async with websockets.connect(url, max_size=2**24) as ws:
        await ws.send(json.dumps({"text": text, "voice": voice}))
        while True:
            msg = await ws.recv()
            if isinstance(msg, bytes):
                if ttfa is None:
                    ttfa = (time.perf_counter() - t0) * 1000
                audio_bytes += len(msg)
            else:
                data = json.loads(msg)
                if data.get("type") == "end":
                    break
                if data.get("type") == "error":
                    raise RuntimeError(data.get("message"))
    e2e = (time.perf_counter() - t0) * 1000
    return ttfa, e2e, audio_bytes


async def run_concurrency_ws(url: str, level: int, sentences, voice):
    sem = asyncio.Semaphore(level)

    async def worker(i):
        async with sem:
            text = sentences[i % len(sentences)]
            try:
                ttfa, e2e, nbytes = await one_ws_request(url, text, voice)
                return {"ttfa_ms": ttfa, "e2e_ms": e2e, "bytes": nbytes}
            except Exception as exc:
                return {"error": str(exc)}

    t0 = time.perf_counter()
    results = await asyncio.gather(*[worker(i) for i in range(level)])
    wall = time.perf_counter() - t0
    return [r for r in results], wall


def summarize(results, wall_s: float, sample_rate: int = 24000):
    ok = [r for r in results if "ttfa_ms" in r]
    errs = [r for r in results if "error" in r]
    ttfas = [r["ttfa_ms"] for r in ok]
    e2es = [r["e2e_ms"] for r in ok]
    total_bytes = sum(r["bytes"] for r in ok)
    audio_seconds = total_bytes / 2 / sample_rate
    return {
        "n_requests": len(results),
        "n_errors": len(errs),
        "ttfa_p50_ms": round(percentile(ttfas, 50), 1),
        "ttfa_p95_ms": round(percentile(ttfas, 95), 1),
        "ttfa_p99_ms": round(percentile(ttfas, 99), 1),
        "ttfa_mean_ms": round(statistics.mean(ttfas), 1) if ttfas else None,
        "e2e_p50_ms": round(percentile(e2es, 50), 1),
        "e2e_p95_ms": round(percentile(e2es, 95), 1),
        "e2e_p99_ms": round(percentile(e2es, 99), 1),
        "req_per_sec": round(len(ok) / wall_s, 3),
        "audio_minutes": round(audio_seconds / 60, 4),
        "audio_min_per_hour_at_this_load": round(audio_seconds / 60 / (wall_s / 3600), 1)
        if wall_s > 0
        else None,
        "errors": errs[:3],
    }


async def run_in_process(sentences, levels, voice):
    from movio.pipeline import SynthesisRequest, TTSPipeline
    from movio.textnorm.normalizer import load_settings

    pipeline = TTSPipeline(load_settings())
    await pipeline.warmup()
    out = {}
    sampler = ResourceSampler().start()
    for level in levels:
        sem = asyncio.Semaphore(level)

        async def worker(i):
            async with sem:
                t0 = time.perf_counter()
                req = SynthesisRequest(text=sentences[i % len(sentences)], voice=voice)
                res = await pipeline.synthesize(req)
                first_audio_ms = (
                    res.timings.stage_a_ms
                    + res.timings.stage_b_ms
                    + res.timings.stage_c_first_chunk_ms
                )
                return {
                    "ttfa_ms": first_audio_ms,
                    "e2e_ms": (time.perf_counter() - t0) * 1000,
                    "bytes": len(res.audio) * 2,
                }

        t0 = time.perf_counter()
        results = await asyncio.gather(*[worker(i) for i in range(level)])
        wall = time.perf_counter() - t0
        out[f"concurrency_{level}"] = summarize(results, wall)
        print(f"conc={level}: {out[f'concurrency_{level}']}")
    out["resources"] = sampler.stop()
    return out


async def run_ws(url, sentences, levels, voice):
    out = {}
    sampler = ResourceSampler().start()
    for level in levels:
        results, wall = await run_concurrency_ws(url, level, sentences, voice)
        out[f"concurrency_{level}"] = summarize(results, wall)
        print(f"conc={level}: {out[f'concurrency_{level}']}")
    out["resources"] = sampler.stop()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://localhost:8000/tts/stream")
    ap.add_argument("--in-process", action="store_true")
    ap.add_argument("--levels", default="1,5,10,15,20")
    ap.add_argument("--voice", default=None)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out", default="bench/results.json")
    args = ap.parse_args()

    levels = [int(x) for x in args.levels.split(",")]
    sentences = DEFAULT_SENTENCES

    async def go():
        if args.in_process:
            return await run_in_process(sentences, levels, args.voice)
        for i in range(args.warmup):
            await one_ws_request(args.url, "warmup", args.voice)
        return await run_ws(args.url, sentences, levels, args.voice)

    report = asyncio.run(go())

    passed = all(
        report[f"concurrency_{c}"]["ttfa_p99_ms"] <= 500
        and report[f"concurrency_{c}"]["n_errors"] == 0
        for c in levels
    )
    report["verdict"] = {
        "p99_ttfa_target_ms": 500,
        "all_levels_within_target": passed,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report["verdict"], indent=2))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
