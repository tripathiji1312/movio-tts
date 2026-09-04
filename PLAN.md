# PLAN — Enable live streaming for movio IndicF5 TTS (first chunk immediately, not 2-min wait)

Date: 2026-09-04. Research log: `research/findings-2026-09-04.md`. Index: `research/INDEX.md`.

## 1. Objective & Success Criteria

**Objective:** Make `WS /tts/stream` (`movio/server/app.py:133-147`) truly incremental so the client hears/plays the first audio within milliseconds of request, instead of `start → ~2 min silence → burst → end`.

**Done looks like (all must hold on a warm server):**
- [ ] Single-utterance warm TTFA (WS connect-send → first `bytes` frame) `<=500 ms` for a short taxi-domain sentence (e.g. `Your cab will arrive in 10 minutes.`), measured by `bench/benchmark.py:87-108` `one_ws_request`.
- [ ] Chunks arrive incrementally (≥2 `send_bytes` before `end`, inter-chunk gap << total), each binary frame `2400 B` (`1200 samples × 50 ms @ 24000 Hz`, see §5 Phase 2 rule).
- [ ] No regression: `pytest tests/ -q` passes; streamed audio intelligible (quality gate §5 Phase 4); `POST /tts` timings still returned.
- [ ] Cold-start path no longer masquerades as streaming stall: `/healthz` reports readiness; first request either waits on explicit warmup or returns `error` fast, never silent 2-min hang.

How to verify: run §7 checklist top-to-bottom; `bench/results.json` `verdict.all_levels_within_target` is informative but **concurrency 15–20 p99≤500 is OUT of v1 scope on CPU** (see §2).

## 2. Locked Scope & Constraints (CPU-first update 2026-09-04: user approved GPU later, CPU now, 12 steps OK if natural — I pick 12 Euler)

**In (v1 CPU-first, GPU-ready):**
- Wire existing `IndicF5Engine.synthesize_stream(text, voice_name, min_syl=8, max_syl=14)` (`movio/acoustic/indicf5_engine.py:233-263`) into `TTSPipeline.stream_pcm_chunks` (`movio/pipeline.py:231-254`) with per-chunk `float_to_pcm16` yield + 50 ms re-slice.
- Config locked for CPU: `config/settings.yaml:51-63` `device: "cpu"`, `num_flow_steps: 12`, `ode_method: "euler"`, `sway_sampling_coef: -1.0`, `cfg_strength: 2.0`, `speed: 1.0` (single line change from `24/midpoint`; restart required because `ode_method` bound at `load_model` `:145-154` and `synthesize_chunk` never forwards it `:214-225`). GPU later = same code, flip `device: "cuda"` only (engine already falls back `cuda→cpu` at `:122-127`).
- CPU runtime: keep single serial queue (`pipeline.py:139-159`), dedicated 1-thread stream executor, front-end in executor+timeouts, blocking warmup; do NOT add parallel DiT workers or `torch.set_num_threads`/OMP pinning in v1 (repo reads none); raise Docker `memory 2G→4G+` (2G OOMs 337M DiT+Vocos+CT2+NeMo), keep `cpus: 4`; `server.sample_rate` → `24000` to match engine (skip librosa resample).
- Server readiness + disconnect-cancel + sample-rate truth (24000 Hz) + docs fixes.
- Verification: unit + WS single + chunk audit + bench single/low-concurrency + 20-sample quality smoke → 100-sample gate for 12-step naturalness (fallback `16/euler` only if gates fail).

**Out (explicitly):**
- True frame-level incremental DiT (needs causal retrain; bidirectional DiT requires full context — see research Aspect 2). Rejected.
- Triton decoupled rewrite (stub only: `deployment/triton/model_repo/indicf5_streaming/1/model.py:28-46` single-response, no `send_response` loop; FastAPI never imports it). Rejected for v1.
- Per-chunk cache tier, Opus/SSE/REST-streaming endpoint, FP16/FP8/TensorRT, vLLM-Omni, multi-queue/GPU micro-batching, `/voices` real listing, frontend player rebuild. Deferred with reasons in §4.
- Promising `p99≤500 ms @ 15–20 concurrent on CPU serial queue` (`movio/pipeline.py:139-159` single worker). Impossible without GPU replicas; scoped to GPU follow-up.

**Constraints:** `python >=3.11,<3.14` (`pyproject.toml:10`); locked `fastapi 0.141.1 / uvicorn 0.52.4 / websockets 17.1 / redis 8.1.0` (`uv.lock`); gated `ai4bharat/IndicF5` requires `HF_TOKEN` env (`README.md:40`); `IndicF5Engine.__init__(config:dict)` only (`indicf5_engine.py:66`); never mix pip `f5-tts` with vendored `third_party/IndicF5` (`indicf5_engine.py:89-97` purges `f5_tts`).

## 3. Research Findings (evidence-backed)

- **All serving paths block today.** `POST /tts` (`app.py:91-111`) and `POST /tts/wav` (`app.py:114-130`) call `await pipeline.synthesize` (full utterance, `pipeline.py:173-229`) then return. `WS /tts/stream` (`app.py:133-147`) sends `start`, loops `stream_pcm_chunks → send_bytes`, sends `end` — but `stream_pcm_chunks` (`pipeline.py:231-254`, docstring `234-235` “Yields the full audio in one chunk”) does `audio = await _synthesize_queued(...)` (`:249`) then slices (`:252-254`). TTFA==E2E. Source: subagent-1 `file:line` above.
- **True chunker exists but dead.** `synthesize_stream(min_syl=8, max_syl=14)` (`indicf5_engine.py:233-263`) does `chunk_text` (`chunking.py:38-71` clause-first, 8–14 syllable budget) + per-chunk `synthesize_chunk→infer_process(nfe_step, sway, cfg, speed)` (`:200-226`) + 20 ms crossfade (`ms_to_samples` `audio.py:52-53`). Neither `pipeline.py` nor `app.py` calls it. Generic `engine_base.py:23-37` fallback also unused; `IndicF5Engine` does NOT inherit it (`class IndicF5Engine:` `:51`). Source: subagent-1/2/6.
- **Slowest config is live.** Shipped `config/settings.yaml:51-63` = `device cpu, num_flow_steps 24, ode midpoint`; code defaults (`indicf5_engine.py:66-74`) = `12/euler/sway -1.0/cfg 2.0`; docstring ladder (`:57-60`) `8 steps ~150–250 ms / 12 ~250–400 ms / 20 ~500–700 ms on T4`; Triton stub uses 12 (`config.pbtxt:50-53`). Upstream F5-TTS: NAR CFM-DiT, Sway, 16 NFE RTF 0.15 WER 2.53 vs 2.42@32 (arxiv 2410.06885); socket chunk-stream + auto-chunk long text (github SWivid/F5-TTS); EPSS 7-step RTF 0.030 (Interspeech 2025); Vocos single-pass whole-mel decode, no official streaming (`gemelo-ai/vocos`; community `streaming-vocos` only). Live URLs verified 2026-09-04. Source: subagent-2.
- **Transport contract.** `WS ws://localhost:8000/tts/stream` (bench default `:204`): send `{"text","voice"}` (`app.py:138-143`), recv `{"type":"start"}` (`:144`), N×`bytes` PCM16 (`:145-146` via `float_to_pcm16` `audio.py:18-20` clip×32767→`<i2`), `{"type":"end"}` (`:147`); empty→`error` (`:140-142`); disconnect logged (`:148-149`); other→`error synthesis failed` (`:150-155`). `samples_per_chunk = sample_rate*ws_chunk_ms//1000`, `ws_chunk_ms=50` (`pipeline.py:106,252`, `settings.yaml:16`). No `StreamingResponse`/SSE exists (`rg` finds only WS). Uvicorn invoked with host/port/log only (`app.py:163-168`); live flags `--ws auto|websockets|websockets-sansio|wsproto|none` default auto, `--ws-max-size 16777216` (uvicorn docs). Bench client must use `max_size=2**24` (`bench:87-108`). Browser demo `index.html:278-362` collects Blobs then `pcm16ToWav` on `end` (not live playback) + stale `22050` (`:307`). Source: subagent-3.
- **Bottleneck ranking.** (1) Cold lazy load on first request: background warmup (`app.py:26-38`, `_background_warmup :17-21`) + inline `warmup()` in `_synthesize_queued` (`pipeline.py:163-165`) + `hf_hub_download(ai4bharat/IndicF5)` + DiT(1024/22/16) + Vocos (`indicf5_engine.py:99-163`) + NLTK/CTranslate2/NeMo lazies (`en2ta.py:113-129,188-216`, `normalizer.py:72-87`) on CPU/24-midpoint = minutes. (2) Fake streaming (above). (3) Single serial queue (`pipeline.py:112-159`) head-of-line blocking; sweep `1,5,10,15,20` (`bench:111-126`) p99 explodes. (4) Front work blocks first chunk; streaming path runs norm/route sync on loop no timeout (`pipeline.py:240-241`) vs blocking path executor+`wait_for` (`:178-191`, timeouts `pipeline.py:108-110` defaults 200/150 ms vs yaml `50/50/500` `settings:71-76`, C never enforced). (5) Whole-utterance cache key `sha256(voice|steps|text)` (`audio_cache.py:62-65`) always called with `steps=0` (`pipeline.py:208,218`); streaming path no cache check; disk `phrase_cache` unused by pipeline (only `HybridEngine`). (6) No `half()`/FP8 in serving (`float16` only Triton shim `:14`); dtype float32 via upstream `load_checkpoint` (`utils_infer:175-185`); report admits FP8/TRT not done (`TECHNICAL_REPORT:84-87`). Source: subagent-4.
- **Test/bench reality.** `tests/test_pipeline.py:1-141` 23 offline tests (chunking/domain/numbers/LID/cache) — no stream/TTFA/concurrency/server tests. `bench/benchmark.py` defines TTFA first-`bytes` (`:87-108`), sweep+`summarize` (`:111-152`, hardcodes `sample_rate=24000` `:129`), `run_in_process` synthetic TTFA `A+B+C` (`:155-188`), verdict `p99≤500 + zero errors` (`:224-232`). `bench/cost_analysis.py` $/min converter (profiles a100 $3/l4 $1.6/t4 $0.9). `eval/` quality only (whisper-tamil WER, UTMOS, resemblyzer sim; gates MOS≥4.0/CS≥3.8/entity 100%/UTMOS≥base-0.05/WER≤base+2%/p99≤500@15-20 in `HUMAN_EVAL_PROTOCOL:33-42`). `StageTiming` (`pipeline.py:70-77`); `/tts` returns timings, `/tts/wav` headers, `/tts/stream` no timings. `scripts/test_synthesis.py:47` references nonexistent `pipeline.engine.stats` (will `AttributeError`; only `HybridEngine:128` has `cache_stats`). `scripts/test_english.py:2-5` hardcodes live `HF_TOKEN`. No `results.json/cost.json/quality.json` present. Source: subagent-5.
- **Exact signatures (anti-hallucination).** `IndicF5Engine(config:dict)` only; `synthesize(text, voice_name)->np.ndarray` sync (`:228`); `synthesize_chunk(text, voice:VoiceProfile, flow_steps|None)->np.ndarray` sync (`:200`); `synthesize_stream(text, voice_name|None, min_syl=8, max_syl=14)->Iterator[np.ndarray]` sync generator (`:233-239`, `yield audio` `:262`); `chunk_text(text, min_syl=8, max_syl=14)->list[str]` (`chunking:38`); base `synthesize_stream(text, chunk_words=6)` NOT applicable (no inheritance); `TTSPipeline(config:dict)`, `sample_rate=engine.SAMPLE_RATE` (`:103`), `ws_chunk_ms` from `server` (`:105-106`), `timeout_a/b` from `pipeline.stage_timeout_ms` (`:108-110`); `_synthesis_worker` offloads `engine.synthesize` (`:152-154`); `_synthesize_queued(text, voice_name)->np.ndarray` (`:161`); `stream_pcm_chunks(request:SynthesisRequest)` no `->`, yields `bytes` (`:231-254`); WS handler `tts_stream(ws:WebSocket)` accept/receive_json/send start/send_bytes/send end (`app.py:133-147`); `float_to_pcm16(np.ndarray)->bytes` (`audio:18`), `ms_to_samples(ms, sr)->int` (`:52`); `AudioCache.get/set(text, voice, steps)` + `make_key` (`:63-68,80-81`); settings keys `server{host,port,ws_chunk_ms:50,sample_rate:16000,opus_bitrate_kbps:24}`, `stage_c.indicf5{model_path,device,num_flow_steps,sway,cfg,ode,speed,voices_dir,default_voice}`, `cache{backend,redis_url,ttl,max_audio_bytes_mb}`, `pipeline{enable_cache,stage_timeout_ms{a,b,c}}`. Blocking if direct: engine sync gens + `normalize` + `route` + `transliterate_english_to_tamil`; non-blocking today: Stage A/B via executor+`wait_for`, Stage C via queued worker, `warmup` via executor (`pipeline:120,180,188,152-154`). Streaming path `:240-241` blocks loop. Correct loop: executor normalize/route → cs-strip+transliterate+`_auto_voice` (`:242-247` preserved) → `gen=engine.synthesize_stream(synth_text, voice_name)` → `await run_in_executor(None, next, gen)` until `StopIteration` → `yield float_to_pcm16(chunk)`. Source: follow-up subagent (read all 8 files).
- **Reviewer corrections applied.** 24000 Hz single truth; 12/euler decision; real-steps cache key + flush; kwargs call; drop `test_synthesis`/`REDIS_URL`/Triton claims; executor+`StopIteration`+timeouts+cs preservation; per-chunk cache OUT of v1; disconnect-cancel; ready-gate; secret rotation; re-slice rule; exact verify commands; voices stub disclosure; hybrid/mms comment fix; streaming test + quality gate; concurrency scoping. Source: plan-reviewer verdict FAIL→fixed below. Full text in research log.

## 4. Chosen Approach & Why

**Ship (a)+(c) CPU-tuned: chunk-level first-audio streaming + 12×Euler (my pick for fastest-natural CPU).** Rewrite `stream_pcm_chunks` to iterate `synthesize_stream` (prosody 8–14 syllable chunks, 20 ms crossfade) and `yield` each chunk’s PCM immediately (re-sliced to 50 ms transport frames), with runtime `12×Euler, sway -1.0, cfg 2.0, speed 1.0, device cpu`. Why 12: upstream F5-TTS paper WER 2.53@16 vs 2.42@32 (RTF 0.15 vs 0.31), repro 2.44@16 vs 2.37@32; EPSS shows 7–12 stable slight degradation, 7-plain cliff WER 4.16 UTMOS 3.36, ≤6 collapse; 12 is 2.67× faster than 32 and ~4× faster than current `24×midpoint` (midpoint=2 evals/step via `odeint_kwargs method` in `utils_infer.py:254-256`), sitting between safe 16 and edge 8 with no EPSS code needed. 8×Euler (4×) is preview-only; 16×Euler (2×) is fallback if smoke fails; 24×Euler offline/cache only. Add blocking warmup + executor offload + disconnect-cancel + 24000 Hz truth. TTFA drops from full-utterance cost to one-chunk cost with zero retraining, using only verified APIs. GPU later needs no code fork.

- **Rejected (b) true frame-level incremental DiT+streaming vocoder:** requires causal/block-causal retrain or distillation; released IndicF5 weights are bidirectional, duration-upfront; Vocos decode already ~3–14 ms/chunk vs DiT >90% latency — vocoder shim alone cannot fix TTFA. Research-only.
- **Rejected Triton rewrite now:** `config.pbtxt decoupled:True` but `model.py execute()` single-response, no streaming loop, FastAPI unwired; needs Triton server + client + export work. Keep file as stub; revisit for GPU scale-out.
- **Rejected FP16/FP8/TensorRT in v1:** no `half()` path in `indicf5_engine.py:145-156`; upstream stability for IndicF5 fork UNVERIFIED — needs WER/MOS A/B. Keep float32; revisit on GPU with quality gate.
- **Rejected per-chunk cache / SSE / Opus / multi-queue in v1:** undefined key, no endpoint, no backpressure design; adds risk without TTFA benefit. Scoped to follow-up (see §8).

## 5. Implementation Plan

### Phase 0 — Preconditions (do not skip; closes when baseline green)
- [ ] Tasks:
  - `ls` repo root; `cat config/settings.yaml`; `cat pyproject.toml | head -40`; confirm `HF_TOKEN` present in env only (`printenv HF_TOKEN | wc -c` non-zero; never commit).
  - Baseline: `pytest tests/ -q` (expect 23 pass; file `tests/test_pipeline.py:1-141`).
  - Inspect (no guessing loader): `python -c "import yaml; d=yaml.safe_load(open('config/settings.yaml')); import json; print(json.dumps({'server':d.get('server'),'indicf5':d.get('stage_c',{}).get('indicf5'),'cache':d.get('cache'),'pipeline':d.get('pipeline')}, indent=2))"`.
  - Check GPU: `nvidia-smi || echo NO_GPU`; `python -c "import torch; print(torch.cuda.is_available())"`.
  - Secret hygiene: `rg -n "hf_" scripts/ movio/ config/ || true` — must find hardcoded token in `scripts/test_english.py:2-5`; rotate it in HF dashboard, delete line, use env only (see Phase 3).
- [ ] Exact commands:
  ```bash
  cat config/settings.yaml
  pytest tests/ -q
  python -c "import yaml; d=yaml.safe_load(open('config/settings.yaml')); import json; print(json.dumps(d.get('stage_c',{}).get('indicf5'), indent=2))"
  nvidia-smi || echo NO_GPU
  ```
- [ ] Close test: `pytest` passes; `indicf5.device/num_flow_steps/ode_method` values recorded (expect `cpu/24/midpoint` before Phase 1).

### Phase 1 — Config locked for CPU: 12/euler/cpu + cache invalidation (my recommendation — no A/B needed to start)
- [ ] Tasks (file `config/settings.yaml:51-63`):
  - Set `stage_c.indicf5.device: "cpu"`, `num_flow_steps: 12`, `ode_method: "euler"`; keep `sway_sampling_coef: -1.0`, `cfg_strength: 2.0`, `speed: 1.0`. Restart process after edit (ode bound at load `:145-154`).
  - Set `server.sample_rate: 24000` (match `IndicF5Engine.SAMPLE_RATE`; kills librosa resample tax on this path).
  - Docker CPU: raise `docker-compose.yaml:12-16` `memory: 2G→4G` (keep `cpus: 4`); keep single queue; do NOT set `torch` threads/OMP (unread by repo).
  - GPU-ready note: flipping `device: "cpu"→"cuda"` later is the only change (fallback `:122-127` covers); no code fork.
  - Update stale header comment (`settings.yaml:1-11` says hybrid/MMS live, indicf5 offline-only) to: `Active runtime is IndicF5Engine (movio/pipeline.py:101); hybrid/mms retained but unwired`.
  - Invalidate stale cache: changing steps without key change falsely hits (bug `pipeline.py:208,218` steps=0 vs key `voice|steps|text` `audio_cache.py:62-65`). Flush: if `cache.backend==redis`, `redis-cli -u <redis_url> --scan --pattern 'movio:tts:*' | xargs -r redis-cli -u <redis_url> DEL`; always restart server (in-memory `MemoryCache` dies with process). Code fix for real-steps key lands in Phase 2.
  - Do NOT touch `scripts/test_english.py:24,45` (`ode midpoint/nfe 24` hardcode) except secret purge — it is not live config.
- [ ] Exact commands:
  ```bash
  python - <<'EOF'
  import yaml
  p='config/settings.yaml'
  d=yaml.safe_load(open(p))
  d['stage_c']['indicf5']['num_flow_steps']=12
  d['stage_c']['indicf5']['ode_method']='euler'
  # set device explicitly after checking torch.cuda above; example keeps file value unless GPU proven:
  # d['stage_c']['indicf5']['device']='cuda'
  open(p,'w').write(yaml.safe_dump(d, sort_keys=False, allow_unicode=True))
  print('wrote', d['stage_c']['indicf5'])
  EOF
  cat config/settings.yaml | sed -n '51,76p'
  ```
- [ ] Close test: re-run inspect command from Phase 0; assert `device==cpu`, `num_flow_steps==12`, `ode_method==euler`, `sway==-1.0`, `cfg==2.0`; server restarted; cache flushed (no `movio:tts:*` keys).

### Phase 2 — Pipeline: true per-chunk streaming (core fix; file `movio/pipeline.py:231-254`)
- [ ] Tasks (rewrite `stream_pcm_chunks` only; keep WS protocol `start/bytes/end` unchanged):
  1. Offload front work: `norm = await loop.run_in_executor(None, self.normalizer.normalize, request.text)` with `asyncio.wait_for(..., self.timeout_a)`; same for `router.route(norm.text)` with `timeout_b` (mirror `synthesize :178-191`; timeouts from `config.pipeline.stage_timeout_ms`, yaml `50/50 ms` wins over code defaults `200/150 ms`). On `TimeoutError` → `raise` to WS handler which sends `{"type":"error","message":"normalize timeout"}` (add mapping in Phase 3; do NOT silently fall back).
  2. Preserve mandatory text steps verbatim (`pipeline.py:242-247`): strip `self.router.cs_token` (`<cs>`), `transliterate_english_to_tamil(synth_text)`, `_auto_voice(request.voice, synth_text)` (currently pinned `ta_female_neutral` `:41-60`; keep pinned in v1, disclose stub).
  3. Whole-utterance fast-path with REAL steps: `steps = self.engine.num_flow_steps` (not `0`); `cached = await self.audio_cache.get(synth_text, voice_name or "default", steps)`; if hit → `audio = pcm16→float` (`np.frombuffer(cached, dtype="<i2>").astype(np.float32)/32768.0` as in `:208-211`) → re-slice to `samples_per_chunk` and `yield` immediately (TTFA ≈ cache lookup).
  4. Else incremental: `gen = self.engine.synthesize_stream(synth_text, voice_name, min_syl=8, max_syl=14)` — call with **kwargs** (Liskov: base uses `chunk_words`, IndicF5 uses `min_syl/max_syl`; do NOT pass `chunk_words`). Iterate on a **dedicated single** `ThreadPoolExecutor(max_workers=1)` (module-level, not default pool, to avoid starving loop and racing `_synthesis_worker` which still serves `POST /tts`): `chunk = await loop.run_in_executor(_stream_executor, next, gen, None)` — use `None` sentinel instead of catching `StopIteration` across threads is NOT safe; correct pattern: `await run_in_executor(_stream_executor, lambda: next(gen, None))`; break on `None`; skip empty (`len==0` continue).
  5. Transport framing rule (not optional): each prosodic `np.ndarray` chunk → optionally re-slice to `samples_per_chunk = max(1, self.sample_rate * self.ws_chunk_ms // 1000)` (`:252`; `ws_chunk_ms=50` from `server` `:105-106`). At `self.sample_rate == 24000` (`pipeline.py:103` = `engine.SAMPLE_RATE`, single truth — ignore `server.sample_rate 16000` stale) this is `1200 samples = 2400 B` per `send_bytes`. `yield float_to_pcm16(slice)` (`audio.py:18-20`). Record `first_yield_ms = now - t0` and `logger.info("stream TTFA %.1f ms ...")` (replaces misleading `:250` log which fired after full synth).
  6. After stream completes: accumulate chunks (list of float arrays → concat) and `asyncio.create_task(self.audio_cache.set(synth_text, voice_name or "default", steps, float_to_pcm16(full)))` mirroring `:217-220` (whole-key only; **per-chunk cache explicitly OUT of v1**). Respect `max_audio_bytes_mb:8` drop (existing `audio_cache.py:80-91`).
  7. First-`next` TTFA: `synthesize_stream` calls `self.load()` synchronously (`:233-247`); ensure model warm (Phase 3) so first `next` ≈ one-chunk `infer_process`, not download.
- [ ] Reference skeleton (uses ONLY verified APIs; adapt indentation/imports to file):
  ```python
  # movio/pipeline.py — inside class TTSPipeline (add: _stream_executor = ThreadPoolExecutor(max_workers=1) module-level)
  async def stream_pcm_chunks(self, request):
      import asyncio, time
      import numpy as np
      from movio.utils.audio import float_to_pcm16
      t0 = time.perf_counter()
      loop = asyncio.get_running_loop()
      norm = await asyncio.wait_for(loop.run_in_executor(None, self.normalizer.normalize, request.text), self.timeout_a)
      route = await asyncio.wait_for(loop.run_in_executor(None, self.router.route, norm.text), self.timeout_b)
      synth_text = route.normalized_text.replace(self.router.cs_token, "")  # keep existing strip semantics (:242-244)
      from movio.router.en2ta import transliterate_english_to_tamil
      synth_text = transliterate_english_to_tamil(synth_text)
      voice_name = self._auto_voice(request.voice, synth_text)
      steps = int(getattr(self.engine, "num_flow_steps", 12))
      if self.audio_cache is not None:
          cached = await self.audio_cache.get(synth_text, voice_name or "default", steps)
          if cached:
              audio = np.frombuffer(cached, dtype="<i2>").astype(np.float32) / 32768.0
              spc = max(1, self.sample_rate * self.ws_chunk_ms // 1000)
              for i in range(0, len(audio), spc):
                  yield float_to_pcm16(audio[i:i+spc])
              return
      gen = self.engine.synthesize_stream(synth_text, voice_name, min_syl=8, max_syl=14)
      first = True
      bufs = []
      try:
          while True:
              chunk = await loop.run_in_executor(_stream_executor, lambda: next(gen, None))
              if chunk is None:
                  break
              if len(chunk) == 0:
                  continue
              bufs.append(chunk)
              spc = max(1, self.sample_rate * self.ws_chunk_ms // 1000)
              for i in range(0, len(chunk), spc):
                  yield float_to_pcm16(chunk[i:i+spc])
              if first:
                  dt = (time.perf_counter()-t0)*1000
                  import logging; logging.getLogger(__name__).info("stream TTFA %.1f ms voice=%s len=%d", dt, voice_name, len(synth_text))
                  first = False
      finally:
          try: gen.close()
          except Exception: pass
          if bufs and self.audio_cache is not None:
              import numpy as _np
              full = _np.concatenate([_np.asarray(b, dtype=_np.float32) for b in bufs])
              asyncio.create_task(self.audio_cache.set(synth_text, voice_name or "default", steps, float_to_pcm16(full)))
  ```
  Also fix `synthesize` cache call sites (`:208,218`) to pass `steps` (same `self.engine.num_flow_steps`) instead of `0`.
- [ ] Close test: `python -m py_compile movio/pipeline.py`; `pytest tests/ -q` still passes; WS single (§7 #3) shows first `bytes` in ≤500 ms warm with ≥2 frames before `end`.

### Phase 3 — Server: readiness, cancel, rate truth, hygiene (file `movio/server/app.py`)
- [ ] Tasks:
  - Lifespan: replace background-only warmup (`:26-38`, `create_task(_background_warmup)`) with **blocking warmup + ready flag**: `await pipeline.warmup()` in `lifespan` before `yield` (accept slower boot), OR keep background but add `app.state.ready: bool`. Contract (new, document in code): `GET /healthz` returns `200 {"status":"ok","ready":true}` when `pipeline` set and `engine.is_ready()` (`indicf5_engine.py:165-171`) else `503 {"status":"starting","ready":false}`; `/engine/stats` (`:77-88`) adds `if pipeline is None: raise HTTPException(503, "warming")` (fixes cold `AttributeError`).
  - WS handler (`:133-155`): keep `receive_json({"text","voice"})` → `start` → `send_bytes*` → `end`; add `TimeoutError→{"type":"error","message":"normalize timeout"}` mapping; wrap `async for` in `try/except WebSocketDisconnect: gen.aclose()/return` + `finally: await gen.aclose()`-equivalent (ensure generator from Phase 2 `close()` called on disconnect so GPU stops); check `ws.client_state` before each `send_bytes` (backpressure minimal: no queue, fail fast).
  - Rate truth: never use `config server.sample_rate 16000` or demo `22050` for framing; always `pipeline.sample_rate` (24000). Fix `movio/server/static/index.html:307` `22050` → fetch `/engine/stats` `sample_rate` at runtime; add comment.
  - Cache wiring disclosure: `REDIS_URL` env is NOT read (`audio_cache.py:50-53` reads `config.cache.redis_url`). Either set `cache.backend: "redis"` + `redis_url: "redis://localhost:6379/0"` in `settings.yaml:65-69` for Docker (`docker-compose.yaml:9` provides `redis:7-alpine`), or add explicit `os.getenv("REDIS_URL")` mapping in `app.py` lifespan — do one, document which. Default stays `memory` for local.
  - Secret: rotate `HF_TOKEN` exposed in `scripts/test_english.py:2-5`, delete hardcoded line, use `os.environ["HF_TOKEN"]` only; `rg -n "hf_" scripts/` must return nothing.
  - Title/docs: `app.py:42` title `Hybrid CPU (disk cache + MMS-TTS, 16kHz)` → `movio TTS — IndicF5 streaming (24kHz)`; note `/voices` (`:72-74`) stays stub `["default"]` in v1 (real `config/voices/*` listing deferred; `_auto_voice` pinned `ta_female_neutral` disclosed).
- [ ] Exact commands:
  ```bash
  rg -n "hf_" scripts/ movio/ config/ || echo CLEAN
  curl -s localhost:8000/healthz | python -m json.tool
  curl -s localhost:8000/engine/stats | python -m json.tool
  ```
- [ ] Close test: cold boot returns `503 ready:false` then `200 ready:true` after warmup; `rg hf_` clean; `/engine/stats sample_rate==24000`.

### Phase 4 — Verification: streaming proof + no-regression (closes whole plan)
- [ ] Tasks (in order; stop on first red):
  1. `pytest tests/ -q` (offline baseline).
  2. Boot: `HF_TOKEN=$HF_TOKEN python -m movio &` (or `uvicorn movio.server.app:app --host 0.0.0.0 --port 8000 &`); wait for `ready:true`; `curl -s -X POST localhost:8000/tts -H 'Content-Type: application/json' -d '{"text":"உங்கள் pickup location எங்கே?"}' | python -m json.tool` — assert `timings.total_ms` present, `cache_hit` false→true on repeat. Do NOT use `scripts/test_synthesis.py` as-is (crashes on `.stats`); either fix `:47` to `is_ready` or skip.
  3. WS single + 20× TTFA + chunk audit (uses verified `bench/benchmark.py:87-108` helper):
  ```bash
  python -c "import asyncio; from bench.benchmark import one_ws_request; print(asyncio.run(one_ws_request('ws://localhost:8000/tts/stream','warmup sentence for load',None)))"
  python - <<'EOF'
  import asyncio
  from bench.benchmark import one_ws_request
  async def m():
      for i in range(20):
          ttfa,e2e,nb = await one_ws_request('ws://localhost:8000/tts/stream','Your cab will arrive in 10 minutes.',None)
          print(f'{i} ttfa={ttfa:.0f} e2e={e2e:.0f} bytes={nb}')
          assert ttfa < 500, f'TTFA {ttfa} >500 (warm single-stream gate)'
  asyncio.run(m())
  EOF
  python - <<'EOF'
  import asyncio, json, websockets
  async def m():
      async with websockets.connect('ws://localhost:8000/tts/stream', max_size=2**24) as ws:
          await ws.send(json.dumps({"text":"Your cab will arrive in 10 minutes.","voice":None}))
          sizes=[]
          async for msg in ws:
              if isinstance(msg, bytes): sizes.append(len(msg))
              else:
                  d=json.loads(msg)
                  if d.get("type")=="end": break
                  if d.get("type")=="error": raise AssertionError(d)
          print(sizes)
          assert len(sizes)>=2, "must stream incrementally, not one burst"
          assert all(s==2400 for s in sizes[:-1]), f"expected 2400B @24k/50ms, got {sizes}"
  asyncio.run(m())
  EOF
  ```
  4. Bench sweep (informative; v1 gate only `1,5`): `python bench/benchmark.py --url ws://localhost:8000/tts/stream --levels 1,5 --warmup 3 --out bench/results.json && cat bench/results.json` — require `ttfa_p99_ms<=500` and `n_errors==0` at 1 and 5. Full `--levels 1,5,10,15,20` recorded but allowed to exceed on CPU serial queue (note as follow-up needs GPU×3 per `deployment/docker-compose.yml:26-27`). Fix `bench/benchmark.py:129` hardcode by passing `--sample-rate 24000` if added, else document 2400B math.
  5. Quality gate for locked 12-step CPU (UNVERIFIED until run; my call: ship 12 unless this fails): `python eval/run_quality_eval.py --model ai4bharat/IndicF5 --testset eval/testsets/tanglish_transport_200.tsv --ref-audio config/voices/ta_female_neutral/ref.wav --num-samples 20 --out eval/quality_12step_smoke.json` then `--num-samples 100 --out eval/quality_12step.json`. Require human `MOS>=4.0/CS>=3.8/entity 100%` + `UTMOS >= base-0.05`, `WER <= base+2%` per `HUMAN_EVAL_PROTOCOL:33-42`. On fail ONLY: set `num_flow_steps: 16` keep `euler`, re-run smoke; promote 16 only if passes + `p99<=500`. Do NOT try `cfg 1.5` or `sway !=-1.0` (unvalidated). `python bench/cost_analysis.py --results bench/results.json --profile a100 --out bench/cost.json` for info.
- [ ] Close test: §7 boxes 1–6 checked; `bench/results.json` + `eval/quality_12step.json` stored as new baselines.

## 6. Risks & Mitigations

- **Prosody seams from per-chunk ref conditioning + 20 ms crossfade** (`indicf5_engine:249-262`): short/clause-less text yields oversized chunk or choppy joins. Mitigation: keep `min_syl 8/max_syl 14` + clause-first (`chunking:21-35,64-72`); quality smoke in Phase 4; on fail raise `min_syl` or fall back to full-synth for `<8-syl` inputs (`or [text]` path `:247`).
- **12/euler quality drop vs 24/midpoint:** UNVERIFIED until WER/UTMOS run. Mitigation: 20-sample smoke gate; revert ladder `12→16→24` or `cfg` tweak; cache versioned by real steps so rollback invalidates cleanly.
- **Stale cache false-hits after step change:** pipeline passed `0` always. Mitigation: Phase 1 flush + Phase 2 real-steps key; document `DEL movio:tts:*` on any step/device change.
- **Event-loop blocking / pool starvation:** sync `torch infer_process` + lazy `load()` inside generator. Mitigation: dedicated `max_workers=1` executor, `next(gen,None)` sentinel, `gen.close()` in `finally`, timeouts on front work, keep `_synthesis_worker` for POST paths (no shared-queue race in v1).
- **Client disconnect burns GPU:** WS had no cancel. Mitigation: Phase 3 `try/finally close()` + `client_state` guard; verify by killing client mid-stream and watching GPU idle (UNVERIFIED — must be checked).
- **Cold boot still minutes (downloads + CPU DiT):** background warmup hid it. Mitigation: blocking warmup + `503 ready:false` + pre-download in Docker build (`hf_hub_download` at build, `model-cache` volume per `deployment/docker-compose.yml`); document `HF_TOKEN` gated (`HF HUB ai4bharat/IndicF5` login required).
- **Rate mismatch playback garbage:** 24k vs 16k vs 22050. Mitigation: single truth `pipeline.sample_rate`; chunk audit asserts `2400B`; demo fetches `/engine/stats`.
- **Secret leak:** hardcoded `HF_TOKEN` in `scripts/test_english.py`. Mitigation: rotate now, purge, env-only, `rg` clean gate.
- **Concurrency illusion:** serial queue cannot meet `p99≤500 @20` on CPU. Mitigation: scope v1 to single/low-concurrency; scale-out via `deployment/docker-compose --scale tts=3` + Redis LRU + GPU (`nvidia/cuda:12.4.1`) as follow-up; do not advertise 15–20 on CPU.
- **`<cs>`/Latin leak → garbled Tamil:** streaming rewrite must preserve strip+transliterate. Mitigation: code review checklist item + LID test still passes + spot-listen.

## 7. Verification Checklist

- [ ] `pytest tests/ -q` passes (23 tests).
- [ ] `rg -n "hf_" scripts/ movio/ config/` clean (secret rotated).
- [ ] `cat config/settings.yaml` shows `num_flow_steps: 12`, `ode_method: euler`, device recorded; `python -c yaml inspect` matches.
- [ ] `curl -s localhost:8000/healthz` → `{"status":"ok","ready":true}` (503 while warming); `curl -s localhost:8000/engine/stats` → `sample_rate 24000`, `is_ready true`.
- [ ] `curl POST /tts` returns `audio_wav_base64 + timings{stage_a_ms,stage_b_ms,stage_c_first_chunk_ms,total_ms,cache_hit}`; repeat → `cache_hit true`.
- [ ] WS 20× `ttfa<500` passes; chunk audit `len>=2`, all but last `2400B`; `start→bytes*→end`, no `error`.
- [ ] `bench/benchmark.py --levels 1,5` verdict `p99≤500, errors 0`; full `1,5,10,15,20` recorded (may exceed on CPU — noted).
- [ ] `eval/quality_12step.json` (20-sample) meets `UTMOS≥base-0.05, WER≤base+2%`; else rolled to 16 steps and re-run.
- [ ] `bench/cost.json` generated (info).
- [x] Live demo page shipped 2026-09-04: `WS /tts/stream` emits `progress` (AB/C stages, 20-chunk heartbeat) + `end.chunks`; `static/index.html` plays each 50 ms chunk live via WebAudio (@ server `sample_rate`), shows timestamped stage log + TTFA/Total/Chunks + replay element; stats bar shows Engine/Rate/Steps/Status from extended `/engine/stats` (verified: 49-frame fresh stream, progress order intact, old `bench` client unaffected).
- [ ] Disconnect test: kill WS mid-stream, server logs disconnect, GPU returns idle, no `error` storm. UNVERIFIED — must be checked.
- [ ] `research/INDEX.md` updated; `bench/results.json` + `eval/quality_12step.json` stored.

## 8. Open Questions

- [ ] GPU target for prod (A100/L4/T4/CPU-only)? Determines `device`, batching, and whether `p99≤500 @15–20` is claimable. UNVERIFIED — decision needed (default v1: single-stream CPU/GPU-agnostic, concurrency via replicas).
- [ ] Accept 12-step quality tradeoff, or require 16 steps after listening? UNVERIFIED — needs Phase 4 smoke + native-listener MOS (`eval/HUMAN_EVAL_PROTOCOL.md`).
- [ ] Redis shared cache (`backend redis`) vs in-memory per-replica? Affects repeat-stream hit rate under `--scale tts=3`. UNVERIFIED — default v1: memory local, redis opt-in via `settings.yaml:65-69`.
- [ ] Keep whole-utterance `POST /tts` on serial `_synthesis_worker` while WS streams bypass it — accept queue bypass in v1, or unify to chunk-granular queue later? Decision for follow-up.
- [ ] Client live-playback (play chunk N while N+1 generates) vs current demo buffer-till-`end` (`index.html:278-362`)? Frontend streaming player design deferred — needs product decision.
- [ ] FP16/TRT-LLM/distilled-steps adoption after quality A/B? No source confirms IndicF5-fork stability. UNVERIFIED — must be checked on GPU before claiming.
