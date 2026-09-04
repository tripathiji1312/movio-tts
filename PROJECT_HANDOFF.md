# MOVIO TTS — Complete Project Handoff Document
> **Purpose:** Give this single file to another AI (or human) and it has everything needed to understand, run, modify, train, evaluate, benchmark, and deploy the project. No other context needed.
> Generated: 2026-09-04 from live codebase inspection (`movio-tts v0.2.0`, `movio/__init__.py` says `0.1.0`, server `app.py` says `0.4.0`).
> Repo root: `/home/tripathiji/projects/movio`

---

## TABLE OF CONTENTS
1. [Project Identity & Goal](#1-project-identity--goal)
2. [Tech Stack (exact)](#2-tech-stack-exact)
3. [Full Directory Structure](#3-full-directory-structure)
4. [Configuration Reference](#4-configuration-reference)
5. [Domain Concepts & Vocabulary](#5-domain-concepts--vocabulary)
6. [Inference Pipeline End-to-End (Stage A→B→C→D)](#6-inference-pipeline-end-to-end-stage-abc--d)
7. [Module-by-Module Function Reference](#7-module-by-module-function-reference)
8. [Server API Spec (REST + WebSocket)](#8-server-api-spec-rest--websocket)
9. [Caching System](#9-caching-system)
10. [Audio Utilities](#10-audio-utilities)
11. [Training Pipeline (Kaggle-only LoRA)](#11-training-pipeline-kaggle-only-lora)
12. [Evaluation Pipeline](#12-evaluation-pipeline)
13. [Benchmark & Cost Pipeline](#13-benchmark--cost-pipeline)
14. [Deployment & Infrastructure](#14-deployment--infrastructure)
15. [Models, Downloads, Output, Temp Folders](#15-models-downloads-output-temp-folders)
16. [Tests](#16-tests)
17. [CLI / Commands Cheat Sheet](#17-cli--commands-cheat-sheet)
18. [Env Vars, Ports, Volumes](#18-env-vars-ports-volumes)
19. [Known Drifts, Gotchas, Limitations](#19-known-drifts-gotchas-limitations)
20. [Reproduction Recipes for a New AI](#20-reproduction-recipes-for-a-new-ai)
21. [Glossary](#21-glossary)

---

## 1. Project Identity & Goal

- **Name:** movio-tts (`pyproject.toml:6-8` — description says "Solution 4 — VITS + FastSpeech2 cascaded" but actual active code is **Solution 1: IndicF5 Modular Polyglot TTS**).
- **License:** Apache-2.0 (`pyproject.toml:11`).
- **Python:** `>=3.11,<3.14` (`pyproject.toml:10`). Dev venv is CPython 3.11 via `uv 0.12.5`.
- **What it does:** Tamil / English / Tanglish (Tamil written in Latin script, code-mixed) Text-to-Speech for taxi/transport domain (OTP, booking ID, vehicle plate, phone, time, date, distance, fare). Target: `p99 TTFA ≤500ms`, `15-20 concurrent streams` on GPU, CPU fallback path.
- **Architecture slogan:** `Text → Stage A Normalize (WFST/rules) → Stage B Tanglish Router (LID+Xlit+<cs>) → Stage C IndicF5 chunked CFM-DiT → Stage D Vocoder+streaming PCM → WS/REST` (`README.md:9-14`, `movio/pipeline.py:1-18`).
- **Backbone model:** `ai4bharat/IndicF5` — 337M CFM-DiT, 24kHz, trained 1417h over 11 Indian languages. Consumed via vendored `third_party/IndicF5` (NOT pip `f5-tts` at runtime — see gotchas). Fallbacks: `facebook/mms-tts-tam` (80MB, 16kHz, CPU 50-150ms), legacy VITS/FastSpeech2 (not wired in current pipeline).
- **Design docs:** `README.md` (Solution-1 quickstart), `docs/TECHNICAL_REPORT.md` (template with `[FILL]` placeholders filled from `bench/results.json`, `bench/cost.json`, `eval/quality.json`), `TTS-Solutions-Blueprint.md` v1.1 (4 solutions comparison, S1 default lowest-risk ~435ms $0.0046/min; S2 Parler 938M; S3 MeloTTS+IndicF5+Parler+Kokoro cheapest; S4 VITS quick-win), `raw.md` (3-solution version), `training/README.md` (Kaggle LoRA), `eval/HUMAN_EVAL_PROTOCOL.md`.

---

## 2. Tech Stack (exact)

### 2.1 Canonical dependencies (`pyproject.toml:13-30`, `requirements.txt` shim)
| Category | Packages |
|---|---|
| Serving | `fastapi>=0.110` (locked `0.141.1`), `uvicorn[standard]>=0.29` (locked `0.52.4`), `websockets>=12` (locked `17.1`), `pydantic` (via FastAPI), `redis>=5` (locked `8.1.0`), `httpx>=0.27` (locked `0.28.1`), `psutil>=5.9` (locked `7.2.2`) |
| Audio/Num | `numpy>=1.26` (locked `1.26.4`), `soundfile>=0.12` (locked `0.14.0`), `librosa>=0.10` (locked `0.11.0`) |
| ML/TTS | `torch>=2.2` (locked `2.13.0`), `transformers>=4.40` (locked `5.16.1`), `huggingface_hub>=0.22` (locked `1.29.0`), `f5-tts>=1.1.0` (locked `1.1.22` — used for training conversion only), `safetensors>=0.4` (locked `0.8.0`), `vocos>=0.1.0` (locked `0.1.0`), `onnxruntime>=1.17` (in `requirements.txt` only), `ctranslate2` (transliteration backend), `nltk/cmudict` (pronunciation) |
| Optional/heavy (auto-detect, graceful degrade) | `pynini` + `indic-text-normalization` (WFST Stage A), `nemo_text_processing` (NeMo normalizer), `ai4bharat-transliteration` (`XlitEngine`, `IndicLID`), `fasttext`, `resemblyzer`, `jiwer`, `SpeechMOS` (`utmos22_strong`), `vasista22/whisper-tamil-base` |
| Dev | `pytest>=9.1.1` (locked `9.1.1`), `pytest-asyncio>=1.4.0` (locked `1.4.0`), `ruff line-length 100 py311` (`pyproject.toml:35-37`), `pytest.ini: asyncio_mode=auto, testpaths=tests` |
| Infra | `python:3.11-slim` (CPU dev image), `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` (GPU prod), `redis:7-alpine`, Triton Python backend, `ffmpeg`, `libsndfile1`, `curl` |
| Frontend | None. `package.json` is stub (`movio@1.0.0`, no JS). `movio/server/static/` may hold `index.html` if present. |

Install: `uv sync` (canonical) or `pip install -r requirements.txt` (Docker/Kaggle compat).

---

## 3. Full Directory Structure

```
movio/                          # installable package (setuptools find movio*)
  __init__.py                   # __version__=0.1.0 (stale vs pyproject 0.2.0)
  __main__.py                   # python -m movio → movio.server.app:main()
  pipeline.py                   # TTSPipeline orchestrator A→B→C
  textnorm/
    normalizer.py               # Stage A: WFST→NeMo→DomainRuleEngine
    domain_rules.py             # deterministic transport verbalization
  router/
    __init__.py (empty)
    tanglish_router.py          # Stage B: LID→Xlit→<cs> insertion
    lid.py                      # HeuristicLID + IndicLIDBackend wrapper
    xlit.py                     # XlitBackend wrapper (ai4bharat)
    en2ta.py                    # Latin→Tamil phonetic transliterator (ACTIVE, called by pipeline)
  acoustic/
    __init__.py                 # re-exports legacy VITS/FS2 engines
    engine_base.py              # AcousticEngine ABC
    indicf5_engine.py           # ACTIVE engine: CFM-DiT 337M + Vocos (24kHz)
    chunking.py                 # prosody chunking 8-14 syllables
    vocoder.py                  # Stage D: builtin→Vocos fallback
    hybrid_engine.py            # disk phrase_cache + MMS (NOT wired in pipeline.py currently)
    mms_engine.py               # facebook/mms-tts-tam CPU fallback (NOT wired currently)
    cascaded_engine.py          # legacy VITS→FS2 cascade (NOT wired)
    vits_engine.py              # legacy
    fastspeech2_engine.py       # legacy
    vits_text_processor.py      # legacy phoneme IDs
    transliterate.py            # legacy (superseded by router/en2ta.py)
    phrase_cache.py             # offline pre-synth ~200 taxi phrases → .pcm files
  cache/
    audio_cache.py              # Redis→in-memory LRU sub-sentence cache
  server/
    app.py                      # FastAPI: /healthz /voices /engine/stats /tts /tts/wav WS /tts/stream
    static/ (optional index.html)
  utils/
    audio.py                    # PCM16/wav/resample/crossfade/trim/stopwatch

config/
  settings.yaml                 # RUNTIME config (see §4)
  gazetteer/proper_nouns.txt    # place/driver names kept Latin
  voices/<voice>/voice.yaml + ref.wav   # 4 voices (only ta_female_neutral has ref.wav in checkout)

scripts/                        # ad-hoc tools
  download_models.py            # --models all|vits|fastspeech2 → models/
  download_model.py / download_model.sh
  export_onnx.py
  inspect_indicf5_repo.py / check_vocab_coverage.py / indicf5_vocab.txt
  test_tts.py / test_synthesis.py / test_english.py

training/
  README.md                     # Kaggle-only v2 LoRA guide
  configs/lora_tanglish.yaml    # LoRA hyperparams
  scripts/01_download_data.py … 06_evaluate.py
  kaggle/movio_kaggle_training.ipynb + run_all_kaggle.py

eval/
  make_testset.py               # deterministic 200-utt TSV
  run_quality_eval.py           # WER/CER + UTMOS + speaker-sim
  HUMAN_EVAL_PROTOCOL.md
  testsets/tanglish_transport_200.tsv

bench/
  benchmark.py                  # TTFA/E2E p50/p95/p99 @ conc 1,5,10,15,20
  cost_analysis.py              # $/audio-min vs commercial

tests/
  test_pipeline.py              # 26 tests, no weights needed

deployment/
  Dockerfile                    # GPU prod (CUDA 12.4)
  docker-compose.yml            # GPU stack (redis + tts x3)
  triton/model_repo/indicf5_streaming/{config.pbtxt,1/model.py}

third_party/IndicF5/            # vendored ai4bharat/IndicF5 fork of f5-tts
models/                         # runtime weights (see §15)
downloads/ output/ temp/
Dockerfile (root, CPU slim) / docker-compose.yaml (root, CPU)
pyproject.toml / requirements.txt / uv.lock / pytest.ini
README.md / docs/TECHNICAL_REPORT.md / TTS-Solutions-Blueprint.* / raw.md
```

---

## 4. Configuration Reference

### 4.1 `config/settings.yaml` (annotated, verbatim keys)
```yaml
server:
  host: "0.0.0.0"; port: 8000; ws_chunk_ms: 50
  sample_rate: 16000        # NOTE: stale comment — pipeline.py uses IndicF5Engine.SAMPLE_RATE=24000
  opus_bitrate_kbps: 24
stage_a.text_normalization:
  use_wfst: true; wfst_language: "ta"; nemo_fallback: true; domain_rules: true
  gazetteer_path: "config/gazetteer/proper_nouns.txt"
stage_b.router:
  lid_backend: "auto"; xlit_backend: "none"
  lid_target_langs: ["ta","en","ta_roman"]
  insert_cs_tokens: true; cs_token: "<cs>"
  keep_proper_nouns_latin: true
  gazetteer_path: "config/gazetteer/proper_nouns.txt"
  max_xlit_word_len: 24
stage_c.hybrid.cache_dir: "models/phrase_cache"; default_voice: "ta_female_neutral"
stage_c.mms.model_id: "facebook/mms-tts-tam"
stage_c.indicf5:
  model_path: "base"        # "base"=HF ai4bharat/IndicF5 else bundle dir with model.pt
  device: "cpu"; num_flow_steps: 24; sway_sampling_coef: -1.0
  cfg_strength: 2.0; ode_method: "midpoint"; speed: 1.0
  voices_dir: "config/voices"; default_voice: "ta_female_neutral"
cache: backend: "memory"; redis_url: "redis://localhost:6379/0"
  ttl_seconds: 604800 (7d); max_audio_bytes_mb: 8
pipeline.enable_cache: true
pipeline.stage_timeout_ms: {stage_a: 50, stage_b: 50, stage_c: 500}
```
Loader: `movio/textnorm/normalizer.py:121-123 load_settings()` reads `config/settings.yaml`.

### 4.2 Voices — `config/voices/<name>/voice.yaml`
Schema: `{name, ref_audio: ref.wav, ref_text: <Tamil transcript 5-8s>, gender, style}` (+ extracted `speaker_id,duration_s,quality_score` from training `02`). 4 voices: `ta_female_neutral` (+`ref.wav` present), `ta_english_natural`, `ta_service_calm`, `ta_tanglish_natural`. Zero-shot prompt for IndicF5: ref audio + ref text conditions the CFM-DiT. Loaded by `movio/acoustic/indicf5_engine.py:33-48 load_voice_profiles()` scanning `voices_dir/*/voice.yaml`.

### 4.3 Gazetteer — `config/gazetteer/proper_nouns.txt`
One proper noun per line (Chennai places, driver names). Used in two places: `TextNormalizer._expand_abbreviations` guard + `TanglishRouter._is_gazetteer(tanglish_router.py:50-60)` → label `proper_noun`, kept Latin, never transliterated.

### 4.4 `training/configs/lora_tanglish.yaml`
`model.base: ai4bharat/IndicF5 trust_remote_code`; `lora: rank32 alpha64 dropout0.05 target_modules_patterns=[attention.qkv, attention.proj, attention.out, feed_forward.w1/w3/w2] bias none`; `data: manifest_train/val audio_dir sr24000 min1.0 max12.0 ref_pool num_ref4`; `train: batch2 accum8 lr1e-4 warmup0.05 epochs6 clip1.0 decay0.01 cosine fp16 ckpt log20/eval200/save400 seed42`; `export.merged_output_dir`; `eval: utmos sarulab-speech/utmos22_strong, whisper vasista22/whisper-tamil-base, num100`.

### 4.5 Triton — `deployment/triton/model_repo/indicf5_streaming/config.pbtxt`
`backend:python, max_batch_size:6, decoupled:True`, inputs `TEXT (+optional VOICE):TYPE_STRING[1]`, output `PCM_CHUNK:TYPE_INT16[-1]`, `dynamic_batching max_queue 5ms preferred [4,6]`, `instance_group 3x KIND_GPU:0`, params `MODEL_ID=ai4bharat/IndicF5, NUM_FLOW_STEPS=12`.

---

## 5. Domain Concepts & Vocabulary

- **Stage A (Normalize):** expand abbreviations/numbers/dates/currency/phone/OTP/plates into speakable words. Always runs `DomainRuleEngine` last.
- **Stage B (Router):** token-level language ID → transliterate `ta_roman→ta_native` → keep `en/proper_noun` Latin → insert `<cs>` on `ta↔en` transitions → count `cs_boundaries`. `<cs>` is INTERNAL marker, stripped before acoustic model.
- **en2ta:** post-router Latin→Tamil phonetic pass (because IndicF5 only speaks Tamil script). 3-tier: OVERRIDES dict → CMUdict ARPA→Tamil → IndicXlit CTranslate2.
- **Stage C (Acoustic):** IndicF5 CFM-DiT mel-infill conditioned on ref voice → Vocos vocoder → 24kHz float32 waveform. `num_flow_steps` trades quality/speed (8 fast / 12 default streaming / 20-24 best).
- **Stage D (Vocoder/stream):** PCM16 chunking (`ws_chunk_ms=50` → 1200 samples @24kHz), 20ms crossfade between chunks, optional resample.
- **TTFA:** time-to-first-audio (request → first chunk bytes). **E2E:** request → last byte. **p50/p95/p99** over concurrency levels.
- **Voices:** zero-shot speaker prompts (ref.wav + ref_text), NOT trained speaker IDs.
- **Phrase cache (disk):** offline pre-synth of ~200 taxi templates → `sha256(voice|text).pcm` files, ~5ms hit. Distinct from `AudioCache` (Redis/in-memory runtime).
- **Tanglish:** Tamil in Latin script mixed with English, e.g. `unga OTP 4821` → `உங்கள் ...`. Categories in eval: `ta / en / ta_en_mix / ta_roman / ta_roman_en`.

---

## 6. Inference Pipeline End-to-End (Stage A→B→C→D)

### 6.1 Entry & lifecycle (`movio/__main__.py:1-4`, `movio/server/app.py:26-38,158-168`)
```
python -m movio
 → movio.__main__.main → movio.server.app:main()
 → load_settings() [config/settings.yaml]
 → uvicorn movio.server.app:app (host/port from settings)
 → lifespan: TTSPipeline(settings) + asyncio.create_task(_background_warmup)
 → warmup(): run_in_executor(engine.load) + Queue()+create_task(_synthesis_worker)
```
Warmup is background — server starts immediately, first request triggers lazy load if needed (`pipeline.py:163-165`).

### 6.2 `TTSPipeline.synthesize(request)` (`movio/pipeline.py:173-229`) — step-by-step
1. **Stage A** (`normalizer.normalize(text)` in threadpool, `wait_for(timeout_a)` default 50ms):
   `TextNormalizer.normalize(normalizer.py:95-118)`: `_detect_language` (TA-vs-Latin ratio) → WFST `indic_text_normalization.normalize` if `pynini` installed → NeMo `Normalizer(lang).normalize` if digits present → `DomainRuleEngine(language).normalize` (ALWAYS) → `_expand_abbreviations` (`St./Rd./Nr./opp.`).
   `DomainRuleEngine.normalize(domain_rules.py:124-134)` order: `TA_DIGITS→ASCII` → `_expand_booking_ids(BOOKING_ID_RE)` → `_expand_vehicle(VEHICLE_RE: TN45AB1234 → T N four five A B ...)` → `_expand_phone(PHONE_RE: 10-digit digit-wise)` → `_expand_time(TIME_RE: 7:30 PM → seven thirty PM + Tamil period காலை/மதியம்/மாலை/இரவு)` → `_expand_dates(DATE_SLASH_RE/DATE_TEXT_RE)` → `_expand_distances(4.5km)` → `_expand_currency(₹/Rs./ரூ)` → `_expand_numbers(DIGIT_RE: 4-8 digits digit-wise else tamil_number/english_number)`. Helpers: `tamil_number:28-52, _tens_join:55-63, english_number:70-89, _spell_alnum:142-150, _year_words:193-196, expand_otp (digit-wise)`.
   Returns `NormalizationResult(text, backend_used, latency_ms, warnings)`.
2. **Stage B** (`router.route(norm.text)` in threadpool, `wait_for(timeout_b)`):
   `TanglishRouter.route(tanglish_router.py:62-122)`: split tokens → if `_is_gazetteer(tok)` → `proper_noun` else `lid.classify(tok)` → if `ta_roman` and len≤max_xlit_word_len → `xlit.translit_word` → map to `ta/en/other` → insert `cs_token` on `ta↔en` transition, count `cs_boundaries`.
   `HeuristicLID.classify(lid.py:33-51)`: Tamil Unicode→`ta_native`, Devanagari→`hi_native`, `COMMON_TANGLISH_WORDS (unga/iruku/vanakkam...)`+regex+suffix+vowel heuristic→`ta_roman` else `en`. `IndicLIDBackend(lid.py:57-77)` tries IndicLID/fasttext else heuristic. `XlitBackend(xlit.py:14-42)`: `XlitEngine(src=en,beam4)` else `none` passthrough.
   Returns `RouterResult(normalized_text, token_labels, cs_boundaries, latency_ms)`.
3. **Strip + transliterate** (`pipeline.py:195-202`): `route.normalized_text.replace(<cs>," ")`, collapse spaces, then `transliterate_english_to_tamil(synth_text)` (`router/en2ta.py:281-306`): `_TIME_RE→_time_to_tamil`, `_DIGIT_SEQ_RE→_DIGIT_TAMIL`, `_LATIN_WORD_RE→_transliterate_word` = abbreviation `LETTER_NAMES (OTP→ஓ டீ பீ)` → `OVERRIDES (~50 loanwords booking→புக்கிங்)` → CMU `ARPAbet→Tamil (_ARPA_CONSONANT/_ARPA_VOWEL/_arpabet_to_tamil via nltk cmudict, N→ன் rule)` → IndicXlit-CT2 (`models/indicxlit_ct2` or `Singla0009/all-indic-transliteration`) → passthrough; per-word `_cache`.
4. **Voice:** `_auto_voice(norm.text, engine.voices)` (`pipeline.py:41-60`) — currently hardcoded `ta_female_neutral` (Latin-ratio branches commented out). Caller-specified `request.voice` overrides.
5. **Cache check:** `AudioCache.get(synth_text, voice, 0)` (`cache/audio_cache.py:67-78`, key `movio:tts:sha256(voice|steps|text)`). Hit → `np.frombuffer(pcm)<50ms`. Miss → `_synthesize_queued` (timed as `stage_c_first_chunk_ms`, `engine_used=indicf5`) → background `cache.set(float_to_pcm16)`.
6. **Stage C synth** via single-worker queue (`_synthesis_worker:139-159` serializes `engine.synthesize` in executor to avoid CPU/GPU thrash; `_synthesize_queued:161-169`):
   `IndicF5Engine.synthesize→synthesize_chunk(indicf5_engine.py:200-231)`: `load()` → `_get_ref_audio` cached `preprocess_ref_audio_text` → `infer_process(ref_audio,ref_text,text,model,vocoder,nfe_step=num_flow_steps,sway_sampling_coef,cfg_strength,speed)` → float32 24kHz.
   `load/_load(indicf5_engine.py:115-163)`: `_ensure_indicf5_path` (prepend `third_party/IndicF5`, purge pip `f5_tts`), `DiT(dim1024/depth22/heads16/ff2/text512/conv4)` + `load_model(mel=vocos,vocab,ode,use_ema)` + `load_vocoder(vocos)`, device `cuda→cpu` fallback; `model_path base→hf_hub_download ai4bharat/IndicF5 model.safetensors+vocab.txt` else bundle `model.pt (use_ema False)`.
   `synthesize_stream(indicf5_engine.py:233-263)`: `chunk_text(min8/max14)` + 20ms crossfade yield.
7. **Result:** `StageTiming(total_ms)` + `SynthesisResult(audio,sr,normalized,routed,timings)` (`pipeline.py:63-87`).
8. **Streaming variant** `stream_pcm_chunks(pipeline.py:231-254)`: sync normalize+route+xlit+auto-voice → queued synth → `yield float_to_pcm16 slices (sample_rate*ws_chunk_ms/1000)`; logs TTFA.

### 6.3 Latency budget (blueprint S1 §3)
`A70 + B35 + C280 + D50 = ~435ms cold`, `~155ms cache hit`. Timeouts enforce `A50/B50/C500ms`.

---

## 7. Module-by-Module Function Reference

| File | Class / Functions | Input → Output | Notes |
|---|---|---|---|
| `movio/__init__.py` | `__version__` | — | stale `0.1.0` |
| `movio/__main__.py` | `main` | — | re-export `server.app:main` |
| `movio/pipeline.py` | `SynthesisRequest(text,voice?,language_hint?)`, `StageTiming(a_ms,b_ms,c_ms,total_ms,cache_hit,engine_used)`, `SynthesisResult(audio,sr,normalized,routed,timings)`, `TTSPipeline(config).warmup/shutdown/_synthesis_worker/_synthesize_queued/synthesize/stream_pcm_chunks`, `_auto_voice(text,voices)` | text → waveform+timings | single-queue concurrency |
| `movio/textnorm/normalizer.py` | `NormalizationResult(text,backend_used,latency_ms,warnings)`, `TextNormalizer(use_wfst,nemo_fallback,gazetteer,domain_engine).normalize/_nemo_normalize/_detect_language/_expand_abbreviations`, `load_settings()` | raw → normalized | WFST→NeMo→rules chain |
| `movio/textnorm/domain_rules.py` | `DomainRuleEngine(language).normalize/_expand_booking_ids/_expand_vehicle/_expand_phone/_expand_time/_expand_dates/_expand_distances/_expand_currency/_expand_numbers/expand_otp/to_json`, `tamil_number(), english_number(), _spell_alnum(), _year_words()` | normalized → verbalized | Tamil native digits ௦-௯ handled |
| `movio/router/tanglish_router.py` | `RouterResult(normalized_text,token_labels,cs_boundaries,latency_ms)`, `TanglishRouter(insert_cs,cs_token,keep_proper,gazetteer,lid,xlit).route/_is_gazetteer` | norm → routed+`<cs>` | token LID loop |
| `movio/router/lid.py` | `HeuristicLID.classify/classify_batch`, `IndicLIDBackend.classify_tokens` | token → `ta_native/ta_roman/en/proper_noun/other` | auto fallback |
| `movio/router/xlit.py` | `XlitBackend.translit_word` | roman → native | no-op if missing |
| `movio/router/en2ta.py` | `transliterate_english_to_tamil(), _transliterate_word(), _cmu_transliterate(), _xlit_transliterate(), _arpabet_to_tamil(), _time_to_tamil(), _load_xlit(), _get_cmu()` + consts `_OVERRIDES, _ARPA_*, _DIGIT_TAMIL, LETTER_NAMES` | Latin-mixed → Tamil script | 3-tier, local no-network |
| `movio/acoustic/engine_base.py` | `AcousticEngine(sample_rate,is_ready,synthesize,synthesize_stream)` ABC | — | legacy interface |
| `movio/acoustic/indicf5_engine.py` | `VoiceProfile(name,ref_audio_path,ref_text)`, `load_voice_profiles()`, `IndicF5Engine.load/_load/_ensure_indicf5_path/get_voice/synthesize/synthesize_chunk/synthesize_stream`, `SAMPLE_RATE=24000` | text+voice → float32 wav | ACTIVE |
| `movio/acoustic/chunking.py` | `chunk_text(), split_at_clauses(), count_syllables_ta/en()` | long text → 8-14 syl chunks | clause-first |
| `movio/acoustic/vocoder.py` | `Vocoder(mel_to_audio, vocos_available)` | mel → wav | builtin→Vocos fallback |
| `movio/acoustic/hybrid_engine.py` | `HybridEngine.load/synthesize/synthesize_stream/cache_stats`, `_cache_key()` | text → pcm (disk ~5ms / MMS ~100ms) | NOT wired in pipeline |
| `movio/acoustic/mms_engine.py` | `MMSEngine.load/synthesize/synthesize_stream` | text → 16kHz wav resampled | `facebook/mms-tts-tam` |
| `movio/acoustic/phrase_cache.py` | `main(), _expand_templates(), _synthesize_indicf5/_synthesize_mms()`, `PHRASE_TEMPLATES` | templates → `.pcm` files | `python -m movio.acoustic.phrase_cache` |
| `movio/acoustic/cascaded_engine.py` | `CascadedEngine.load/synthesize/synthesize_stream/stats`, `_is_valid_audio(), _normalize_volume()` | text → wav | legacy VITS→FS2 |
| `movio/acoustic/vits_engine.py`, `fastspeech2_engine.py`, `vits_text_processor.py (text_to_sequence,get_vocab_size)`, `transliterate.py (transliterate_word, transliterate_english_segments)` | legacy | — | superseded |
| `movio/cache/audio_cache.py` | `MemoryCache.get/set (LRU 2048, TTL)`, `AudioCache.make_key/get/set` | text+voice → pcm bytes | Redis→memory |
| `movio/server/app.py` | `TTSPipeline pipeline, lifespan(), TTSRequest, healthz/voices/engine_stats/tts/tts_wav/tts_stream/main()` | HTTP/WS | see §8 |
| `movio/utils/audio.py` | `float_to_pcm16/pcm16_to_float/wav_bytes/resample(librosa)/crossfade/trim_silence/ms_to_samples/stopwatch` | — | primitives |
| `bench/benchmark.py` | `one_ws_request/run_concurrency_ws/summarize/ResourceSampler/run_in_process` | server → stats | TTFA/E2E |
| `bench/cost_analysis.py` | `PROFILES{a100 $3.0,l4 $1.6,t4 $0.9}` | results → $/min | cache-adjusted |
| `eval/make_testset.py` | `TEMPLATES, PLACES, main(--out --n --seed)` | slots → TSV | deterministic |
| `eval/run_quality_eval.py` | `synthesize/asr_transcribe/wer_cer/utmos_scores/speaker_similarity` | model+testset → JSON | WER/UTMOS/sim |
| `training/scripts/01…06` | see §11 | — | Kaggle-only |
| `scripts/*` | `download_models/test_tts/test_synthesis/test_english/inspect/check_vocab/export_onnx` | — | smoke/diag |

---

## 8. Server API Spec (REST + WebSocket)

Base: `http://<host>:8000` (`movio/server/app.py`). Title `movio TTS — Hybrid CPU`, version `0.4.0`.

| Method | Path | Req | Resp | Code ref |
|---|---|---|---|---|
| GET | `/` | — | `static/index.html` or `{"message":...}` | `app.py:52-58` |
| GET | `/healthz` | — | `{"status":"ok"}` | `app.py:67-69` |
| GET | `/voices` | — | `{"voices":["default"]}` (stub — real voices in engine) | `app.py:72-74` |
| GET | `/engine/stats` | — | `{"engine":"indicf5","model_path":...,"sample_rate":24000,"is_ready":bool}` | `app.py:77-88` |
| POST | `/tts` | `{"text":str,"voice"?:str,"language_hint"?:str}` | `{"audio_wav_base64":str,"sample_rate":int,"normalized_text":str,"routed_text":str,"timings":{a_ms,b_ms,c_ms,total_ms,cache_hit}}` | `app.py:91-111` |
| POST | `/tts/wav` | same | `audio/wav` bytes + headers `X-Normalized-Text/X-Cache-Hit/X-Total-Ms` | `app.py:114-130` |
| WS | `/tts/stream` | `{"text":str,"voice"?:str}` per message | `{"type":"start"}` → binary PCM16 24kHz ~50ms chunks → `{"type":"end"}` / `{"type":"error"}` | `app.py:133-155` |

Examples:
```bash
curl -X POST localhost:8000/tts -H 'Content-Type: application/json' \
  -d '{"text":"unga OTP 4821"}'
curl -X POST localhost:8000/tts/wav -H 'Content-Type: application/json' \
  -d '{"text":"Driver varugirar"}' --output out.wav
# WS (JS): ws.send(JSON.stringify({text, voice})); onmessage: bytes=PCM16, json=start/end
```
Errors: `400 text is required` on empty; WS `{"type":"error","message":"empty text"|"synthesis failed"}`.

---

## 9. Caching System

Two independent layers — do not confuse:
1. **Disk phrase cache** (`HybridEngine`, `models/phrase_cache/*.pcm`): key `sha256(voice|text)[:16]` (`hybrid_engine.py:36-38`), value raw PCM16. Built offline by `phrase_cache.py` (~200 templates × ~32KB ≈ 6MB). Hit ~5ms. NOT consulted by current `TTSPipeline` (only `HybridEngine.synthesize` uses it).
2. **Runtime AudioCache** (`movio/cache/audio_cache.py:40-91`): key `movio:tts:sha256(voice|steps|text)` (`make_key:62-65`). Backend `config.cache.backend`: `redis` (`redis.asyncio`, `redis_url`) else `MemoryCache (LRU 2048, TTL 7d, 8MB cap)`. `pipeline.py:208,217-220` get/set around synthesis. Hit TTFA <50ms. `enable_cache:false` disables. Stats via `cache_stats()` if engine exposes.

---

## 10. Audio Utilities (`movio/utils/audio.py:8-71`)

- `float_to_pcm16(float32[-1,1]) → bytes<int16` / `pcm16_to_float` inverse.
- `wav_bytes(audio, sr) → WAV bytes` (used by `/tts`, smoke tests).
- `resample(audio, orig_sr, target_sr)` via librosa (e.g. MMS 16k→24k).
- `crossfade(a, b, ms=20)` chunk stitching; `trim_silence`; `ms_to_samples(ms,sr)`; `stopwatch()` context manager.

---

## 11. Training Pipeline (Kaggle-only LoRA)

Never run locally — Kaggle GPU only (`training/README.md`, `training/kaggle/run_all_kaggle.py` prints 9 cells, `movio_kaggle_training.ipynb` all-in-one). Base `ai4bharat/IndicF5` (gated, needs `HF_TOKEN` secret + custom dataset `audio/*.wav+transcripts.tsv`).

| Step | Script | Args | What it does |
|---|---|---|---|
| 01 | `01_download_data.py:207-292` | `--out --dataset rasa|indicvoices_r|both --language --max-gb` | Budget `(free-5) cap15GB`, `_download_files` pattern-match `Tamil/train-*.parquet` (Rasa) / `*Tamil*/ta/*` (indicvoices_r) via `HfApi.list_repo_tree+hf_hub_download`, corrupt-partial cleanup |
| 02 | `02_prepare_corpus.py:323-513` | `--raw-dir --out --sample-rate24000 --min-dur1.0 --max-dur12.0 --min-snr12.0 --quality-top-pct30 --rasa-styles --delete-parquets --val/test-frac0.03 --extract-ref` | `find_transcript_pairs` + style filter, `trim_silence (librosa.split 40dB)`, `snr_proxy`, `quality_score (SNR+clip+flatness)`, hash dedupe, `sf.write audio/utt_*.wav`, prune, shuffle split `train/val/test.csv (audio,text,duration_s,source)` + `ref_pool.csv` (200); `--extract-ref SPEAKER --voice-out` best 8-15s → `ref.wav+voice.yaml` |
| 03 | `03_build_dataset.py:16-65` | `--data-dir --sample-rate` | `metadata.csv (audio|text)`, Arrow `Dataset.from_pandas+Audio(24k)+ref_audio/ref_text` |
| 04 | `04_train_lora.py:476-498` | `--path f5tts|peft --config --data-dir --out --epochs6 --batch-size2 --list-modules` | `path_f5tts`: clone SWivid/F5-TTS 1.1.22, Arrow `movio_tanglish_char`, download IndicF5 `model.safetensors+vocab.txt`, strip `_orig_mod+vocoder`+add `initted/step`→converted, hardlink `ckpts/movio_tanglish/pretrained_*`, write `duration.json+vocab.txt` (assert space@0, Tamil coverage), patch `model/__init__`, free cache, `finetune_cli --exp F5TTS_v1_Base --pretrain indicf5 --lr5e-6 --warmup200 --save50000 --last2000 --finetune` (6ep×20h ≈12-20 GPU-h T4 fp16). `path_peft`: PEFT r32 α64 dropout0.05 targets `attention.qkv/proj/out+feed_forward.w1/w3/w2`, AdamW+amp+accum8 (experimental) |
| 05 | `05_merge_export.py:31-154` | `--ckpt --f5tts-dir --out` | Load `.pt (ema/model_state_dict)` or `.safetensors`, strip `ema_model.+skip initted/step/mel fb`, save `model.pt`, copy `vocab.txt`, write `config.json (F5TTS_v1 mel100 sr24k finetune_base IndicF5)` |
| 06 | `06_evaluate.py:260-300` | `--model --f5tts-dir --test --ref-pool --num-samples50 --out --skip-wer` | `load_f5tts_model (CFM DiT1024/22/16 + vocos)`, `synth_samples (infer_process+preprocess_ref)`, `utmos_scores (SpeechMOS utmos22_strong)`, `wer_scores (whisper-tamil-base+jiwer)`; gate `UTMOS≥base-0.05 AND WER≤base+2% + 30 Tanglish spot-check` |

After training: set `stage_c.model_id: /models/indicf5_tanglish_merged` (same `AutoModel trust_remote_code` path).

---

## 12. Evaluation Pipeline

- **Testset:** `python eval/make_testset.py --out eval/testsets/tanglish_transport_200.tsv --n200 --seed42` — 18 `TEMPLATES` tags `ta/en/ta_en_mix/ta_roman/ta_roman_en` with slots (minutes/otp/booking_id/place/time_12/price/distance/phone/fee/vehicle_type; `PLACES/VEHICLE_TYPES`), deterministic `rng.choice`, TSV `id/text/category`. Frozen file committed.
- **Auto quality:** `python eval/run_quality_eval.py --model ai4bharat/IndicF5 --testset ... --ref-audio ... --ref-text ... --num-samples100 --out eval/quality.json [--skip-asr --skip-sim]` — `synthesize (AutoModel trust_remote_code →/tmp/movio_eval_audio 24kHz)`, `asr_transcribe (vasista22/whisper-tamil-base 16kHz)`, `wer_cer (jiwer lower+punct+spaces)`, `utmos_scores (SpeechMOS)`, `speaker_similarity (resemblyzer VoiceEncoder cosine)`, per-category WER → JSON.
- **Human:** `eval/HUMAN_EVAL_PROTOCOL.md` — 5+ Chennai natives, 200 or 60 stratified subset, blind randomized headphones ≤30min; 1-5 naturalness/pronunciation/EN-in-TA/CS-smoothness/intelligibility + binary entity accuracy; mean±95%CI per category; gates `MOS≥4.0 CS≥3.8 entity 100% UTMOS≥base-0.05 WER≤base+2% p99≤500ms@15-20`.
- **Report:** `docs/TECHNICAL_REPORT.md` §§1-9 map to deliverables; fill `[FILL]` from `bench/results.json`, `bench/cost.json`, `eval/quality.json`.

---

## 13. Benchmark & Cost Pipeline

- **Benchmark:** `python bench/benchmark.py --url ws://localhost:8000/tts/stream [--in-process] --levels1,5,10,15,20 --voice --warmup3 --out bench/results.json` — `DEFAULT_SENTENCES` 8 mixed; `one_ws_request:87-108` TTFA (first bytes) + E2E (last); `run_concurrency_ws (semaphore gather)`, `run_in_process (TTSPipeline.synthesize A+B+C as TTFA)`; `summarize (p50/p95/p99+req/s+audio-min/h)`; `ResourceSampler (psutil cpu/ram+torch gpu)`; verdict `p99≤500 && 0 errors`.
- **Cost:** `python bench/cost_analysis.py --results bench/results.json --profile a100|l4|t4|custom --usd-per-hour --cache-hit-frac0.35 --out bench/cost.json` — `PROFILES a100 $3.00 / l4 $1.60 / t4 $0.90` per hour → `$/audio-min` raw+cached vs commercial `$0.015-0.040` (expected `~$0.004-0.005 ≈4-9x cheaper`).

---

## 14. Deployment & Infrastructure

- **Root `Dockerfile` (CPU/dev):** `python:3.11-slim` + `libsndfile1 ffmpeg curl`, `pip install -r requirements.txt`, `RUN python scripts/download_models.py --models vits`, `EXPOSE 8000`, `HEALTHCHECK /healthz`, `CMD python -m movio`.
- **Root `docker-compose.yaml` (CPU):** `movio` (build `.`, `8000:8000`, `REDIS_URL=redis://redis:6379/0`, limits `2G/4cpu`, `./models:/app/models:ro`) + `redis:7-alpine` + `redis_data`.
- **`deployment/Dockerfile` (GPU prod):** `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` + `python3.11 ffmpeg libsndfile1 curl`, same port/healthcheck/cmd. Env `PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1`.
- **`deployment/docker-compose.yml` (GPU):** `redis` (8GB LRU) + `tts` (build `../`+`deployment/Dockerfile`, `HF_TOKEN`, `NVIDIA_VISIBLE_DEVICES=all`, volumes `../config:/app/config model-cache:/root/.cache/huggingface`, scale `up -d --scale tts=3` ≈15-20 streams on A100-40GB FP16).
- **Triton:** `deployment/triton/...` (see §4.5). `deployment/.../1/model.py:TritonPythonModel.initialize(model_id,NUM_FLOW_STEPS,voices.json)→execute(TEXT,VOICE→PCM_CHUNK)` loads `AutoModel.from_pretrained(MODEL_ID, fp16|cuda)`.

---

## 15. Models, Downloads, Output, Temp Folders

- **`models/` (pre-downloaded in this checkout):** `vits/tamil_vits/ (samprabin/tamil_vits)`, `indicf5_tanglish/{config.json,model.pt,vocab.txt}`, `indicf5_tanglish_v2/{config.json,model.pt,model/,vocab.txt}`, `indicxlit_ct2/{config.json,model.bin,source/target_vocabulary.json}` (CTranslate2), `phrase_cache/*.pcm` (6 cached: `090f… 5346… 7009… 7feb… 976b… 9c70…`). Created on demand (`mkdir parents` in `hybrid_engine.py:57`, `download_models.py:33,60`).
- **`downloads/`:** only CTranslate2 FP32 drops: `indicxlit_ct2_fp32/{model.bin,config.json,source/target_vocabulary.json}`, `indicxlit_indic_en_ct2_fp32/{...}` (+ `:Zone.Identifier` sidecars). Legacy staging (superseded by `01_download_data --out` + `download_model.py/.sh` HF gated fetch).
- **`output/`:** 6 smoke-test wavs `sample_01…06.wav` (from `scripts/test_synthesis.py --out output`, 6 `TEST_SENTENCES`).
- **`temp/`:** empty scratch dir (git-ignored).
- **`third_party/IndicF5/`:** vendored upstream backbone (`f5_tts/api.py,infer/,model/,train/,eval/,configs/,model.py,config.json,checkpoints/vocab.txt`) + `.cache/huggingface/` noise. Not app code.

---

## 16. Tests

`pytest tests/ -q` (26 tests, `tests/test_pipeline.py:1-141`, no server/weights):
- `TestChunking: chunk_text` — clause split `உங்கள் OTP 4821. ஓட்டுநர் வருகிறார்!` ≥1 non-empty; long Tamil ≥2; `Your driver is arriving now` lossless.
- `TestDomainRules: DomainRuleEngine` — Tamil digits `௧௦௦→நூறு`; `expand_otp("4821")=="four eight two one"`; `Rs.250→rupees/ரூபாய்`; `english_number(42)=="forty-two"`; booking `TN45AB1234→T N four five A B...`; 10-digit phone digit-wise; `7:30 PM→seven thirty PM`; Tamil `7:30→ஏழு+முப்பது+நிமிடம்`; `25/08/2026→August twenty-five + twenty twenty-six`; `4.5 km→four point five kilometers`.
- `TestTamilNumbers: tamil_number` — `5→ஐந்து 10→பத்து 100→நூறு 21→இருபத்தொன்று 45→நாற்பத்ஐந்து`.
- `TestHeuristicLID` — `உங்கள்→ta_native unga/iruku→ta_roman pickup→en`, batch `["unga","pickup","எங்கே"]==["ta_roman","en","ta_native"]`.
- `TestCache (asyncio)` — `set/get` roundtrip (`np.zeros(100,i2)`), `enable_cache:False→None`, TTL, `max_audio_bytes_mb:8`.
Aux (not pytest): `scripts/test_tts.py` (vocab 0-map + `text_embed` rows==`vocab+1` + synth>0.5s), `test_synthesis.py` (full `warmup()+synthesize()` → RTF/timings), `test_english.py` (DiT-22+vocos CPU, roman→Tamil), `check_vocab_coverage.py`/`inspect_indicf5_repo.py`, `export_onnx.py`.

---

## 17. CLI / Commands Cheat Sheet

```bash
# serve
python -m movio                                   # host/port from settings.yaml
uvicorn movio.server.app:app --host 0.0.0.0 --port 8000
# offline phrase cache
python -m movio.acoustic.phrase_cache --config config/settings.yaml --engine auto|indicf5|mms --voice default --extra-phrases FILE --dry-run
# smoke
python scripts/test_synthesis.py --text "..." --out output
python scripts/test_tts.py        # vocab/load/synth (models/indicf5_tanglish)
python scripts/test_english.py
python scripts/inspect_indicf5_repo.py; python scripts/check_vocab_coverage.py
python scripts/export_onnx.py --...
python scripts/download_models.py --models all|vits|fastspeech2
# tests / quality / perf
pytest tests/ -q
python eval/make_testset.py --out eval/testsets/tanglish_transport_200.tsv --n 200 --seed 42
python eval/run_quality_eval.py --model ai4bharat/IndicF5 --testset eval/testsets/tanglish_transport_200.tsv --ref-audio config/voices/ta_female_neutral/ref.wav --ref-text "..." --num-samples 100 --out eval/quality.json
python bench/benchmark.py --url ws://localhost:8000/tts/stream --levels 1,5,10,15,20 --warmup 3 --out bench/results.json
python bench/benchmark.py --in-process --levels 1,5,10,15,20
python bench/cost_analysis.py --results bench/results.json --profile a100 --cache-hit-frac 0.35 --out bench/cost.json
# training (Kaggle only)
python training/scripts/01_download_data.py --out ... --dataset indicvoices_r --language ta --max-gb 10
python training/scripts/02_prepare_corpus.py --raw-dir ... --out ... --extract-ref SPEAKER --voice-out config/voices/...
python training/scripts/03_build_dataset.py --data-dir ...
python training/scripts/04_train_lora.py --path f5tts --config training/configs/lora_tanglish.yaml --data-dir ... --out ...
python training/scripts/05_merge_export.py --ckpt ... --f5tts-dir ... --out /models/indicf5_tanglish_merged
python training/scripts/06_evaluate.py --model ... --test ... --ref-pool ... --num-samples 50
python training/kaggle/run_all_kaggle.py   # prints notebook cells
# deploy
cd deployment && HF_TOKEN=... docker compose up -d --scale tts=3
docker compose -f docker-compose.yaml up   # CPU root stack
```

---

## 18. Env Vars, Ports, Volumes

- **Ports:** `8000` app (`/healthz` healthcheck), `6379` redis.
- **Env:** `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` (gated IndicF5/indicvoices_r), `REDIS_URL` (default `redis://localhost:6379/0`, compose `redis://redis:6379/0`), `NVIDIA_VISIBLE_DEVICES=all`, `MODEL_ID=ai4bharat/IndicF5`, `NUM_FLOW_STEPS=12` (Triton), `PYTHONUNBUFFERED=1`.
- **Volumes/mounts:** `./models:/app/models:ro`, `../config:/app/config`, `model-cache:/root/.cache/huggingface`, `redis_data:/data`.

---

## 19. Known Drifts, Gotchas, Limitations

1. **Description drift:** `pyproject.toml:8` says Solution 4 (VITS+FS2) but active pipeline is Solution 1 (IndicF5). `settings.yaml` header says CPU-hybrid+MMS but `pipeline.py:28,101-103` wires only `IndicF5Engine (24kHz)`; `HybridEngine/MMS/Cascaded` exist but are NOT instantiated — available via `phrase_cache.py` or manual swap. `server.sample_rate 16000` comment is stale (engine is 24000). `app.py` title says Hybrid CPU but `engine_stats` reports `indicf5`.
2. **Version drift:** `pyproject 0.2.0` vs `movio/__init__ 0.1.0` vs `server/app 0.4.0`.
3. **`third_party/IndicF5` vs pip `f5-tts`:** runtime purges pip `f5_tts` and uses vendored path (`indicf5_engine.py:89-97`). Training `04` clones SWivid/F5-TTS separately. Do not mix.
4. **`<cs>` must be stripped** before acoustic model (`pipeline.py:195-198,242-244`) — router marker confuses TTS if passed through.
5. **IndicF5 only speaks Tamil script** — `en2ta.transliterate_english_to_tamil` is mandatory, not optional.
6. **`_auto_voice` hardcoded** to `ta_female_neutral` (language-mix branches commented `pipeline.py:51-60`).
7. **Single-queue worker** serializes ALL synthesis — good for p99 on CPU/GPU-short texts, but long-text head-of-line blocking possible. Timeouts `A50/B50` are tight; NeMo/WFST cold load can exceed → `asyncio.TimeoutError`.
8. **Legacy modules** (`acoustic/transliterate.py`, `vits_*`, `fastspeech2_*`, `cascaded_*`) are dead code paths unless manually wired — use `router/en2ta.py` instead.
9. **Limitations (TECHNICAL_REPORT §§6-7):** v1 router-only Tanglish (no LoRA yet), fp16 not FP8, no vLLM-Omni batching (manual Triton micro-batch 4-6), number edge-cases patched in Stage A, `package.json` stub, `temp/` empty.
10. **Blueprint exclusions:** XTTS-v2 CPML, F5-base CC-BY-NC, Spark CC-BY-NC-SA, Higgs non-commercial, Orpheus/Chatterbox Llama-inheritance, Emilia CC-BY-NC — do not introduce.

---

## 20. Reproduction Recipes for a New AI

**A. Run inference locally (CPU):**
1. `pip install -r requirements.txt` (or `uv sync`), set `HF_TOKEN` for gated `ai4bharat/IndicF5`.
2. `python -m movio` → `GET /healthz` → `POST /tts {"text":"unga OTP 4821"}` → check `normalized_text/routed_text/timings`.
3. Trace code path: `server/app.py:tts → pipeline.py:synthesize → normalizer.py:normalize → tanglish_router.py:route → en2ta.py:transliterate_english_to_tamil → indicf5_engine.py:synthesize_chunk → audio_cache.py:set`.

**B. Add a new domain rule (e.g. UPI IDs):**
1. Read `movio/textnorm/domain_rules.py:113-290`, add `_expand_upi` + regex + insert into `normalize()` chain + Tamil/English branches.
2. Add tests in `tests/test_pipeline.py::TestDomainRules`, run `pytest tests/ -q`.

**C. Swap acoustic engine to MMS (CPU fast path):**
1. Read `movio/acoustic/hybrid_engine.py:41-136` + `mms_engine.py:21-71`.
2. In `movio/pipeline.py:97-103` replace `IndicF5Engine` with `HybridEngine`, update `sample_rate`, `engine_used`, `requirements` (MMS via transformers `VitsModel`).

**D. Fine-tune for Tanglish (v2):**
Follow `training/README.md` + §11 strictly on Kaggle; gate with `06_evaluate.py` before merging.

**E. Prove p99 ≤500ms:**
`python bench/benchmark.py --in-process` then `--url ws://...` at levels 1,5,10,15,20 → `bench/cost_analysis.py --profile a100` → paste into `docs/TECHNICAL_REPORT.md [FILL]`.

---

## 21. Glossary

- **CFM-DiT:** Conditional Flow-Matching Diffusion Transformer (IndicF5 backbone).
- **Vocos:** neural vocoder (mel→waveform), fallback in `vocoder.py`.
- **WFST/Pynini:** weighted finite-state transducer for text normalization (Stage A).
- **NeMo:** NVIDIA text normalization fallback (Stage A).
- **LID:** language identification (Stage B). **Xlit:** transliteration roman→native (Stage B + en2ta).
- **`<cs>`:** code-switch boundary token inserted by router, stripped before TTS.
- **CMUdict/ARPA:** English pronunciation dictionary → Tamil phone mapping in `en2ta.py`.
- **TTFA/E2E/p99:** time-to-first-audio / end-to-end / 99th percentile latency.
- **UTMOS/WER/CER:** predicted MOS / word/character error rate (quality).
- **LoRA:** low-rank adaptation fine-tuning (training v2).
- **Triton:** NVIDIA inference server (deployment micro-batching).

---
*End of handoff. If anything is missing, the source of truth order is: live code (`movio/`) > `config/settings.yaml` > `README.md` > `docs/TECHNICAL_REPORT.md` > `TTS-Solutions-Blueprint.md` > this document.*
