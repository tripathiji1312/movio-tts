# movio — Solution 1: IndicF5 Modular Polyglot TTS

Production implementation of **Solution 1** from `TTS-Solutions-Blueprint.md`:
an IndicF5 (337M, CFM-DiT, Apache-2.0) based pipeline serving **Tamil,
English, and Tanglish** with sub-500 ms p99 TTFA and 15–20 concurrent
streams.

```
Text ─▶ Stage A ─▶ Stage B ─▶ Stage C ─▶ Stage D ─▶ WebSocket/REST audio
       Normalize  Tanglish   IndicF5     Vocoder +
       (WFST/     Router     (chunked    streaming PCM/
        rules)    (LID+Xlit   CFM-DiT)   Opus chunks)
                  +<cs>)
```

## Repository layout

| Path | Purpose |
|---|---|
| `movio/textnorm/` | **Stage A** — WFST hook → NeMo fallback → deterministic domain rules (OTP digit-wise, phone, time, vehicle plates, ₹ currency, Tamil native digits) |
| `movio/router/` | **Stage B** — token-level LID (IndicLID w/ heuristic fallback), IndicXlit romanized→Tamil Unicode, `<cs>` boundary insertion, proper-noun gazetteer |
| `movio/acoustic/` | **Stage C/D** — IndicF5 engine (lazy load, fp16, configurable flow steps), prosody-preserving syllable chunking, Vocos fallback vocoder |
| `movio/cache/` | Sub-sentence Redis cache (in-memory fallback) → cache-hit TTFA <50 ms on boilerplate |
| `movio/pipeline.py` | Orchestrator with per-stage latency accounting + async streaming generator |
| `movio/server/app.py` | FastAPI: `POST /tts`, `POST /tts/wav`, `WS /tts/stream` (50 ms PCM16 chunks) |
| `deployment/` | Dockerfile, docker-compose (+Redis), Triton decoupled-streaming model repo |
| `bench/` | Performance benchmark (p50/p95/p99 TTFA, concurrency sweep, resource sampling) + cost analysis ($/min vs concurrency) |
| `eval/` | Quality suite: frozen 200-utterance Tanglish test set, WER/CER + UTMOS + speaker similarity, human-eval protocol |
| `docs/TECHNICAL_REPORT.md` | Report template mapped to project deliverables 1–9 |
| `training/` | **Complete Kaggle training pipeline** — never run locally; see below |

## Quickstart

```bash
pip install -r requirements.txt

# 1. Add a reference voice (zero-shot cloning prompt for IndicF5):
#    config/voices/<name>/voice.yaml + ref.wav  (5-8s clean clip + transcript)

export HF_TOKEN=hf_...            # requires approved access to ai4bharat/IndicF5

python -m movio                   # serves on :8000
curl -s localhost:8000/tts -H 'content-type: application/json' \
     -d '{"text":"உங்கள் pickup location எங்கே?"}' | jq .timings
```

WebSocket streaming:

```js
const ws = new WebSocket("ws://localhost:8000/tts/stream");
ws.send(JSON.stringify({ text: "Unga OTP enna?", voice: "ta_female_neutral" }));
// binary frames = 24 kHz mono PCM16, ~50 ms each; JSON {"type":"end"} terminates
```

## Heavy backends (auto-detected, graceful degradation)

| Component | Install / requirement | Fallback if absent |
|---|---|---|
| IndicF5 | GPU + HF gated access | — (required for synthesis) |
| indic-text-normalization / Pynini | `pip install pynini` | built-in domain rules |
| NeMo ITN | `nemo_toolkit[tts]` | domain rules |
| IndicLID | per ai4bharat release | script/lexicon heuristic LID |
| IndicXlit | `pip install ai4bharat-transliteration` | tokens pass through untransliterated |
| Vocos | `pip install vocos` | built-in IndicF5 vocoder |
| Redis | service in `docker-compose.yml` | in-memory LRU cache |

## Production deployment

```bash
cd deployment && HF_TOKEN=hf_... docker compose up -d --scale tts=3
```

3 replicas/A100-40GB ≈ 15–20 concurrent streams. Triton artifacts in
`deployment/triton/model_repo/indicf5_streaming/` (`decoupled: True`,
dynamic batching 4–6, 5 ms queue delay) when you move off the FastAPI path.

## Training (Kaggle only)

The v2 Tanglish LoRA fine-tune is a **complete Kaggle-ready pipeline** under
[`training/`](training/README.md):

1. `01_download_data.py` — indicvoices_r Tamil subset (CC-BY-4.0)
2. `02_prepare_corpus.py` — resample 24 kHz, trim, SNR/duration filter, dedupe, splits
3. `03_build_dataset.py` — Arrow dataset + F5-TTS `metadata.csv`
4. `04_train_lora.py` — fine-tune via SWivid/F5-TTS trainer (**recommended**) or PEFT LoRA
5. `05_merge_export.py` — merge adapter → servable safetensors checkpoint
6. `06_evaluate.py` — UTMOS objective MOS + Whisper-Tamil WER gate

Cell-by-cell Kaggle notebook blocks: [`training/kaggle/run_all_kaggle.py`](training/kaggle/run_all_kaggle.py).
Setup checklist (GPU accelerator, internet ON, `HF_TOKEN` secret) is at the top of that file.

## Benchmarks & evaluation (project deliverables)

```bash
# 1. Latency/concurrency benchmark against a running server:
python bench/benchmark.py --url ws://localhost:8000/tts/stream --out bench/results.json

# 2. Cost analysis from measured throughput:
python bench/cost_analysis.py --results bench/results.json --profile a100 --out bench/cost.json

# 3. Objective quality (GPU machine):
python eval/make_testset.py                       # frozen 200-utterance test set
python eval/run_quality_eval.py --model ai4bharat/IndicF5 \
    --testset eval/testsets/tanglish_transport_200.tsv \
    --ref-audio config/voices/ta_female_neutral/ref.wav
```

Human evaluation protocol: `eval/HUMAN_EVAL_PROTOCOL.md`.
Report template with promotion gates: `docs/TECHNICAL_REPORT.md`.

## Tests

```bash
pytest tests/ -q      # 26 tests: chunking, normalization, numbers, LID, caching
```
