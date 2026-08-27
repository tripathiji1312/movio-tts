# Technical Report — Self-Hosted Low-Latency Tamil / English / Tanglish TTS

> Status: **TEMPLATE** — fill every `[FILL]` from `bench/results.json`,
> `bench/cost.json`, `eval/quality.json`, and the human-eval sheets before
> submission. Sections map 1:1 to problem-statement deliverables 1–9.

## 1. Architecture (Deliverables 1–3)

**Chosen backbone:** IndicF5 (`ai4bharat/IndicF5`) — 337M CFM-DiT, Apache-2.0,
trained on 1,417 h across 11 Indian languages including Tamil.

**Pipeline:** text → Stage A normalization (WFST/NeMo/domain rules) →
Stage B Tanglish router (token LID + IndicXlit + `<cs>` boundaries +
gazetteer) → Stage C chunked flow-matching synthesis → Stage D streaming
PCM16/Opus over WebSocket. Sub-sentence Redis cache for boilerplate.

**Why this over alternatives:**

| Alternative | Rejected because |
|---|---|
| Indic Parler-TTS (938M) | Higher per-request latency; code-mix implicit only |
| VITS/FastSpeech2 cascade | Lower naturalness (MOS 3.5–3.8); kept as future fallback tier |
| Commercial APIs | Violates self-hosting constraint; 4–9× cost at target concurrency |

Full rationale: `TTS-Solutions-Blueprint.md` §3, §7, §8.

## 2. Context-Aware Pronunciation (Deliverable 2)

Verified behaviors (`tests/test_pipeline.py`):

| Input | Spoken output |
|---|---|
| `Your OTP is 4821.` | four eight two one |
| `booking ID is TN45AB1234` | T N four five A B one two three four |
| `9876543210` | nine eight seven six five four three two one zero |
| `7:30 PM` | seven thirty PM / மாலை ஏழு மணி முப்பது நிமிடம் |
| `25/08/2026` | August twenty-five, twenty twenty-six |
| `4.5 km`, `Rs. 250` | four point five kilometers / இருநூறு ஐம்பது ரூபாய் |

WFST backend (indic-text-normalization / NeMo) activates automatically when
installed; deterministic rule engine is always the final pass.

## 3. Performance Benchmark (Deliverable 5)

Protocol: `bench/benchmark.py` — 8 fixed transport-domain sentences,
concurrency sweep {1, 5, 10, 15, 20}, TTFA = request sent → first playable
audio chunk. Hardware: [FILL].

| Concurrency | TTFA p50 | p95 | p99 | E2E p99 | req/s | audio-min/h |
|---|---|---|---|---|---|---|
| 1 | [FILL] | | | | | |
| 5 | | | | | | |
| 10 | | | | | | |
| 15 | | | | | | |
| 20 | | | | | | |

Resource utilization: CPU [FILL]%, RAM [FILL] GB, GPU mem [FILL] GB.
**Verdict vs ≤500 ms p99 target:** [FILL].

## 4. Cost Analysis (Deliverable 6)

From `bench/cost.json` (profile: [FILL], $[FILL]/h):

| Concurrency | audio-min/h | $/min raw | $/min w/ cache | vs commercial ($0.015–0.040/min) |
|---|---|---|---|---|
| 15 | [FILL] | | | |
| 20 | | | | |

Conclusion: [FILL — expected ~$0.004–0.005/min ≈ 4–9× cheaper].

## 5. Quality Evaluation (Deliverable 7)

Objective (`eval/quality.json`): WER [FILL], CER [FILL], UTMOS [FILL],
speaker similarity [FILL]. Per-category WER incl. code-mix categories: [FILL].

Human eval (`eval/HUMAN_EVAL_PROTOCOL.md`, n=[FILL] raters): MOS [FILL],
code-switch smoothness [FILL], structured-entity accuracy [FILL]%.
Fine-tuned (v2 LoRA on Tanglish corpus) vs base delta: [FILL].

## 6. Trade-offs & Limitations

- v1 Tanglish quality depends on router script-unification; production-grade
  switch-point prosody requires the v2 LoRA (`training/README.md`).
- FP8/TensorRT-LLM quantization is specified but not yet benchmarked here;
  current serving is fp16.
- vLLM-Omni continuous batching pending upstream F5-TTS support; Triton
  micro-batching config provided as interim.
- Reported IndicF5 number-pronunciation edge cases are patched in Stage A;
  residual cases should be logged to the gazetteer/rules.

## 7. Production Recommendation (Deliverable 9)

[FILL after benchmarks — decision rule:]
- All gates pass → deploy Solution 1 as production backbone per blueprint §3.10.
- p99 fails only under load → add replicas / enable Triton batching / FP8.
- CS-smoothness < 3.8 → run the Kaggle LoRA fine-tune before launch.

## Reproducibility

```bash
pytest tests/ -q                                   # correctness of norm/router/cache
python bench/benchmark.py --url ws://... --out bench/results.json
python bench/cost_analysis.py --results bench/results.json --profile a100
python eval/make_testset.py                        # frozen test set
python eval/run_quality_eval.py --model ai4bharat/IndicF5 \
    --testset eval/testsets/tanglish_transport_200.tsv
```
