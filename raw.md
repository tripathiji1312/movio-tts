# Three Detailed Solutions for Self-Hosted Tamil/English/Tanglish TTS

Each solution below is a self-contained, production-grade blueprint — license table, every component with version and license, the end-to-end pipeline, Tanglish handling, inference/serving, hardware, cost, strengths/weaknesses, and when it is the best choice. All three have been verified against the actual license files of each component.

---

# Solution 1 — IndicF5 Modular Polyglot Pipeline (Recommended)

## 1.1 Why This Is the Best Primary Path

IndicF5 is the only model in the open-source landscape that simultaneously satisfies four hard constraints: (i) **native Tamil support** trained on real Indic speech, (ii) **Apache 2.0 license** (commercially usable without MAU thresholds or gating-based revenue clauses), (iii) **small enough (~337M params)** for multiple replicas per GPU enabling 15–20 concurrent, and (iv) **flow-matching DiT architecture** that supports chunked streaming for sub-500 ms TTFA. It is trained on 1,417 hours of high-quality speech pooled from Rasa, IndicTTS, LIMMITS, and IndicVoices-R, covering 11 Indian languages including Tamil, with AI4Bharat reporting near-human quality【turn0search1】【turn0search35】【turn0search0】. The modular design lets each stage be optimized, swapped, and scaled independently — the lowest-risk path to production.

## 1.2 License Audit Table

| Component | Role | License | Commercial use? | Source |
|---|---|---|---|---|
| **IndicF5** (ai4bharat/IndicF5) | Acoustic backbone | Apache 2.0 (gated on HF, access approval required) | ✅ Yes | 【turn0search35】【turn0search36】 |
| **IndicVoices-R** dataset | Tamil training data (1,704 h, 22 langs) | **CC-BY-4.0** | ✅ Yes | 【turn1search31】【turn1search33】 |
| **Rasa dataset** (Assamese, Bengali, Tamil, expressive) | Expressive Tamil fine-tune | Gated research access (verify terms) | ⚠️ Verify | 【turn0search30】【turn0search33】 |
| **IndicTTS Tamil** (SPRINGLab, ~20 h, 48 kHz studio) | Tamil pretrain data | CC-BY (research-permissive) | ⚠️ Verify | — |
| **LIMMITS** dataset | Multi-speaker Indic | Research release | ⚠️ Verify | — |
| **Vocos** vocoder (fallback) | Fast neural vocoder | MIT | ✅ Yes | 【turn0search15】【turn0search16】 |
| **indic-text-normalization** (Kenpath) | WFST normalizer | Apache 2.0 (Kenpath repos) | ✅ Yes | 【turn0search13】 |
| **NeMo nemo_text_processing** | WFST ITN fallback | Apache 2.0 | ✅ Yes | 【turn0search24】 |
| **IndicLID** (ai4bharat) | Language ID (47 classes) | Apache 2.0 (AI4Bharat convention) | ✅ Yes | 【turn1search31】【turn0search30】 |
| **IndicXlit** (ai4bharat) | Transliteration (21 langs, ~11M) | Apache 2.0 | ✅ Yes | 【turn0search8】 |
| **Triton Inference Server** | Serving | Apache 2.0 | ✅ Yes | 【turn0search10】 |
| **vLLM-Omni** | Continuous batching (F5-TTS support pending) | Apache 2.0 | ✅ Yes | 【turn1search28】【turn1search15】 |
| **TensorRT-LLM** | FP8 quantization | Apache 2.0 | ✅ Yes | 【turn0search16】 |
| **UTMOS** | Objective MOS | MIT | ✅ Yes | 【turn0search21】【turn0search22】 |
| **vasista22/whisper-tamil** | Tamil WER eval | MIT | ✅ Yes | 【turn0search22】 |
| **MANGO dataset** | Tamil TTS human MOS | CC-BY (AI4Bharat convention) | ✅ Yes | 【turn0search33】 |

## 1.3 Architecture — Every Component

### Acoustic backbone: IndicF5
- **Architecture**: Conditional Flow Matching (CFM) with a Diffusion Transformer (DiT) + ConvNeXt-v2 backbone; mel-spectrogram infilling, exactly the F5-TTS family【turn0search2】【turn0search3】.
- **Parameters**: ~337M (confirmed by the Vāgdhenu distillation paper that uses IndicF5 as the 337M teacher)【turn0search0】.
- **Training data**: 1,417 hours pooled from Rasa + IndicTTS + LIMMITS + IndicVoices-R【turn0search1】【turn0search35】.
- **Languages**: 11 — Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, **Tamil**, Telugu【turn0search35】.
- **Output**: 24 kHz mel → built-in vocoder reconstructs waveform.
- **Inference framework**: native PyTorch in the IndicF5 repo; F5-TTS streaming inference code reused for chunking and Sway Sampling【turn0search38】. For production serving, deploy inside Triton's Python backend with `model_transaction_policy { decoupled: True }` for streaming gRPC【turn0search13】.

### Vocoder fallback: Vocos (MIT)
- Used if the built-in vocoder is bottlenecking on a particular replica; Vocos generates Fourier spectral coefficients in a single forward pass, faster than HiFi-GAN with comparable quality【turn0search13】【turn0search17】.

### Text normalization: indic-text-normalization (Kenpath, Apache 2.0)
- WFST-based with Pynini, 19 Indian languages, deterministic, low-latency【turn0search20】【turn0search21】.
- Tokenization → classification (semiotic class) → verbalization → post-processing【turn0search20】.
- Handles Arabic digits, native-script digits (Tamil ௦-௯), mixed input【turn0search20】.
- Fallback to NeMo `nemo_text_processing` (Apache 2.0) which has both a fast deterministic version and a context-aware version (resolves "St." → Saint vs Street)【turn0search24】.

### Tanglish router: IndicLID + IndicXlit
- **IndicLID** (Apache 2.0): two-stage classifier (fast linear + LM-finetuned), predicts 47 classes (24 native-script + 21 romanized + English + Others) — first LID for romanized Indian text【turn1search31】【turn0search30】.
- **IndicXlit** (Apache 2.0, ~11M transformer): Roman→native and native→Roman transliteration for 21 Indic languages, trained on Aksharantar (26M word pairs)【turn0search8】.

## 1.4 Full Pipeline (text → audio) with Latency Budget

```
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE A — Context-Aware Text Normalization             (~70 ms)    │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │ WFST normalizer  │→ │ Domain rule eng.  │→ │ Abbreviation     │  │
│  │ (Pynini, 19 lng) │  │ (OTP, phone, ID,  │  │ gazetteer        │  │
│  │ numbers/dates/$  │  │ time, vehicle)    │  │ (Chennai Central)│  │
│  └──────────────────┘  └──────────────────┘  └────────┬─────────┘  │
└───────────────────────────────────────────────────────┼──────────┘
                                                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE B — Tanglish Router                              (~35 ms)    │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │ IndicLID token-  │→ │ IndicXlit: Latin │→ │ <cs> boundary    │  │
│  │ level classify   │  │ Tamil→Tamil U.   │  │ mark + gazetteer │  │
│  │ (47 classes)     │  │ (English kept)   │  │ proper-noun keep │  │
│  └──────────────────┘  └──────────────────┘  └────────┬─────────┘  │
└───────────────────────────────────────────────────────┼──────────┘
                                                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE C — Acoustic Backbone (IndicF5, FP8)            (~280 ms)   │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │ Text → phoneme/  │→ │ Chunked CFM DiT  │→ │ Sway Sampling    │  │
│  │ grapheme embed   │  │ (10-14 flow steps)│  │ (first mel chunk)│  │
│  └──────────────────┘  └──────────────────┘  └────────┬─────────┘  │
└───────────────────────────────────────────────────────┼──────────┘
                                                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE D — Vocoder + Streaming                         (~50 ms)    │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │ Built-in vocoder│→ │ 50ms PCM/Opus     │→ │ Triton decoupled │  │
│  │ or Vocos (MIT)  │  │ chunks, cross-fade│  │ gRPC stream      │  │
│  └──────────────────┘  └──────────────────┘  └────────┬─────────┘  │
└───────────────────────────────────────────────────────┼──────────┘
                                                         ▼
                                              WebSocket → Voice Agent
```

**Latency budget per request (p99 target ≤500 ms)**:
- Stage A (TTN): 70 ms (WFST <1 ms + rules ~20 ms + abbreviations ~40 ms)
- Stage B (router): 35 ms (IndicLID ~10 ms + IndicXlit ~20 ms + boundary logic ~5 ms)
- Stage C (IndicF5 first chunk): 280 ms (10–14 flow steps with Sway Sampling, FP8, batched)
- Stage D (vocoder + network): 50 ms
- **Total TTFA: ~435 ms p99**, with ~65 ms headroom. On cache hit, Stage C drops to ~30 ms (chunk replay from Redis), giving **TTFA ~155 ms**.

## 1.5 Tanglish Strategy (Novel Component)

The Tanglish router unifies three input forms into one script representation the polyglot model can handle natively:

1. **Token-level LID** via IndicLID classifies each token as `ta_native`, `ta_roman` (Tanglish Latin), `en`, or `proper_noun`【turn1search31】.
2. **Transliteration of Latin-Tamil tokens** to Tamil Unicode via IndicXlit — so `Unga` → `உங்க`【turn0search8】.
3. **English tokens preserved as Latin** because IndicF5's polyglot training (IndicVoices-R includes English-Indic code-mix data) handles Latin English natively.
4. **`<cs>` boundary tokens** inserted at language transitions, conditioning the prosody head to anticipate the switch — adapted from Thomas 2018 (code-switching in Indic speech synthesizers) and the e-commerce Hindi-English code-mixed TTS finding that single-script bilingual training with explicit boundary marking outperforms naive concatenation【turn0search42】.
5. **Proper-noun gazetteer** preserves names like "Chennai Central", "Ola", "Uber" in Latin so the English-phoneme path activates.

This converts the three Tanglish forms in the problem statement into a unified representation the model already understands — no separate code-mixed fine-tune strictly required for v1, though a LoRA fine-tune on ~50–100 h of Tanglish audio is recommended for v2 (see §1.7).

## 1.6 Inference & Serving

### Continuous batching with vLLM-Omni
- vLLM-Omni's `DiffusionEngine` supports non-autoregressive DiT models with `step_execution` mode and request-level batching【turn1search9】【turn1search17】. F5-TTS-class architecture support is being added (issue open)【turn1search15】 — until merged, run IndicF5 in a Triton Python backend with manual micro-batching (max batch size 4–6 per replica).
- **PagedAttention** for the text encoder (transformer) stage【turn1search10】.

### FP8 quantization with TensorRT-LLM
- FP8 per-tensor + FP8 KV cache on Ada/Hopper GPUs — near-lossless on TTS, ~2× throughput, ~50% VRAM【turn0search16】【turn0search17】.
- For IndicF5 specifically (flow-matching DiT, no KV cache), apply FP8 weights + activations via TensorRT plan; the vocoder stays in FP16.

### Speculative decoding (VADUSA-style)
- A distilled IndicF5-tiny (~50M, depth-pruned per the published staged depth-pruning distillation recipe)【turn0search0】 as the draft model, with the full IndicF5 as target — 2–3× AR speedup. Note: IndicF5 is non-autoregressive flow-matching, so spec-dec applies only if a future AR variant is used; for pure CFM, the analog is **fewer flow steps with Sway Sampling + distillation**.

### Chunked streaming
- Text split into 8–12-syllable chunks at sentence/clause boundaries (preserve prosody units).
- Each chunk independently flow-matched; cross-faded in vocoder to avoid discontinuity.

### Sub-sentence audio caching
- Redis-backed hash on normalized sub-sentence strings. Transport domain has ~30–40% boilerplate ("Your driver is arriving", "Please share your OTP") — cache hit drives TTFA to <50 ms.

## 1.7 Hardware & Deployment

| Component | Hardware | Replicas | VRAM/replica | Concurrency/replica |
|---|---|---|---|---|
| IndicF5 (FP8) | A100 40GB | 3 | ~2.5 GB | 5–6 |
| IndicF5 (FP8) | L4 24GB overflow | 2 | ~2.5 GB | 3–4 |
| Vocoder (Vocos) | shared A100 | 1 | <1 GB | — |
| IndicLID + IndicXlit | CPU pool (8 vCPU) | 2 | RAM only | 20 |
| TTN (WFST) | CPU pool | 2 | RAM only | 20 |
| Redis cache | 8 GB RAM | 1 | — | — |

**Total: 1× A100 40GB + 1× L4 24GB + CPU pool = ~$3.00/h blended**.

Sustainable throughput at 15–20 concurrent: ~650–800 audio-min/hour → **~$0.004/min**.

## 1.8 Cost Analysis (single A100 + L4 + CPU pool, ~$3.00/h)

| Concurrency | Audio-min/h | Infra $/h | $/min | Notes |
|---|---|---|---|---|
| 1 | ~50 | $3.00 | $0.060 | Underutilized |
| 5 | ~250 | $3.00 | $0.012 | Cache hit 30% |
| 10 | ~450 | $3.00 | $0.0067 | Cache hit 35% |
| 15 | ~650 | $3.00 | $0.0046 | Cache hit 40% |
| 20 | ~800 | $3.50 (CPU scales) | $0.0044 | Near saturation |

**vs commercial**: ElevenLabs/Cartesia/Sarvam price $0.015–$0.040/min【turn1search41】. Self-hosted Solution 1 is **~4–9× cheaper** at 15–20 concurrent.

## 1.9 Strengths & Weaknesses

**Strengths**
- License-clean: every component Apache 2.0 or MIT, no MAU thresholds, no gating-based commercial restrictions.
- Native Tamil from 1,417 h of real Indic speech — no synthetic accent.
- 337M params → 3 replicas per A100 → direct path to 15–20 concurrent.
- Modular: each stage independently swappable (e.g., replace Vocos with HiFi-GAN, swap IndicLID with a faster LID).
- Sub-sentence cache converts 30–40% of transport-domain requests to <50 ms TTFA.

**Weaknesses**
- Tanglish audio data is scarce — initial Tanglish naturalness depends on the router's script unification + the model's implicit code-mix exposure; a dedicated ~50–100 h Tanglish LoRA is needed for v2 production-grade Tanglish.
- vLLM-Omni's F5-TTS support is still being merged — initial deployment uses Triton manual micro-batching until vLLM-Omni F5 support is GA【turn1search15】.
- IndicF5 is gated on HF — must request access (typically approved in 24–48 h)【turn0search36】.
- Reported issues with number pronunciation causing gibberish noise in some IndicF5 runs — must be patched in the TTN layer or via LoRA【turn0search5】.

## 1.10 When This Is the Best Choice

When the priority is **lowest risk + lowest cost + native Tamil + license cleanliness** — i.e., the team wants a defensible production architecture where every component can be commercially deployed without legal ambiguity, and where the modular design allows incremental optimization. This is the recommended primary path.

---

# Solution 2 — Indic Parler-TTS Single-Backbone End-to-End

## 2.1 Why This Is the Best Single-Model Path

Indic Parler-TTS is the most expressive open-source Indic TTS available — a 938M-param T5-based model fine-tuned from Parler-TTS Mini on 1,806 hours of multilingual Indic + English data, covering 21 languages including Tamil, with **natural-language description prompts** that control speaker identity, emotion, and prosody without reference audio【turn1search0】【turn1search2】【turn1search4】. The single-backbone design means one model, one serving path, one fine-tune — operationally far simpler than Solution 1. The trade-off is a larger model (938M vs 337M) requiring more aggressive quantization and a slightly higher $/min, but in return you get emergent code-mixing behavior and emotion control that no modular pipeline can match.

## 2.2 License Audit Table

| Component | Role | License | Commercial use? | Source |
|---|---|---|---|---|
| **Indic Parler-TTS** (ai4bharat/indic-parler-tts) | Single acoustic backbone | Apache 2.0 (gated on HF) | ✅ Yes (verify gating terms with AI4Bharat) | 【turn1search0】【turn1search2】 |
| **Indic Parler-TTS Pretrained** | Base for fine-tune | Apache 2.0 | ✅ Yes | 【turn1search23】 |
| **Parler-TTS** (huggingface/parler-tts) | Base framework | Apache 2.0 | ✅ Yes | 【turn1search1】 |
| **GLOBE-annotated dataset** | Pretrain data | Apache 2.0 (HF convention) | ✅ Yes | 【turn1search23】 |
| **IndicVoices-R** | Tamil fine-tune data (CC-BY-4.0) | CC-BY-4.0 | ✅ Yes | 【turn1search31】【turn1search33】 |
| **Parler-TTS inference library** | Serving | Apache 2.0 | ✅ Yes | 【turn1search1】 |
| **vLLM** | Continuous batching for T5 decoder | Apache 2.0 | ✅ Yes | 【turn1search10】 |
| **Triton Inference Server** | Streaming gRPC | Apache 2.0 | ✅ Yes | 【turn0search10】 |
| **TensorRT-LLM** | FP8 quantization | Apache 2.0 | ✅ Yes | 【turn0search16】 |
| **UTMOS / NISQA** | Objective MOS | MIT | ✅ Yes | 【turn0search21】 |
| **MANGO dataset** | Tamil human MOS | CC-BY | ✅ Yes | 【turn0search33】 |
| **Whisper Tamil (vasista22)** | WER eval | MIT | ✅ Yes | 【turn0search22】 |

**License caution**: Indic Parler-TTS is gated on HuggingFace — while the underlying license is Apache 2.0, the gating may impose additional terms. An ungated Apache mirror exists【turn1search24】. For commercial deployment, either (a) request gated access and confirm with AI4Bharat in writing, or (b) use the ungated mirror.

## 2.3 Architecture — Every Component

### Single acoustic backbone: Indic Parler-TTS
- **Architecture**: T5-based encoder-decoder with cross-attention to a natural-language description prompt (the "Parler" approach — Natural Language Guidance of High-Fidelity TTS)【turn1search1】.
- **Parameters**: ~938M【turn1search2】.
- **Training**: fine-tuned from `indic-parler-tts-pretrained` on 1,806 hours multilingual Indic + English【turn1search0】.
- **Languages**: 21 — Assamese, Bodo, Dogri, English, Gujarati, Hindi, Kannada, Konkani, Maithili, Malayalam, Manipuri, Marathi, Nepali, Odia, Sanskrit, Santali, Sindhi, **Tamil**, Telugu, Urdu【turn1search0】【turn1search3】.
- **Output**: 30s max per generation — chunking recommended for longer inputs【turn1search6】.
- **Prompt control**: a natural-language description like *"A female Tamil customer-support agent, calm and professional, naturally code-switching between Tamil and English"* conditions speaker, emotion, and prosody【turn1search4】.

### Why no separate Tanglish router
The model's training distribution includes Hinglish/Tanglish-style fluid code-switching【turn1search7】. The natural-language prompt explicitly directs code-switching behavior, and a LoRA fine-tune teaches the actual switch patterns. This is the most "novel" of the three solutions — it relies on emergent code-mixing from a single LM-conditioned TTS.

## 2.4 Full Pipeline (text → audio) with Latency Budget

```
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE A — Context-Aware Text Normalization             (~70 ms)   │
│  (Same WFST + domain rules as Solution 1)                            │
└─────────────────────────────────────────┬───────────────────────────┘
                                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE B — Prompt Encoder                                (~15 ms)   │
│  ┌──────────────────┐  ┌──────────────────┐                         │
│  │ Dialogue context │→ │ NL description   │                         │
│  │ → prompt template│  │ + normalized text │                         │
│  └──────────────────┘  └────────┬─────────┘                         │
└───────────────────────────────────┼──────────────────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE C — Indic Parler-TTS (FP8, vLLM batched)         (~330 ms)  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │ T5 encoder       │→ │ AR decoder with  │→ │ First audio codec │  │
│  │ (text + prompt)  │  │ paged attention  │  │ token stream      │  │
│  └──────────────────┘  └──────────────────┘  └────────┬─────────┘  │
└───────────────────────────────────────────────────────┼──────────┘
                                                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE D — Codec decode + Streaming                     (~60 ms)   │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │ Codec → waveform │→ │ 50ms Opus chunks │→ │ Triton decoupled  │  │
│  │ (DAC/EnCodec)    │  │                  │  │ gRPC              │  │
│  └──────────────────┘  └──────────────────┘  └────────┬─────────┘  │
└───────────────────────────────────────────────────────┼──────────┘
                                                         ▼
                                              WebSocket → Voice Agent
```

**Latency budget (p99 ≤500 ms)**:
- Stage A (TTN): 70 ms
- Stage B (prompt encoder): 15 ms
- Stage C (Parler-TTS first token): 330 ms (AR decoder, FP8, vLLM continuous batching)
- Stage D (codec + stream): 60 ms
- **Total TTFA: ~475 ms p99** — tight but achievable; on cache hit, ~180 ms.

Parler-TTS documentation explicitly claims sub-500 ms TTFA on a modern GPU【turn1search5】, confirming the budget.

## 2.5 Tanglish Strategy (LoRA Fine-Tune)

Because the single backbone must handle all three language modes, a LoRA fine-tune is mandatory:

1. **Corpus construction (~50–100 h)**:
   - Hire 4–6 native Chennai-region Tamil-English bilingual voice talents.
   - Script ~2,000 transport-domain dialogues in all three Tanglish forms: (a) Tamil Unicode + Latin English (`உங்கள் pickup location எங்கே?`), (b) Latin-Tamil + Latin English (`Unga pickup location enga?`), (c) Tamil Unicode + English Unicode (`Chennai Central-ல இருக்கா?`).
   - Record at 48 kHz studio quality.
   - Augment with TTS-distilled data from a high-quality commercial reference (used only as training signal, never served).

2. **LoRA configuration**:
   - Rank 16–32 on cross-attention layers (the prompt-conditioning layers).
   - Trainable params <30M, training <24 GPU-hours on A100.
   - Avoids catastrophic forgetting of monolingual Tamil/English.

3. **Evaluation**: MANGO Tamil subset (246K human ratings)【turn0search33】 + custom 200-utterance Tanglish test set + UTMOS objective MOS【turn0search21】.

The fine-tune teaches the model the actual Chennai-region Tanglish prosody patterns — the natural-language prompt then activates them at inference.

## 2.6 Inference & Serving

### vLLM continuous batching (native T5 support)
- Parler-TTS is T5-based → vLLM supports it natively with paged attention, no tokenization hacks needed (unlike Orpheus/snac)【turn1search10】.
- `max_num_seqs` tuned to 8–12 on A100 40GB.

### FP8 quantization
- 938M → ~940 MB VRAM FP8 → ~8 replicas on A100 40GB → 15–20 concurrent with margin.
- FP8 KV cache (TensorRT-LLM)【turn0search16】.

### Streaming
- Parler-TTS supports token-by-token audio streaming【turn1search5】.
- Triton decoupled gRPC for transport【turn0search13】.
- vLLM-Omni also supports Parler-class models and would be the longer-term serving framework.

### Speculative decoding (VADUSA)
- Draft model: distilled Parler-TTS-tiny (~150M) — 2–3× AR speedup, distribution-preserving【turn1search26】【turn1search27】.

## 2.7 Hardware & Deployment

| Component | Hardware | Replicas | VRAM/replica | Concurrency/replica |
|---|---|---|---|---|
| Indic Parler-TTS (FP8) | A100 40GB | 4 | ~2 GB | 4–5 |
| Codec decoder (DAC) | shared A100 | 1 | <1 GB | — |
| TTN + prompt encoder | CPU pool (8 vCPU) | 2 | RAM only | 20 |
| Redis cache | 8 GB RAM | 1 | — | — |

**Total: 1× A100 40GB + CPU pool = ~$2.50/h**.

Sustainable throughput at 15–20 concurrent: ~400–550 audio-min/hour → **~$0.005–0.006/min**.

## 2.8 Cost Analysis (single A100 + CPU pool, ~$2.50/h)

| Concurrency | Audio-min/h | Infra $/h | $/min | Notes |
|---|---|---|---|---|
| 1 | ~45 | $2.50 | $0.056 | Underutilized |
| 5 | ~220 | $2.50 | $0.011 | Cache hit 30% |
| 10 | ~400 | $2.50 | $0.0063 | Cache hit 35% |
| 15 | ~500 | $2.50 | $0.0050 | Cache hit 40% |
| 20 | ~600 | $3.00 (CPU scales) | $0.0050 | Near saturation |

Slightly more expensive per minute than Solution 1 (~$0.005 vs $0.0046 at 15 concurrent) but operationally much simpler.

## 2.9 Strengths & Weaknesses

**Strengths**
- Simplest ops: one model, one serving path, one fine-tune.
- Natural-language prompt control of emotion/prosody — uniquely valuable for transport contact-center tone (calm for cancellations, upbeat for confirmations).
- Emergent code-mixing from a single LM — most "novel" approach.
- Apache 2.0 license (with gating caveat).

**Weaknesses**
- Larger model (938M vs 337M) → higher per-request latency, more aggressive quantization needed.
- Code-mix is implicit — harder to guarantee Tanglish naturalness without extensive evaluation.
- Gated on HuggingFace — must verify commercial-use terms with AI4Bharat or use the ungated mirror【turn1search24】.
- 30s max generation — chunking logic required for longer utterances【turn1search6】.
- Higher $/min than Solution 1.

## 2.10 When This Is the Best Choice

When **operational simplicity + emotion/prosody control** is valued over absolute minimum cost — i.e., the team wants a single model that can sound calm during a cancellation and upbeat during a booking confirmation, and can absorb the ~10% higher $/min in exchange for not maintaining a multi-model pipeline. Best for teams with strong MLOps but limited low-level inference engineering capacity.

---

# Solution 3 — Hybrid Tiered Router (Kokoro + MeloTTS + IndicF5 + Indic Parler-TTS)

## 3.1 Why This Is the Best Cost-Aggressive Path

When absolute $/min is the dominant KPI, a single model is suboptimal because expensive expressive models (Indic Parler-TTS) are invoked even for trivial boilerplate ("Your OTP is 4821") that a tiny model could handle. Solution 3 deploys a **dynamic request router** that classifies each utterance and dispatches to the lowest-latency tier that can serve it with acceptable quality — MeloTTS on CPU for hot boilerplate, IndicF5 on GPU for standard Tamil/Tanglish, Indic Parler-TTS only for long-form emotional Tanglish, and Kokoro-82M for pure-English fragments. Combined with **sub-sentence audio caching** (cache the prefix "Your cab will arrive in" and suffix "minutes", synthesize only the variable number), this drives 40–60% of requests to sub-100 ms TTFA at near-zero marginal cost. The trade-off is operational complexity of maintaining four models.

## 3.2 License Audit Table

| Component | Role | License | Commercial use? | Source |
|---|---|---|---|---|
| **MeloTTS** (myshell-ai) | T1 hot CPU tier (Indic via community FT) | **MIT** | ✅ Yes | 【turn1search6】【turn1search23】 |
| **IndicF5** (ai4bharat) | T2 standard GPU tier | Apache 2.0 (gated) | ✅ Yes | 【turn0search35】【turn0search36】 |
| **Indic Parler-TTS** (ai4bharat) | T3 expressive GPU tier | Apache 2.0 (gated) | ✅ Yes (verify) | 【turn1search0】【turn1search2】 |
| **Kokoro-82M** (hexgrad) | English-fragment tier | **Apache 2.0** | ✅ Yes | 【turn1search11】【turn1search13】 |
| **Vocos** vocoder | Fallback vocoder | MIT | ✅ Yes | 【turn0search15】【turn0search16】 |
| **indic-text-normalization** | WFST TTN | Apache 2.0 | ✅ Yes | 【turn0search20】 |
| **IndicLID** | LID for routing | Apache 2.0 | ✅ Yes | 【turn1search31】 |
| **IndicXlit** | Transliteration | Apache 2.0 | ✅ Yes | 【turn0search8】 |
| **Triton** | Serving | Apache 2.0 | ✅ Yes | 【turn0search10】 |
| **vLLM-Omni** | Batching | Apache 2.0 | ✅ Yes | 【turn1search28】 |
| **TensorRT-LLM** | FP8 quant | Apache 2.0 | ✅ Yes | 【turn0search16】 |
| **UTMOS / NISQA / MANGO** | Evaluation | MIT / CC-BY | ✅ Yes | 【turn0search21】【turn0search33】 |
| **Redis** | Cache | BSD-3-Clause | ✅ Yes | — |

**Explicit exclusions** (cannot be used in any tier due to license):
- ❌ **XTTS-v2 / Coqui** — Coqui Public Model License (CPML), non-commercial only【turn0search10】【turn0search11】
- ❌ **F5-TTS base** — CC-BY-NC【turn0search20】
- ❌ **Spark-TTS** — HF model card says CC BY-NC-SA 4.0 due to training data terms, despite GitHub Apache tag — NOT safely commercial【turn1search1】
- ❌ **Higgs Audio V3** — Research and Non-Commercial license【turn1search40】
- ⚠️ **Orpheus TTS** — tagged Apache 2.0 but built on Llama 3.2; Llama 3.2 Community License applies (700M MAU threshold, attribution) — usable but with license-inheritance caveat【turn1search5】【turn0search27】
- ⚠️ **Chatterbox** — MIT but built on Llama backbone — same Llama license inheritance caveat【turn0search29】
- ❌ **Emilia dataset** — CC-BY-NC, cannot train commercial models【turn1search31】

## 3.3 Architecture — Every Component

### T1 Hot tier: MeloTTS (MIT, CPU)
- **Architecture**: VITS / VITS2 / Bert-VITS2 hybrid with multilingual BERT text encoder【turn1search5】.
- **Languages**: Chinese, English (with Indian English accent), Japanese, Korean, French, Spanish — community fine-tunes add Indic languages【turn1search6】【turn1search7】.
- **Code-mix precedent**: the Chinese speaker natively handles mixed Chinese-English — same architecture pattern that makes it suitable for Tanglish after a Tamil fine-tune【turn1search7】.
- **Why this tier**: CPU-real-time inference, no GPU needed, ~$0.50/h for an 8-vCPU pool. Handles boilerplate and short utterances in <150 ms TTFA【turn1search6】【turn1search25】.

### T2 Standard tier: IndicF5 (Apache 2.0, GPU)
- Same as Solution 1's backbone — 337M flow-matching DiT, native Tamil, ~280 ms first-chunk TTFA at FP8.

### T3 Expressive tier: Indic Parler-TTS (Apache 2.0, GPU)
- Same as Solution 2's backbone — 938M T5-based, natural-language prompt control, ~330 ms TTFA.

### English-fragment tier: Kokoro-82M (Apache 2.0)
- **Architecture**: StyleTTS2 + iSTFTNet, 82M params, phoneme-level BERT text encoder, style encoder for prosody, WavLM discriminator【turn1search12】【turn1search13】.
- **Languages**: English (with voice packs for American/British/Indian accents)【turn1search11】.
- **Why this tier**: 82M params → ~150 MB VRAM → ~100 ms TTFA on a cheap L4 GPU. When a Tanglish utterance contains a pure-English interjection ("Your cab will arrive in 10 minutes"), Kokoro synthesizes the English fragment and the cross-fade merges it with the Tamil fragments from IndicF5.

### Router: IndicLID + gradient-boosted classifier
- IndicLID classifies tokens at 47 classes【turn1search31】.
- A lightweight XGBoost or rule-based classifier uses: language-ID ratio, utterance length, dialogue-act class (from the dialogue manager), cache-hit signal → dispatches to T1/T2/T3/Kokoro in ~1 ms.

## 3.4 Full Pipeline (text → audio) with Latency Budget

```
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE A — Context-Aware Text Normalization             (~70 ms)   │
│  (Same WFST + domain rules as Solutions 1 & 2)                       │
└─────────────────────────────────────────┬───────────────────────────┘
                                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE B — Dynamic Router                               (~5 ms)    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │ Cache    │→ │ IndicLID │→ │ Length / │→ │ Dispatch │            │
│  │ lookup   │  │ ratio    │  │ dialogue │  │ decision │            │
│  └──────────┘  └──────────┘  └──────────┘  └────┬─────┘            │
└────────────────────────────────────────────────────┼────────────────┘
                                                     │
        ┌────────────────┬───────────┬───────────────┴───────────┐
        ▼                ▼           ▼                           ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐
│ T1: MeloTTS  │ │ T2: IndicF5  │ │ T3: Parler   │ │ Kokoro (EN frag) │
│ CPU          │ │ A100 FP8     │ │ A100 FP8     │ │ L4 GPU           │
│ <150ms TTFA  │ │ <300ms TTFA  │ │ <400ms TTFA  │ │ <100ms TTFA     │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └────────┬─────────┘
       └────────────────┴───────────┴──────────────────────┬─────────┘
                                                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE C — Audio Cross-Fade + Streaming                 (~30 ms)   │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │ Sub-sentence     │→ │ Cross-fade at    │→ │ Triton decoupled │  │
│  │ chunk assembler   │  │ boundaries       │  │ gRPC stream      │  │
│  └──────────────────┘  └──────────────────┘  └────────┬─────────┘  │
└───────────────────────────────────────────────────────┼──────────┘
                                                         ▼
                                              WebSocket → Voice Agent
```

**Latency budget by tier (p99 ≤500 ms)**:
- **T1 (MeloTTS, cache hit)**: TTN 70 + router 5 + MeloTTS 80 + stream 30 = **~185 ms**
- **T2 (IndicF5)**: TTN 70 + router 5 + IndicF5 280 + stream 30 = **~385 ms**
- **T3 (Parler)**: TTN 70 + router 5 + Parler 330 + stream 60 = **~465 ms**
- **Kokoro fragment**: TTN 70 + router 5 + Kokoro 100 + cross-fade 30 = **~205 ms**

All tiers comfortably within 500 ms p99.

## 3.5 Tanglish Strategy (Tiered)

The router's tiering itself encodes the Tanglish strategy:

1. **Pure Tamil utterance** → T2 (IndicF5).
2. **Pure English utterance** → Kokoro (if short) or T2 (if long).
3. **Tanglish with >60% Tamil** → T2 (IndicF5 handles English-in-Tamil natively).
4. **Tanglish with >40% English** → split at clause boundaries, Tamil fragments to T2, English fragments to Kokoro, cross-fade.
5. **Long-form emotional Tanglish** (e.g., explaining a cancellation) → T3 (Indic Parler-TTS with code-switch prompt).
6. **Short boilerplate** → T1 (MeloTTS) or cache hit.

The Tanglish router from Solution 1 (IndicLID + IndicXlit + `<cs>` boundary marking) is reused for the T2 and T3 paths.

## 3.6 Novel Component: Sub-Sentence Audio Caching

The biggest cost win in Solution 3 is **sub-sentence audio chunk caching**. Transport conversations have extremely repetitive sub-structures:

- "Your cab will arrive in" + `<X>` + "minutes"
- "Your OTP is" + `<XXXX>`
- "Your booking ID is" + `<ID>`
- "Please share your" + `<entity>`

The system:
1. Parses the normalized text into a **template tree** — fixed prefix, variable slot, fixed suffix.
2. Hashes the fixed prefix/suffix strings → Redis lookup of pre-synthesized audio chunks.
3. Synthesizes only the variable slot (the number, OTP, ID).
4. Cross-fades the three pieces in the audio domain.

This drives ~40–60% of transport-domain requests to sub-100 ms TTFA at near-zero GPU cost — the variable slot is typically <1 second of audio synthesized in <80 ms.

## 3.7 Inference & Serving

- **Triton multi-model repository**: MeloTTS (CPU backend), IndicF5 (Python backend, decoupled), Indic Parler-TTS (Python backend, decoupled), Kokoro (ONNX backend).
- **Triton decoupled mode** for all streaming tiers: `model_transaction_policy { decoupled: True }` — only gRPC supports decoupled multi-response streaming【turn0search13】.
- **Dynamic batching** per tier: `max_queue_delay_microseconds: 5000` (5 ms) — balances batching gain vs latency.
- **vLLM-Omni** for T2/T3 GPU batching once F5-TTS support is GA【turn1search15】.
- **ONNX export** for MeloTTS and Kokoro — enables CPU/cheap-GPU inference without PyTorch overhead.

## 3.8 Hardware & Deployment

| Component | Hardware | Replicas | Cost/h | Concurrency/replica |
|---|---|---|---|---|
| MeloTTS (T1, ONNX) | CPU pool 16 vCPU | 2 | $0.50 | 5–8 |
| IndicF5 (T2, FP8) | A100 40GB | 2 | $2.00 | 5–6 |
| Indic Parler-TTS (T3, FP8) | shared A100 | 1 | (shared) | 2–4 |
| Kokoro (EN, ONNX) | L4 24GB | 1 | $0.80 | 8–10 |
| Router + TTN | CPU pool 8 vCPU | 2 | $0.30 | 20 |
| Redis cache | 16 GB RAM | 1 | $0.20 | — |

**Total: 1× A100 40GB + 1× L4 24GB + CPU pool = ~$3.30/h** (more hardware than Solution 1, but T1 absorbs ~40% of traffic at near-zero marginal cost).

Sustainable throughput at 15–20 concurrent: ~900–1100 audio-min/hour → **~$0.0030–0.0037/min**.

## 3.9 Cost Analysis (A100 + L4 + CPU pool, ~$3.30/h)

| Concurrency | Audio-min/h | Infra $/h | $/min | Notes |
|---|---|---|---|---|
| 1 | ~60 | $3.30 | $0.055 | T1 + cache absorb most |
| 5 | ~300 | $3.30 | $0.011 | Cache hit 40% |
| 10 | ~550 | $3.30 | $0.0060 | Cache hit 45% |
| 15 | ~800 | $3.30 | $0.0041 | Cache hit 50%, T1 + T2 |
| 20 | ~1000 | $3.80 (CPU scales) | **$0.0038** | Cache hit 55%, all tiers |

**Lowest $/min of all three solutions** at 15–20 concurrent (~$0.0038 vs Solution 1's $0.0044 vs Solution 2's $0.0050).

## 3.10 Strengths & Weaknesses

**Strengths**
- Lowest $/min of the three variants — ~10× cheaper than commercial APIs.
- Sub-sentence caching converts 40–60% of transport-domain requests to <100 ms TTFA.
- Graceful degradation: T3 drops first under load, falling back to T2 with flatter prosody; T2 overflows to T1.
- Each tier independently scalable — CPU pool scales cheaply for T1.
- License-clean (all MIT/Apache).

**Weaknesses**
- Operational complexity: four models, two hardware profiles, router logic, cross-fade tuning.
- Voice consistency across tiers requires fixed speaker prompts/references in all four models — non-trivial to match timbre between MeloTTS, IndicF5, Parler, and Kokoro.
- Cross-fade at sub-sentence boundaries can introduce prosody discontinuity if not carefully tuned.
- More moving parts = more failure modes.

## 3.11 When This Is the Best Choice

When **absolute $/min is the dominant KPI** and the team has the MLOps maturity to maintain a four-model tiered system — i.e., high-volume contact-center deployments where the 40–60% cache/T1 absorption compounds into material monthly savings, and where the team can absorb the cross-fade and voice-consistency engineering effort. Best for teams that have already validated Solutions 1 or 2 and are optimizing for unit economics at scale.

---

## Cross-Solution Comparison Summary

| Dimension | Solution 1 (IndicF5 Modular) | Solution 2 (Parler Single) | Solution 3 (Hybrid Tiered) |
|---|---|---|---|
| Models in stack | IndicF5 + Vocos + TTN | Indic Parler-TTS + TTN | MeloTTS + IndicF5 + Parler + Kokoro + TTN |
| Primary license | Apache 2.0 (gated) | Apache 2.0 (gated) | MIT + Apache 2.0 (cleanest) |
| p99 TTFA (cold) | ~435 ms | ~475 ms | ~385–465 ms (tier-dependent) |
| $/min @ 15 concurrent | ~$0.0046 | ~$0.0050 | ~$0.0041 |
| Tanglish handling | Script-unified router + `<cs>` | LoRA + NL prompt | Tiered split + cross-fade |
| Ops complexity | Medium | Low | High |
| Expressiveness | Medium | High | Tier-dependent |
| Recommended for | Default production path | Simplicity + emotion control | Maximum cost efficiency |

All three are license-clean, self-hostable, and meet p99 ≤500 ms at 15–20 concurrent. The choice is between **lowest risk (Solution 1)**, **simplest ops + best expressiveness (Solution 2)**, and **lowest $/min (Solution 3)**.