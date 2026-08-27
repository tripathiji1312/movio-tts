# Human Evaluation Protocol — Regional Naturalness & Code-Switching

Complements the objective metrics (`eval/run_quality_eval.py`). Run with
**5+ native Tamil speakers from the Chennai region**.

## Setup

1. Generate audio for `eval/testsets/tanglish_transport_200.tsv` (or a
   60-utterance subset stratified across categories).
2. Randomize order, blind the system identity (base vs fine-tuned vs
   commercial reference if used for calibration only).
3. Use headphones; quiet room; one session ≤ 30 minutes.

## Rating form (per utterance)

| Dimension | Scale | Question |
|---|---|---|
| Naturalness (MOS) | 1–5 | "Does this sound like a natural speaker?" |
| Pronunciation accuracy | 1–5 | Tamil words pronounced correctly? |
| English-in-Tamil quality | 1–5 | Do embedded English words sound natural? |
| Code-switch smoothness | 1–5 | Are language transitions natural, not stitched? |
| Intelligibility | 1–5 | Could you transcribe it without effort? |
| Structured-entity accuracy | binary | Were OTP / ID / price / time spoken correctly? |

## Aggregation

- Report mean ± 95% CI per dimension, overall and per category
  (`ta`, `en`, `ta_en_mix`, `ta_roman`).
- MOS ≥ 4.0 and CS-smoothness ≥ 3.8 = production-ready per blueprint targets.
- Flag every structured-entity binary failure — these are hard failures that
  block promotion regardless of MOS.

## Promotion gate (combined)

| Metric | Gate |
|---|---|
| Human MOS | ≥ 4.0 |
| Code-switch smoothness | ≥ 3.8 |
| Structured-entity accuracy | 100% on flagged subset |
| UTMOS | ≥ base model − 0.05 |
| WER | ≤ base + 2% absolute |
| p99 TTFA @ 15–20 conc | ≤ 500 ms |
