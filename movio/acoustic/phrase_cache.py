"""Offline phrase pre-synthesis — builds the disk cache for HybridEngine.

Run this ONCE after training (on any machine — GPU preferred, CPU OK since
it's offline):

    python -m movio.acoustic.phrase_cache --config config/settings.yaml

What it does:
  1. Expands all template phrases with representative values
     (OTPs, booking IDs, wait times, distances, driver names, etc.)
  2. Runs each through Stage A text normalisation (same pipeline as runtime)
  3. Synthesizes with IndicF5 (if model_path is set) or MMS-TTS
  4. Saves PCM16 files to models/phrase_cache/ keyed by sha256(voice|text)

After this, HybridEngine serves those phrases in ~5ms from disk.
Dynamic phrases not in the cache fall back to MMS-TTS (~100ms).
"""

import argparse
import hashlib
import itertools
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Phrase templates ─────────────────────────────────────────────────────────
# Each template is either a plain string or a list of variants.
# Templates use {slot} placeholders expanded by _expand_templates().
#
# Cover ~90% of a Tamil taxi/transport voice agent's utterances.

PHRASE_TEMPLATES = [
    # ── Pickup / arrival ─────────────────────────────────────────────────────
    "உங்கள் கேப் {mins} நிமிடத்தில் வரும்.",
    "உங்கள் ஓட்டுநர் {mins} நிமிடத்தில் வருவார்.",
    "உங்கள் வாகனம் வந்துவிட்டது.",
    "உங்கள் வாகனம் கிளம்பிவிட்டது.",
    "நீங்கள் {place}-ல் இறங்கினீர்கள்.",
    "உங்கள் பயணம் தொடங்கிவிட்டது.",
    "உங்கள் பயணம் முடிந்தது.",
    "கேப் {mins} நிமிடம் தாமதமாகும்.",
    # ── OTP / verification ───────────────────────────────────────────────────
    "உங்கள் OTP {otp}.",
    "OTP: {otp}.",
    "உங்கள் verification code {otp}.",
    "Your OTP is {otp}.",
    # ── Fare / payment ───────────────────────────────────────────────────────
    "உங்கள் பயண கட்டணம் ₹{fare}.",
    "மொத்த கட்டணம் ₹{fare}.",
    "₹{fare} செலுத்தவும்.",
    "UPI மூலம் கட்டணம் செலுத்தவும்.",
    "Cash payment ₹{fare}.",
    # ── Booking confirmation ─────────────────────────────────────────────────
    "உங்கள் booking confirm ஆகிவிட்டது.",
    "Booking ID {booking_id}.",
    "உங்கள் பயணம் cancel ஆகிவிட்டது.",
    "Driver cancelled. புதிய கேப் தேடுகிறோம்.",
    # ── Driver info ──────────────────────────────────────────────────────────
    "உங்கள் ஓட்டுநர் {driver} வருகிறார்.",
    "வாகன எண் {plate}.",
    "ஓட்டுநர் {driver}, rating {rating}.",
    # ── Distance / ETA ───────────────────────────────────────────────────────
    "இன்னும் {km} கிலோமீட்டர் உள்ளது.",
    "Destination {km} km away.",
    "நீங்கள் {place} அருகில் இருக்கிறீர்கள்.",
    # ── Safety / instructions ────────────────────────────────────────────────
    "seat belt கட்டிக்கொள்ளவும்.",
    "Door பூட்டிக்கொள்ளவும்.",
    "Emergency? 112 அழைக்கவும்.",
    # ── English variants (Tanglish) ──────────────────────────────────────────
    "Your cab arrives in {mins} minutes.",
    "Your driver is {mins} minutes away.",
    "Your ride has started.",
    "Your ride is complete.",
    "Your booking is confirmed.",
    "Driver is waiting at the pickup point.",
    "Please rate your ride.",
    "Have a safe journey.",
]

# Slot fill values — small representative set covers common cases
SLOT_VALUES = {
    "mins":       ["2", "3", "5", "7", "10", "15"],
    "otp":        ["1234", "5678", "9012", "3456", "7890"],
    "fare":       ["50", "80", "100", "120", "150", "200", "250", "300"],
    "booking_id": ["TXN123456", "BK789012", "RD345678"],
    "driver":     ["Kumar", "Ravi", "Murugan", "Selvam", "Anand"],
    "plate":      ["TN 01 AB 1234", "TN 09 CD 5678", "TN 22 EF 9012"],
    "rating":     ["4.5", "4.8", "4.2"],
    "km":         ["1", "2", "3", "5", "8", "10"],
    "place":      [
        "Chennai Central", "Tambaram", "T Nagar", "Anna Nagar",
        "Velachery", "OMR", "Airport", "Egmore",
    ],
}


def _expand_templates(templates: list[str], slot_values: dict) -> list[str]:
    """Expand each template with up to one value per slot (cross-product capped)."""
    expanded = set()
    for tmpl in templates:
        import re
        slots = re.findall(r"\{(\w+)\}", tmpl)
        if not slots:
            expanded.add(tmpl)
            continue
        # Build value lists for each slot found in this template
        value_lists = [slot_values.get(s, [s]) for s in slots]
        # Cap cross-product at 12 variants per template to keep cache size small
        for combo in itertools.islice(itertools.product(*value_lists), 12):
            text = tmpl
            for slot, val in zip(slots, combo):
                text = text.replace(f"{{{slot}}}", val, 1)
            expanded.add(text)
    return sorted(expanded)


def _synthesize_indicf5(
    texts: list[str],
    model_path: str,
    f5tts_dir: str,
    voice_name: str,
    cache_dir: Path,
    config: dict,
) -> int:
    """Synthesize using IndicF5 (high quality, GPU preferred). Returns count saved."""
    from movio.acoustic.indicf5_engine import IndicF5Engine
    from movio.utils.audio import float_to_pcm16

    engine = IndicF5Engine(config)
    engine.load()
    saved = 0
    for i, text in enumerate(texts):
        key = hashlib.sha256(f"{voice_name}|{text}".encode()).hexdigest()[:16]
        p = cache_dir / f"{key}.pcm"
        if p.exists():
            continue
        try:
            audio = engine.synthesize(text, voice_name=voice_name)
            p.write_bytes(float_to_pcm16(audio))
            saved += 1
            if (i + 1) % 50 == 0:
                logger.info("  [IndicF5] %d/%d synthesized", i + 1, len(texts))
        except Exception as exc:
            logger.warning("  skip '%s': %s", text[:50], exc)
    return saved


def _synthesize_mms(
    texts: list[str],
    voice_name: str,
    cache_dir: Path,
    config: dict,
) -> int:
    """Synthesize using MMS-TTS (fast CPU fallback). Returns count saved."""
    from movio.acoustic.mms_engine import MMSEngine
    from movio.utils.audio import float_to_pcm16

    engine = MMSEngine(config)
    engine.load()
    saved = 0
    for i, text in enumerate(texts):
        key = hashlib.sha256(f"{voice_name}|{text}".encode()).hexdigest()[:16]
        p = cache_dir / f"{key}.pcm"
        if p.exists():
            continue
        try:
            audio = engine.synthesize(text)
            p.write_bytes(float_to_pcm16(audio))
            saved += 1
            if (i + 1) % 50 == 0:
                logger.info("  [MMS] %d/%d synthesized", i + 1, len(texts))
        except Exception as exc:
            logger.warning("  skip '%s': %s", text[:50], exc)
    return saved


def main():
    ap = argparse.ArgumentParser(
        description="Pre-synthesize common taxi phrases into the disk cache."
    )
    ap.add_argument("--config", default="config/settings.yaml")
    ap.add_argument(
        "--engine", default="auto", choices=["auto", "indicf5", "mms"],
        help=(
            "auto: use IndicF5 if model_path is not 'base' and bundle exists, "
            "else fall back to MMS-TTS. "
            "indicf5: force IndicF5 (requires exported bundle or base weights). "
            "mms: force MMS-TTS (fast CPU, lower quality)."
        ),
    )
    ap.add_argument("--voice", default="default", help="Voice name to pre-cache")
    ap.add_argument(
        "--extra-phrases", default=None,
        help="Path to a plain text file with one phrase per line to add to the cache.",
    )
    ap.add_argument("--dry-run", action="store_true",
                    help="Print expanded phrases without synthesizing.")
    args = ap.parse_args()

    import yaml
    config = yaml.safe_load(open(args.config, encoding="utf-8"))

    cache_dir = Path(config.get("stage_c", {}).get("hybrid", {}).get(
        "cache_dir", "models/phrase_cache"
    ))
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Run text normalisation on all phrases (same pipeline as runtime)
    from movio.textnorm.normalizer import TextNormalizer
    normalizer = TextNormalizer(config)

    templates = list(PHRASE_TEMPLATES)
    if args.extra_phrases and Path(args.extra_phrases).exists():
        extra = [ln.strip() for ln in open(args.extra_phrases, encoding="utf-8")
                 if ln.strip() and not ln.startswith("#")]
        templates.extend(extra)
        logger.info("Loaded %d extra phrases from %s", len(extra), args.extra_phrases)

    raw_phrases = _expand_templates(templates, SLOT_VALUES)
    logger.info("Expanded to %d phrases", len(raw_phrases))

    # Normalise text (Stage A) — same as runtime so cache keys match
    normalized = []
    for raw in raw_phrases:
        result = normalizer.normalize(raw)
        normalized.append(result.text)
    # Deduplicate (some templates may normalise to the same string)
    normalized = sorted(set(normalized))
    logger.info("After normalisation: %d unique phrases", len(normalized))

    if args.dry_run:
        for p in normalized:
            print(p)
        return

    # Decide engine
    engine_choice = args.engine
    if engine_choice == "auto":
        model_path = config.get("stage_c", {}).get("indicf5", {}).get("model_path", "base")
        if model_path != "base" and Path(model_path).is_dir():
            engine_choice = "indicf5"
            logger.info("Auto: using IndicF5 (bundle found at %s)", model_path)
        else:
            engine_choice = "mms"
            logger.info("Auto: using MMS-TTS (no IndicF5 bundle; set model_path in settings.yaml)")

    t0 = time.perf_counter()
    already = len(list(cache_dir.glob("*.pcm")))
    logger.info("Cache dir: %s (%d phrases already cached)", cache_dir, already)

    if engine_choice == "indicf5":
        f5tts_dir = config.get("stage_c", {}).get("indicf5", {}).get("f5tts_dir", "")
        model_path = config.get("stage_c", {}).get("indicf5", {}).get("model_path", "base")
        saved = _synthesize_indicf5(
            normalized, model_path, f5tts_dir, args.voice, cache_dir, config,
        )
    else:
        saved = _synthesize_mms(normalized, args.voice, cache_dir, config)

    elapsed = time.perf_counter() - t0
    total = len(list(cache_dir.glob("*.pcm")))
    logger.info(
        "Done: %d new phrases synthesized in %.1fs | total cached: %d",
        saved, elapsed, total,
    )
    logger.info("Cache size: %.1f MB",
                sum(p.stat().st_size for p in cache_dir.glob("*.pcm")) / 1e6)


if __name__ == "__main__":
    main()
