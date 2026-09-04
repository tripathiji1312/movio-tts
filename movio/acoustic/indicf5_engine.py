"""IndicF5Engine — F5-TTS CFM-DiT acoustic backbone (337M params, 24 kHz).

Uses IndicF5's own inference stack (third_party/IndicF5/) — NOT the pip-installed
F5-TTS, which has a different architecture and produces garbage with IndicF5 weights.

model_path config options:
  "base"             — downloads ai4bharat/IndicF5 weights from HF hub (MIT)
  "/path/to/bundle"  — fine-tuned bundle dir (model.pt + vocab.txt + config.json)
"""

import logging
import sys
from pathlib import Path
from typing import Iterator

from pydub import AudioSegment

import numpy as np

from movio.acoustic.chunking import chunk_text
from movio.utils.audio import ms_to_samples

logger = logging.getLogger(__name__)

_INDICF5_SRC = str(Path(__file__).resolve().parent.parent.parent / "third_party" / "IndicF5")


class VoiceProfile:
    def __init__(self, name: str, ref_audio_path: str, ref_text: str):
        self.name = name
        self.ref_audio_path = ref_audio_path
        self.ref_text = ref_text


def load_voice_profiles(voices_dir: str | Path) -> dict[str, VoiceProfile]:
    voices_dir = Path(voices_dir)
    profiles: dict[str, VoiceProfile] = {}
    if not voices_dir.exists():
        return profiles

    # Discover a reliable default reference audio file
    default_ref_path = ""
    candidate = voices_dir / "ta_female_neutral" / "ref.wav"
    if candidate.exists():
        default_ref_path = str(candidate.resolve())
    else:
        all_refs = list(voices_dir.glob("*/ref.wav"))
        if all_refs:
            default_ref_path = str(all_refs[0].resolve())

    default_ref_text = "ஆனா நீங்க இப்போதான் மொத தடவையா இன்டர்நெட் யூஸ் பண்றீங்க அப்படின்னா இதை முழுசா கத்துக்க கொஞ்சம் நாள் ஆகும்."
    female_meta = voices_dir / "ta_female_neutral" / "voice.yaml"
    if female_meta.exists():
        import yaml
        txt = yaml.safe_load(female_meta.read_text(encoding="utf-8")).get("ref_text", "")
        if txt and txt.strip():
            default_ref_text = txt.strip()

    for meta_path in sorted(voices_dir.glob("*/voice.yaml")):
        import yaml
        meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
        audio_rel = meta.get("ref_audio")
        resolved = (meta_path.parent / audio_rel).resolve() if audio_rel else None
        audio_path = str(resolved) if (resolved and resolved.exists()) else default_ref_path
        ref_text = (meta.get("ref_text") or "").strip() or default_ref_text
        profile = VoiceProfile(
            name=meta.get("name", meta_path.parent.name),
            ref_audio_path=audio_path,
            ref_text=ref_text,
        )
        profiles[profile.name] = profile
    return profiles


class IndicF5Engine:
    """F5-TTS CFM-DiT acoustic backbone.

    Lazy-loads on first synthesize call. Caches preprocessed reference audio
    per voice profile to amortise the preprocessing cost across requests.

    num_flow_steps trades quality vs latency:
      8  steps → ~150-250ms/chunk on T4, slightly lower quality
      12 steps → ~250-400ms/chunk on T4, good quality  (default)
      20 steps → ~500-700ms/chunk on T4, best quality
    """

    _MODEL_CFG = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    SAMPLE_RATE = 24000

    def __init__(self, config: dict):
        cfg = config.get("stage_c", {}).get("indicf5", {})
        self.model_path = cfg.get("model_path", "base")
        self.device = cfg.get("device", "cuda")
        self.num_flow_steps = int(cfg.get("num_flow_steps", 12))
        self.sway_coef = float(cfg.get("sway_sampling_coef", -1.0))
        self.cfg_strength = float(cfg.get("cfg_strength", 2.0))
        self.ode_method = cfg.get("ode_method", "euler")
        self.speed = float(cfg.get("speed", 1.0))
        self.default_voice = cfg.get("default_voice", "")
        self.voices = load_voice_profiles(cfg.get("voices_dir", "config/voices"))
        self.sample_rate = self.SAMPLE_RATE
        self._model = None
        self._vocoder = None
        self._vocab_char_map = None
        self._device: str = self.device
        self._ref_cache: dict[str, tuple] = {}

    # ── path / import helpers ────────────────────────────────────────────────

    _path_set = False

    @classmethod
    def _ensure_indicf5_path(cls) -> None:
        if cls._path_set:
            return
        if _INDICF5_SRC not in sys.path:
            sys.path.insert(0, _INDICF5_SRC)
        for k in list(sys.modules):
            if k.startswith("f5_tts"):
                del sys.modules[k]
        cls._path_set = True

    def _vocab_path(self) -> str:
        if self.model_path != "base":
            p = Path(self.model_path) / "vocab.txt"
            if p.exists():
                return str(p)
        indicf5_vocab = Path("models/indicf5_tanglish") / "vocab.txt"
        if indicf5_vocab.exists():
            return str(indicf5_vocab)
        vocab_in_third_party = Path(_INDICF5_SRC) / "checkpoints" / "vocab.txt"
        if vocab_in_third_party.exists():
            return str(vocab_in_third_party)
        from huggingface_hub import hf_hub_download
        return hf_hub_download("ai4bharat/IndicF5", "checkpoints/vocab.txt")

    # ── model loading ────────────────────────────────────────────────────────

    def _load(self) -> None:
        import torch
        self._ensure_indicf5_path()

        from f5_tts.model.backbones.dit import DiT
        from f5_tts.infer.utils_infer import load_model, load_vocoder

        device = (
            self.device
            if self.device != "cuda" or torch.cuda.is_available()
            else "cpu"
        )
        self._device = device

        vocab_path = self._vocab_path()

        if self.model_path == "base":
            from huggingface_hub import hf_hub_download
            ckpt_path = hf_hub_download("ai4bharat/IndicF5", "model.safetensors")
            use_ema = True   # base safetensors has ema_model._orig_mod. keys
        else:
            bundle = Path(self.model_path)
            if not bundle.is_dir():
                raise RuntimeError(
                    f"IndicF5 bundle not found: {bundle}. "
                    "Run training/scripts/05_merge_export.py first."
                )
            ckpt_path = str(bundle / "model.pt")
            use_ema = False  # fine-tuned .pt has model_state_dict directly

        model = load_model(
            DiT,
            self._MODEL_CFG,
            ckpt_path=ckpt_path,
            mel_spec_type="vocos",
            vocab_file=vocab_path,
            ode_method=self.ode_method,
            use_ema=use_ema,
            device=device,
        )
        model = model.eval()
        vocoder = load_vocoder(vocoder_name="vocos", is_local=False, device=device)

        self._model = model
        self._vocoder = vocoder
        logger.info(
            "IndicF5Engine ready: model=%s device=%s flow_steps=%d cfg=%.1f",
            self.model_path, device, self.num_flow_steps, self.cfg_strength,
        )

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if not self.is_ready:
            self._load()

    # ── voice management ─────────────────────────────────────────────────────

    def get_voice(self, voice_name: str | None) -> VoiceProfile:
        name = voice_name or self.default_voice
        if name in self.voices:
            return self.voices[name]
        if self.voices:
            return next(iter(self.voices.values()))
        raise RuntimeError(
            "No voice profiles configured. "
            "Add config/voices/<name>/voice.yaml with ref_audio and ref_text."
        )

    def _get_ref_audio(self, voice: VoiceProfile) -> tuple:
        """Preprocess + cache reference audio (amortised across all requests)."""
        ref_path = voice.ref_audio_path
        if not Path(ref_path).exists():
            default_v = self.voices.get(self.default_voice)
            if default_v and Path(default_v.ref_audio_path).exists():
                ref_path = default_v.ref_audio_path
        ref_text = (voice.ref_text or "").strip() or "ஆனா நீங்க இப்போதான் மொத தடவியா இன்டர்நெட் யூஸ் பண்றீங்க அ Atenas..."
        key = (ref_path, ref_text)
        if key not in self._ref_cache:
            self._ensure_indicf5_path()
            from f5_tts.infer.utils_infer import preprocess_ref_audio_text
            if not ref_path or not Path(ref_path).exists():
                ref_audio = AudioSegment.silent(duration=500)
                proc_ref_text = ref_text
            else:
                ref_audio, proc_ref_text = preprocess_ref_audio_text(
                    ref_path, ref_text
                )
            self._ref_cache[key] = (ref_audio, proc_ref_text)
        return self._ref_cache[key]

    # ── synthesis ────────────────────────────────────────────────────────────

    def synthesize_chunk(
        self,
        text: str,
        voice: VoiceProfile,
        flow_steps: int | None = None,
        speed: float | None = None,
    ) -> np.ndarray:
        self.load()
        self._ensure_indicf5_path()
        from f5_tts.infer.utils_infer import infer_process

        ref_audio, ref_text = self._get_ref_audio(voice)
        steps = flow_steps or self.num_flow_steps
        eff_speed = speed if speed is not None else self.speed

        # infer_process returns (audio_np, sample_rate, spectrogram)
        audio, _, _ = infer_process(
            ref_audio,
            ref_text,
            text,
            self._model,
            self._vocoder,
            device=self._device,
            nfe_step=steps,
            sway_sampling_coef=self.sway_coef,
            cfg_strength=self.cfg_strength,
            speed=eff_speed,
        )
        return np.asarray(audio, dtype=np.float32)

    def synthesize(
        self,
        text: str,
        voice_name: str | None = None,
        speed: float | None = None,
    ) -> np.ndarray:
        """Synthesize full text in one pass. Use synthesize_stream for low TTFA."""
        voice = self.get_voice(voice_name)
        return self.synthesize_chunk(text, voice, speed=speed)

    def synthesize_stream(
        self,
        text: str,
        voice_name: str | None = None,
        min_syl: int = 10,
        max_syl: int = 24,
        speed: float | None = None,
    ) -> Iterator[np.ndarray]:
        """Yield audio chunks as each prosodic chunk is synthesized.

        Trims excessive silence padding from chunk edges so inter-chunk
        boundaries have a natural, gentle breath pause (~40ms) rather
        than awkward 1-second silence gaps.
        """
        self.load()
        voice = self.get_voice(voice_name)
        chunks = chunk_text(text, min_syl=min_syl, max_syl=max_syl) or [text]

        from movio.utils.audio import trim_silence

        for idx, chunk in enumerate(chunks):
            audio = self.synthesize_chunk(chunk, voice, speed=speed)
            if audio is None or len(audio) == 0:
                continue

            import re
            is_sentence_end = bool(re.search(r"[.!?।]\s*$", chunk))
            is_last = (idx == len(chunks) - 1)

            # Human breath timing:
            # - Between sentences (. ! ? ।): 240ms natural breath pause
            # - At clause breaks (, ; :): 130ms gentle pause
            # - Final chunk: 120ms clean decay
            if is_last:
                trail_ms = 120.0
            elif is_sentence_end:
                trail_ms = 240.0
            else:
                trail_ms = 130.0

            trimmed = trim_silence(
                audio,
                threshold_db=-38.0,
                min_silence_ms=30.0,
                trail_silence_ms=trail_ms,
                sample_rate=self.sample_rate,
            )
            if len(trimmed) > 0:
                audio = trimmed

            # Apply a brief 5ms micro-fade to eliminate any boundary clicks without softening onset
            fade_len = int(0.005 * self.sample_rate)
            if len(audio) > 2 * fade_len:
                audio = audio.copy()
                ramp_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
                ramp_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
                audio[:fade_len] *= ramp_in
                audio[-fade_len:] *= ramp_out

            yield audio
