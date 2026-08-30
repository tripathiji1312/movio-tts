"""IndicF5Engine — F5-TTS CFM-DiT acoustic backbone (337M params, 24 kHz).

Uses the F5-TTS inference stack directly. Supports both the ai4bharat/IndicF5
base model (MIT licensed) and a locally exported fine-tuned bundle.

model_path config options:
  "base"             — downloads ai4bharat/IndicF5 weights from HF hub (MIT)
  "/path/to/bundle"  — fine-tuned bundle dir (model.pt + vocab.txt + config.json)
"""

import logging
import sys
from pathlib import Path
from typing import Iterator

import numpy as np

from movio.acoustic.chunking import chunk_text
from movio.utils.audio import ms_to_samples

logger = logging.getLogger(__name__)


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
    for meta_path in sorted(voices_dir.glob("*/voice.yaml")):
        import yaml
        meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
        audio_rel = meta.get("ref_audio")
        profile = VoiceProfile(
            name=meta.get("name", meta_path.parent.name),
            ref_audio_path=str((meta_path.parent / audio_rel).resolve()),
            ref_text=meta.get("ref_text", ""),
        )
        profiles[profile.name] = profile
    return profiles


def _patch_f5tts_init(f5tts_src: Path) -> None:
    """Wrap Trainer import in try/except so __init__.py doesn't trigger
    dataset.py → datasets 5.0.0 → huggingface_hub.parse_hf_uri crash."""
    init_py = f5tts_src / "src" / "f5_tts" / "model" / "__init__.py"
    if not init_py.exists():
        return
    src = init_py.read_text()
    if "try:" not in src and "from f5_tts.model.trainer import Trainer" in src:
        src = src.replace(
            "from f5_tts.model.trainer import Trainer",
            "try:\n    from f5_tts.model.trainer import Trainer\nexcept ImportError:\n    Trainer = None",
        )
        init_py.write_text(src)
        logger.info("Patched f5_tts/model/__init__.py (optional Trainer import)")


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
        self.f5tts_dir = cfg.get("f5tts_dir")  # path to cloned F5-TTS repo; None if pip-installed
        self.device = cfg.get("device", "cuda")
        self.num_flow_steps = int(cfg.get("num_flow_steps", 12))
        self.sway_coef = float(cfg.get("sway_sampling_coef", -1.0))
        self.cfg_strength = float(cfg.get("cfg_strength", 2.0))
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

    def _ensure_path(self) -> None:
        if self.f5tts_dir:
            src = str(Path(self.f5tts_dir) / "src")
            if src not in sys.path:
                sys.path.insert(0, src)
            _patch_f5tts_init(Path(self.f5tts_dir))

    def _vocab_path_for_base(self) -> str:
        if self.f5tts_dir:
            p = Path(self.f5tts_dir) / "src" / "f5_tts" / "infer" / "examples" / "vocab.txt"
            if p.exists():
                return str(p)
        try:
            import f5_tts
            p = Path(f5_tts.__file__).parent / "infer" / "examples" / "vocab.txt"
            if p.exists():
                return str(p)
        except ImportError:
            pass
        raise RuntimeError(
            "Cannot locate F5-TTS vocab.txt. Set stage_c.indicf5.f5tts_dir in settings.yaml."
        )

    # ── model loading ────────────────────────────────────────────────────────

    def _load(self) -> None:
        import torch
        self._ensure_path()

        from f5_tts.model.backbones.dit import DiT
        from f5_tts.model.cfm import CFM
        from f5_tts.infer.utils_infer import get_tokenizer, load_checkpoint, load_vocoder

        device = (
            self.device
            if self.device != "cuda" or torch.cuda.is_available()
            else "cpu"
        )
        self._device = device

        if self.model_path == "base":
            # ai4bharat/IndicF5 — MIT licensed, Tamil-aware, commercially usable.
            from huggingface_hub import hf_hub_download
            ckpt_path = hf_hub_download(
                "ai4bharat/IndicF5", "model.safetensors"
            )
            vocab_path = self._vocab_path_for_base()
            use_ema = False  # IndicF5 safetensors are not EMA-wrapped
        else:
            bundle = Path(self.model_path)
            if not bundle.is_dir():
                raise RuntimeError(
                    f"IndicF5 bundle not found: {bundle}. "
                    "Run training/scripts/05_merge_export.py first."
                )
            ckpt_path = str(bundle / "model.pt")
            vocab_path = str(bundle / "vocab.txt")
            use_ema = False

        vocab_char_map, vocab_size = get_tokenizer(vocab_path, "custom")

        model = CFM(
            transformer=DiT(**self._MODEL_CFG, text_num_embeds=vocab_size, mel_dim=100),
            mel_spec_kwargs=dict(
                n_fft=1024, hop_length=256, win_length=1024,
                n_mel_channels=100, target_sample_rate=self.SAMPLE_RATE,
                mel_spec_type="vocos",
            ),
            vocab_char_map=vocab_char_map,
        )
        load_checkpoint(model, ckpt_path, device=device, use_ema=use_ema)
        model = model.to(device).eval()

        vocoder = load_vocoder(vocoder_name="vocos", is_local=False, device=device)

        self._model = model
        self._vocoder = vocoder
        self._vocab_char_map = vocab_char_map
        logger.info(
            "IndicF5Engine ready: model=%s device=%s flow_steps=%d",
            self.model_path, device, self.num_flow_steps,
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
        key = voice.ref_audio_path
        if key not in self._ref_cache:
            self._ensure_path()
            from f5_tts.infer.utils_infer import preprocess_ref_audio_text
            ref_audio, ref_text = preprocess_ref_audio_text(
                voice.ref_audio_path, voice.ref_text
            )
            self._ref_cache[key] = (ref_audio, ref_text)
        return self._ref_cache[key]

    # ── synthesis ────────────────────────────────────────────────────────────

    def synthesize_chunk(
        self,
        text: str,
        voice: VoiceProfile,
        flow_steps: int | None = None,
    ) -> np.ndarray:
        self.load()
        self._ensure_path()
        from f5_tts.infer.utils_infer import infer_process

        ref_audio, ref_text = self._get_ref_audio(voice)
        steps = flow_steps or self.num_flow_steps

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
            speed=self.speed,
        )
        return np.asarray(audio, dtype=np.float32)

    def synthesize(self, text: str, voice_name: str | None = None) -> np.ndarray:
        """Synthesize full text in one pass. Use synthesize_stream for low TTFA."""
        voice = self.get_voice(voice_name)
        return self.synthesize_chunk(text, voice)

    def synthesize_stream(
        self,
        text: str,
        voice_name: str | None = None,
        min_syl: int = 8,
        max_syl: int = 14,
    ) -> Iterator[np.ndarray]:
        """Yield audio chunks as each prosodic chunk is synthesized.

        First chunk is emitted after ~one synthesis pass (~250-400ms on T4),
        so TTFA is roughly the cost of one chunk rather than the full utterance.
        """
        self.load()
        voice = self.get_voice(voice_name)
        chunks = chunk_text(text, min_syl=min_syl, max_syl=max_syl) or [text]

        fade_n = ms_to_samples(20, self.sample_rate)
        prev_tail: np.ndarray | None = None

        for chunk in chunks:
            audio = self.synthesize_chunk(chunk, voice)
            if audio is None or len(audio) == 0:
                continue
            if prev_tail is not None:
                n = min(fade_n, len(audio), len(prev_tail))
                if n > 0:
                    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
                    audio = audio.copy()
                    audio[:n] = audio[:n] * ramp + prev_tail[-n:] * (1.0 - ramp)
            yield audio
            prev_tail = audio
