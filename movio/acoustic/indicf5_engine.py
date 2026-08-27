import logging
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


class IndicF5Engine:
    """Stage C — IndicF5 acoustic backbone (337M CFM-DiT).

    Loads ai4bharat/IndicF5 lazily; synthesizes chunk-by-chunk so first-chunk
    audio can be emitted before the full utterance finishes. On CUDA it runs
    fp16; num_flow_steps trades quality vs latency (10-14 recommended).
    """

    def __init__(self, config: dict):
        cfg = config.get("stage_c", {})
        self.model_id = cfg.get("model_id", "ai4bharat/IndicF5")
        self.device = cfg.get("device", "cuda")
        self.dtype_str = cfg.get("dtype", "float16")
        self.num_flow_steps = int(cfg.get("num_flow_steps", 12))
        self.sway_coef = float(cfg.get("sway_sampling_coef", -1.0))
        self.default_voice = cfg.get("default_voice", "")
        self.voices = load_voice_profiles(cfg.get("voices_dir", "config/voices"))
        self.sample_rate = int(config.get("server", {}).get("sample_rate", 24000))
        self._model = None

    @property
    def model(self):
        if self._model is None:
            import torch
            from transformers import AutoModel

            dtype = torch.float16 if self.dtype_str == "float16" else torch.float32
            device = (
                self.device
                if self.device != "cuda" or torch.cuda.is_available()
                else "cpu"
            )
            logger.info(
                "Loading %s on %s (%s, %d flow steps)",
                self.model_id, device, self.dtype_str, self.num_flow_steps,
            )
            self._model = AutoModel.from_pretrained(
                self.model_id, trust_remote_code=True, torch_dtype=dtype
            ).to(device)
            self._model.eval()
            self._device = device
        return self._model

    @property
    def device_name(self) -> str:
        _ = self.model
        return getattr(self, "_device", "cpu")

    def get_voice(self, voice_name: str | None) -> VoiceProfile:
        name = voice_name or self.default_voice
        if name in self.voices:
            return self.voices[name]
        if self.voices:
            return next(iter(self.voices.values()))
        raise RuntimeError(
            "No voice profiles found. Add one under config/voices/<name>/voice.yaml "
            "with a reference WAV + transcript."
        )

    def synthesize_chunk(
        self,
        text: str,
        voice: VoiceProfile,
        flow_steps: int | None = None,
    ) -> np.ndarray:
        model = self.model
        steps = flow_steps or self.num_flow_steps
        payload = {
            "text": text,
            "ref_audio": voice.ref_audio_path,
            "ref_text": voice.ref_text,
        }
        kwargs = {}
        if hasattr(model, "infer") :
            kwargs["num_steps"] = steps
        audio = model(payload, **kwargs) if kwargs else model(payload)
        return np.asarray(audio, dtype=np.float32)

    def synthesize_stream(
        self,
        text: str,
        voice: VoiceProfile | None = None,
        min_syl: int = 8,
        max_syl: int = 14,
    ) -> Iterator[np.ndarray]:
        voice = voice or self.get_voice(None)
        chunks = chunk_text(text, min_syl=min_syl, max_syl=max_syl)
        prev_tail = None
        fade_n = ms_to_samples(20, self.sample_rate)
        for chunk in chunks:
            audio = self.synthesize_chunk(chunk, voice)
            if prev_tail is not None:
                n = min(fade_n, len(audio), len(prev_tail))
                if n > 0:
                    head = audio[:n]
                    tail = prev_tail[-n:]
                    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
                    audio[:n] = head * ramp + tail * (1.0 - ramp)
            yield audio
            prev_tail = audio
