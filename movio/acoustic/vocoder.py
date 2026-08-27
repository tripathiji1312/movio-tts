import logging

import numpy as np

logger = logging.getLogger(__name__)


class Vocoder:
    """Stage D vocoder with fallback chain: built-in (IndicF5) → Vocos."""

    def __init__(self, config: dict):
        cfg = config.get("stage_d", {})
        self.choice = cfg.get("vocoder", "builtin")
        self.vocos_model_id = cfg.get("vocos_model_id", "charactr/vocos-mel-24khz")
        self.crossfade_ms = cfg.get("crossfade_ms", 20)
        self._vocos = None

    @property
    def vocos_available(self) -> bool:
        if self._vocos is not None:
            return True
        try:
            from vocos import Vocos
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._vocos = Vocos.from_pretrained(self.vocos_model_id).to(device).eval()
            logger.info("Vocos loaded on %s (fallback vocoder)", device)
            return True
        except Exception as exc:
            logger.warning("Vocos unavailable (%s); built-in vocoder only", exc)
            return False

    def mel_to_audio(self, mel: "np.ndarray") -> np.ndarray:
        if self.choice == "vocos" and self.vocos_available:
            import torch

            with torch.no_grad():
                mel_t = torch.from_numpy(mel).float()
                if next(self._vocos.parameters()).is_cuda:
                    mel_t = mel_t.to("cuda")
                wav = self._vocos.decode(mel_t).cpu().numpy().squeeze()
            return wav.astype(np.float32)
        raise ValueError("mel_to_audio requires the built-in IndicF5 path or vocos='vocos'")
