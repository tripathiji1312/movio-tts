import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from movio.textnorm.domain_rules import DomainRuleEngine

logger = logging.getLogger(__name__)

try:
    import pynini
    HAS_PYNINI = True
except ImportError:
    HAS_PYNINI = False

try:
    from nemo_text_processing.text_normalization.normalize import Normalizer
    HAS_NEMO = True
except ImportError:
    HAS_NEMO = False


@dataclass
class NormalizationResult:
    text: str
    backend_used: str
    latency_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)


class TextNormalizer:
    """Stage A — context-aware text normalization.

    Backend chain (first available wins):
      1. indic-text-normalization / Pynini WFST (<1 ms) — if `pynini` + WFST grammars present
      2. NeMo nemo_text_processing (~50 ms first call, cached after)
      3. Built-in domain rule engine (always on as final pass)
    """

    def __init__(self, config: dict):
        cfg = config.get("stage_a", {}).get("text_normalization", {})
        self.use_wfst = bool(cfg.get("use_wfst", True)) and HAS_PYNINI
        self.nemo_fallback = bool(cfg.get("nemo_fallback", True)) and HAS_NEMO
        self.wfst_language = cfg.get("wfst_language", "ta")
        self.gazetteer = self._load_gazetteer(cfg.get("gazetteer_path"))
        self.domain_engine = DomainRuleEngine(language=self.wfst_language)
        self._nemo_normalizers: dict[str, "Normalizer"] = {}
        self._nemo_cache: dict[str, str] = {}
        if not self.use_wfst:
            logger.info("WFST normalizer unavailable; using domain rules only")

    @staticmethod
    def _load_gazetteer(path: str | None) -> set[str]:
        if not path or not Path(path).exists():
            return set()
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        return {ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")}

    def _expand_abbreviations(self, text: str) -> str:
        abbrev_map = {
            "St.": "Street",
            "Rd.": "Road",
            "Nr.": "Near",
            "opp.": "opposite",
        }
        result = text
        for k, v in abbrev_map.items():
            result = result.replace(k, v)
        return result

    def _nemo_normalize(self, text: str, lang: str) -> str | None:
        if not self.nemo_fallback:
            return None
        try:
            if lang not in self._nemo_normalizers:
                self._nemo_normalizers[lang] = Normalizer(
                    input_case="cased", lang=lang
                )
            if text in self._nemo_cache:
                return self._nemo_cache[text]
            out = self._nemo_normalizers[lang].normalize(text)
            self._nemo_cache[text] = out
            return out
        except Exception as exc:
            logger.warning("NeMo normalization failed (%s); skipping", exc)
            return None

    @staticmethod
    def _detect_language(text: str) -> str:
        ta_chars = sum(1 for ch in text if "\u0B80" <= ch <= "\u0BFF")
        latin_chars = sum(1 for ch in text if ch.isascii() and ch.isalpha())
        return "ta" if ta_chars >= latin_chars else "en"

    def normalize(self, text: str) -> NormalizationResult:
        backend = "rules"
        cleaned = text.strip()
        lang = self._detect_language(cleaned)

        if self.use_wfst:
            try:
                from indic_text_normalization import normalize as wfst_normalize

                cleaned = wfst_normalize(cleaned, lang=lang)
                backend = "wfst"
            except Exception as exc:
                logger.debug("WFST unavailable for this input: %s", exc)

        if backend == "rules" and any(ch.isdigit() for ch in cleaned):
            nemo_out = self._nemo_normalize(cleaned, lang)
            if nemo_out:
                cleaned, backend = nemo_out, "nemo"

        engine = self.domain_engine if lang == self.wfst_language else DomainRuleEngine(language=lang)
        cleaned = engine.normalize(cleaned)
        cleaned = self._expand_abbreviations(cleaned)

        return NormalizationResult(text=cleaned, backend_used=backend)


def load_settings(path: str = "config/settings.yaml") -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)
