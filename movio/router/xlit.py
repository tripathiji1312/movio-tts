import logging

logger = logging.getLogger(__name__)


class XlitBackend:
    """Wraps ai4bharat IndicXlit for romanized-Tamil → Tamil-Unicode.

    Falls back to a no-op (tokens pass through) when the package or model
    is unavailable — the pipeline still works, just with lower Tanglish
    naturalness on Latin-script Tamil tokens.
    """

    def __init__(self, backend: str = "auto", lang: str = "ta"):
        self.lang = lang
        self.engine = None
        self.active_backend = "none"
        if backend in ("auto", "indicxlit"):
            try:
                from ai4bharat.transliteration import XlitEngine

                self.engine = XlitEngine(src_script_type="en", beam_width=4, rescore=False)
                self.active_backend = "indicxlit"
                logger.info("IndicXlit loaded")
            except Exception as exc:
                logger.warning("IndicXlit unavailable (%s); transliteration disabled", exc)

    def translit_word(self, word: str) -> str | None:
        if self.engine is None:
            return None
        try:
            out = self.engine.translit_word(word, topk=1)
            if isinstance(out, dict):
                cand = out.get(self.lang)
                if isinstance(cand, list) and cand:
                    return cand[0]
            if isinstance(out, list) and out:
                return out[0]
            return None
        except Exception as exc:
            logger.debug("xlit failed for %r: %s", word, exc)
            return None
