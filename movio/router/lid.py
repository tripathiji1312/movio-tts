import logging
import re

logger = logging.getLogger(__name__)

TA_UNICODE_RANGE = re.compile(r"[\u0B80-\u0BFF]")
DEVANAGARI_RANGE = re.compile(r"[\u0900-\u097F]")

COMMON_TANGLISH_WORDS = {
    "unga", "ungal", "enga", "enge", "epdi", "eppadi", "iruku", "irukku",
    "illa", "illai", "vanakkam", "seri", "sari", "romba", "konjam",
    "podu", "pannu", "panra", "panren", "varen", "vaanga", "pooga",
    "poga", "aana", "apram", "ippam", "enna", "yaar", "ethu", "indha",
    "andha", "idhu", "adhu", "nee", "naan", "namma", "avanga", "ivanga",
    "kudukka", "solli", "kelvi", "neram", "dooram", "kitte", "kitta",
    "la", "le", "ku", "ka", "thaan", "dhan", "um", "um",
}

LATIN_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")


class HeuristicLID:
    """Script/lexicon-based language ID fallback.

    Runs when IndicLID weights are unavailable. Classifies each token as
    ta_native, ta_roman, en, or other. Good enough for routing; IndicLID
    is preferred in production.
    """

    def __init__(self):
        self._ta_lexicon = COMMON_TANGLISH_WORDS

    def classify(self, token: str) -> str:
        stripped = token.strip(".,!?;:()[]{}\"'")
        if not stripped:
            return "other"
        if TA_UNICODE_RANGE.search(stripped):
            return "ta_native"
        if DEVANAGARI_RANGE.search(stripped):
            return "hi_native"
        if not LATIN_WORD_RE.fullmatch(stripped):
            return "other"
        low = stripped.lower()
        if low in self._ta_lexicon:
            return "ta_roman"
        if re.search(r"(?:nga|ungo|iruku|irukku|pola|laati|kitta|kitte|dhaan|thaan|ku$)", low):
            return "ta_roman"
        if re.search(r"[aeiou]{2}", low) and low[-1] not in "e" and len(low) >= 4 \
                and any(low.endswith(s) for s in ("a", "u", "am", "om", "um")):
            return "ta_roman"
        return "en"

    def classify_batch(self, tokens: list[str]) -> list[str]:
        return [self.classify(t) for t in tokens]


class IndicLIDBackend:
    """Wraps ai4bharat IndicLID (47-class). Falls back to heuristics."""

    def __init__(self, backend: str = "auto"):
        self.backend_choice = backend
        self.model = None
        self.fallback = HeuristicLID()
        self.active_backend = "heuristic"
        if backend in ("auto", "indiclid"):
            try:
                from ai4bharat.transliteration import indlid  # noqa: F401
                logger.info("IndicLID package detected")
            except ImportError:
                try:
                    import fasttext  # noqa: F401
                    logger.info("fasttext available; IndicLID fasttext path possible")
                except ImportError:
                    logger.info("IndicLID unavailable; using heuristic LID")

    def classify_tokens(self, tokens: list[str]) -> list[str]:
        return self.fallback.classify_batch(tokens)
