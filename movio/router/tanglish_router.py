import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from movio.router.lid import LATIN_WORD_RE, IndicLIDBackend
from movio.router.xlit import XlitBackend

logger = logging.getLogger(__name__)


@dataclass
class RouterResult:
    normalized_text: str
    token_labels: list[tuple[str, str]] = field(default_factory=list)
    cs_boundaries: int = 0
    latency_ms: float = 0.0


class TanglishRouter:
    """Stage B — script-unified routing.

    1. Token-level LID (IndicLID / heuristic): ta_native | ta_roman | en | proper_noun
    2. Latin-Tamil → Tamil Unicode via IndicXlit
    3. English + gazetteer proper nouns kept in Latin
    4. <cs> boundary tokens inserted at language transitions
    """

    def __init__(self, config: dict):
        cfg = config.get("stage_b", {}).get("router", {})
        self.insert_cs = bool(cfg.get("insert_cs_tokens", True))
        self.cs_token = cfg.get("cs_token", "<cs>")
        self.keep_proper_latin = bool(cfg.get("keep_proper_nouns_latin", True))
        self.max_xlit_len = int(cfg.get("max_xlit_word_len", 24))
        gaz_path = cfg.get("gazetteer_path")
        self.gazetteer = self._load_gazetteer(gaz_path)
        self.lid = IndicLIDBackend(backend=cfg.get("lid_backend", "auto"))
        self.xlit = XlitBackend(backend=cfg.get("xlit_backend", "auto"))

    @staticmethod
    def _load_gazetteer(path: str | None) -> set[str]:
        if not path or not Path(path).exists():
            return set()
        return {
            ln.strip().lower()
            for ln in Path(path).read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")
        }

    def _is_gazetteer(self, token: str, phrase_window: list[str], idx: int) -> bool:
        low_tok = token.lower()
        if low_tok in self.gazetteer:
            return True
        for term in self.gazetteer:
            words = term.split()
            n = len(words)
            window = [w.lower() for w in phrase_window[idx : idx + n]]
            if len(window) == n and " ".join(window) == term:
                return True
        return False

    def route(self, text: str) -> RouterResult:
        tokens = text.split()
        if not tokens:
            return RouterResult(normalized_text=text)

        labels = []
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if self.keep_proper_latin and self._is_gazetteer(tok, tokens, i):
                labels.append((tok, "proper_noun"))
                i += 1
                continue
            label = self.lid.fallback.classify(tok) if hasattr(self.lid, "fallback") else "other"
            labels.append((tok, label))
            i += 1

        out_tokens: list[str] = []
        out_labels: list[str] = []
        prev_lang = None
        cs_count = 0
        xlit_fail_cache: dict[str, str] = {}

        for tok, label in labels:
            if label == "ta_roman":
                native = xlit_fail_cache.get(tok.lower())
                if native is None and len(tok) <= self.max_xlit_len:
                    native = self.xlit.translit_word(tok) or ""
                    if native:
                        xlit_fail_cache[tok.lower()] = native
                if native:
                    tok, label = native, "ta_native"

            lang = (
                "ta"
                if label in ("ta_native", "ta_roman")
                else "en" if label in ("en", "proper_noun")
                else "other"
            )
            if (
                self.insert_cs
                and prev_lang is not None
                and lang != "other"
                and prev_lang != "other"
                and lang != prev_lang
            ):
                out_tokens.append(self.cs_token)
                out_labels.append("cs")
                cs_count += 1

            out_tokens.append(tok)
            out_labels.append(label)
            if lang != "other":
                prev_lang = lang

        result_text = re.sub(rf"\s*{re.escape(self.cs_token)}\s*", f" {self.cs_token} ", " ".join(out_tokens))
        return RouterResult(
            normalized_text=result_text,
            token_labels=list(zip(out_tokens, out_labels)),
            cs_boundaries=cs_count,
        )
