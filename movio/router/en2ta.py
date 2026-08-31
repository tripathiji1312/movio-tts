"""English/Romanized-Tamil → Tamil script preprocessing for IndicF5.

Pipeline (priority order):
  1. Curated lexicon — common transport/English words with natural Tamil loanword forms
  2. IndicXlit — trained on 26M real Roman→Tamil pairs, produces natural orthography
  3. g2p-en ARPAbet fallback — only for truly unknown words IndicXlit can't handle

Why this matters: IndicF5 was trained on natural Tamil orthography. Phonetic
reconstructions from g2p look foreign to the model → robotic prosody.
Natural loanword forms (கேப் not காப்பு) match training data → natural output.
"""

from __future__ import annotations
import re
import logging

logger = logging.getLogger(__name__)

# ── Curated lexicon: common English words with natural Tamil loanword forms ──
# These are the forms Tamil speakers actually write — matching IndicF5 training data.

_LEXICON: dict[str, str] = {
    # Transport domain
    "cab": "கேப்",
    "taxi": "டாக்சி",
    "auto": "ஆட்டோ",
    "bus": "பஸ்",
    "train": "ட்ரெயின்",
    "flight": "ஃப்ளைட்",
    "bike": "பைக்",
    "car": "கார்",
    "pickup": "பிக்கப்",
    "drop": "ட்ராப்",
    "driver": "டிரைவர்",
    "booking": "புக்கிங்",
    "book": "புக்",
    "booked": "புக்ட்",
    "cancel": "கேன்சல்",
    "cancelled": "கேன்சல்ட்",
    "confirm": "கன்ஃபர்ம்",
    "confirmed": "கன்ஃபர்ம்ட்",
    "location": "லொக்கேஷன்",
    "address": "அட்ரஸ்",
    "route": "ரூட்",
    "map": "மேப்",
    "arrive": "அரைவ்",
    "arrival": "அரைவல்",
    "ready": "ரெடி",
    "start": "ஸ்டார்ட்",
    "started": "ஸ்டார்ட்டட்",
    "wait": "வெயிட்",
    "waiting": "வெயிட்டிங்",
    "reached": "ரீச்ட்",
    "share": "ஷேர்",
    "track": "ட்ராக்",
    # OTP / tech
    "otp": "ஓடிபி",
    "pin": "பின்",
    "id": "ஐடி",
    "app": "ஆப்",
    "online": "ஆன்லைன்",
    "offline": "ஆஃப்லைன்",
    "mobile": "மொபைல்",
    "phone": "ஃபோன்",
    "number": "நம்பர்",
    "call": "கால்",
    "message": "மெசேஜ்",
    "chat": "சாட்",
    # Payment
    "pay": "பே",
    "payment": "பேமென்ட்",
    "cash": "கேஷ்",
    "card": "கார்ட்",
    "upi": "யூபிஐ",
    "price": "ப்ரைஸ்",
    "fare": "ஃபேர்",
    "charge": "சார்ஜ்",
    "free": "ஃப்ரீ",
    # Common English words in Tamil conversation
    "please": "ப்ளீஸ்",
    "thank": "தேங்க்",
    "thanks": "தேங்க்ஸ்",
    "sorry": "சாரி",
    "ok": "ஓகே",
    "okay": "ஓகே",
    "yes": "யெஸ்",
    "no": "நோ",
    "hello": "ஹலோ",
    "hi": "ஹாய்",
    "bye": "பை",
    "sir": "சார்",
    "madam": "மேடம்",
    "name": "நேம்",
    "help": "ஹெல்ப்",
    "problem": "ப்ராப்ளம்",
    "issue": "இஷ்யூ",
    "safe": "சேஃப்",
    "fast": "ஃபாஸ்ட்",
    "slow": "ஸ்லோ",
    "minute": "மினிட்",
    "minutes": "மினிட்ஸ்",
    "hour": "அவர்",
    "hours": "அவர்ஸ்",
    "second": "செகண்ட்",
    "service": "சர்வீஸ்",
    "support": "சப்போர்ட்",
    "review": "ரிவ்யூ",
    "rating": "ரேட்டிங்",
    "good": "குட்",
    "bad": "பேட்",
    "safe": "சேஃப்",
    "first": "ஃபர்ஸ்ட்",
    "last": "லாஸ்ட்",
    "next": "நெக்ஸ்ட்",
    "new": "நியூ",
    "done": "டன்",
    "complete": "கம்ப்ளீட்",
    "completed": "கம்ப்ளீட்டட்",
}

# English digit words — spoken individually for codes (4832 → four eight three two)
_DIGIT_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}

# Tamil letter names for abbreviations (OTP → ஓ டீ பீ)
_LETTER_NAMES: dict[str, str] = {
    "A": "ஏ", "B": "பீ", "C": "சீ", "D": "டீ", "E": "ஈ",
    "F": "எஃப்", "G": "ஜீ", "H": "எச்", "I": "ஐ", "J": "ஜே",
    "K": "கே", "L": "எல்", "M": "எம்", "N": "என்", "O": "ஓ",
    "P": "பீ", "Q": "க்யூ", "R": "ஆர்", "S": "எஸ்", "T": "டீ",
    "U": "யூ", "V": "வீ", "W": "டபிள்யூ", "X": "எக்ஸ்",
    "Y": "வை", "Z": "ஸெட்",
}

# Digit sequence: expand digits not in times/decimals
_DIGIT_SEQ_RE = re.compile(r"(?<![:\d\.])(\d{2,6})(?![:\.\d])")
_LATIN_WORD_RE = re.compile(r"[A-Za-z]+")
_TAMIL_RE = re.compile(r"[஀-௿]")

# ── Backend: IndicXlit ───────────────────────────────────────────────────────

_xlit_engine = None
_xlit_tried = False


def _get_xlit():
    global _xlit_engine, _xlit_tried
    if _xlit_tried:
        return _xlit_engine
    _xlit_tried = True
    try:
        from ai4bharat.transliteration import XlitEngine
        _xlit_engine = XlitEngine("ta", beam_width=6, rescore=True, src_script_type="en")
        logger.info("IndicXlit loaded for Tamil transliteration")
    except Exception as e:
        logger.warning("IndicXlit unavailable (%s); falling back to g2p", e)
        _xlit_engine = None
    return _xlit_engine


# ── Backend: g2p ARPAbet fallback ────────────────────────────────────────────

# ARPAbet consonant → Tamil base
_C = {
    "B": "ப", "CH": "ச", "D": "ட", "DH": "த",
    "F": "ஃப", "G": "க", "HH": "ஹ", "JH": "ஜ",
    "K": "க", "L": "ல", "M": "ம", "N": "ந",
    "NG": "ங", "P": "ப", "R": "ர", "S": "ஸ",
    "SH": "ஷ", "T": "ட", "TH": "த", "V": "வ",
    "W": "வ", "Y": "ய", "Z": "ஸ", "ZH": "ஷ",
}
# ARPAbet vowel → (independent, dependent diacritic)
_V = {
    "AA": ("ஆ", "ா"), "AE": ("அ", ""), "AH": ("அ", ""),
    "AO": ("ஆ", "ா"), "AW": ("அவ்", "ாவ்"), "AY": ("ஐ", "ை"),
    "EH": ("எ", "ெ"), "ER": ("அர்", "ர்"), "EY": ("ஏ", "ே"),
    "IH": ("இ", "ி"), "IY": ("ஈ", "ீ"), "OW": ("ஓ", "ோ"),
    "OY": ("ஓய்", "ோய்"), "UH": ("உ", "ு"), "UW": ("யூ", "ூ"),
}

_g2p = None
_g2p_tried = False


def _get_g2p():
    global _g2p, _g2p_tried
    if _g2p_tried:
        return _g2p
    _g2p_tried = True
    try:
        import nltk
        for pkg, path in [("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng"),
                           ("cmudict", "corpora/cmudict")]:
            try:
                nltk.data.find(path)
            except LookupError:
                nltk.download(pkg, quiet=True)
        from g2p_en import G2p
        _g2p = G2p()
    except Exception as e:
        logger.warning("g2p-en unavailable (%s)", e)
    return _g2p


def _arpabet_to_tamil(phones: list[str]) -> str:
    out: list[str] = []
    pending: str | None = None

    def flush():
        nonlocal pending
        if pending is not None:
            out.append(pending + "்")
            pending = None

    for ph in phones:
        bare = ph.rstrip("012")
        if bare in _C:
            flush()
            pending = _C[bare]
        elif bare in _V:
            indep, dep = _V[bare]
            if pending is not None:
                if dep == "":
                    out.append(pending)
                elif dep == "ர்":
                    out.append(pending + "ர்")
                else:
                    out.append(pending + dep)
                pending = None
            else:
                out.append(indep)
        else:
            flush()
            out.append(ph)
    flush()
    return "".join(out)


def _g2p_transliterate(word: str) -> str:
    g2p = _get_g2p()
    if g2p is None:
        return word
    try:
        phones = g2p(word)
        result = _arpabet_to_tamil(phones)
        return result if result.strip() else word
    except Exception:
        return word


# ── Core word transliterator ─────────────────────────────────────────────────

def _is_abbreviation(word: str) -> bool:
    return word.isupper() and len(word) >= 2 and word.isalpha()


def _transliterate_word(word: str) -> str:
    """Convert one Latin word to Tamil script.

    Priority: lexicon → IndicXlit → g2p fallback.
    """
    lower = word.lower()

    # 1. Curated lexicon — natural loanword forms matching training data
    if lower in _LEXICON:
        return _LEXICON[lower]

    # 2. Abbreviations — spell out letter by letter with spaces
    if _is_abbreviation(word):
        return " ".join(_LETTER_NAMES.get(c, c) for c in word)

    # 3. IndicXlit — trained on real Roman→Tamil pairs
    xlit = _get_xlit()
    if xlit is not None:
        try:
            out = xlit.translit_word(lower, topk=1)
            if isinstance(out, dict):
                cand = out.get("ta", "")
                if isinstance(cand, list):
                    cand = cand[0] if cand else ""
                if cand and cand.strip():
                    return cand
            elif isinstance(out, list) and out:
                return out[0]
        except Exception as e:
            logger.debug("IndicXlit failed for %r: %s", word, e)

    # 4. g2p ARPAbet fallback
    return _g2p_transliterate(word)


# ── Sentence-level entry point ───────────────────────────────────────────────

def transliterate_english_to_tamil(text: str) -> str:
    """Convert all Latin-script words and digit codes to Tamil script.

    Tamil-script tokens pass through unchanged. Times (7:30) and decimals
    (3.5) are protected from digit expansion.
    """
    # Expand digit sequences (OTPs, PINs, codes) to individual digit words
    def _expand_digits(m: re.Match) -> str:
        return " ".join(_transliterate_word(_DIGIT_WORDS[d]) for d in m.group(1))

    text = _DIGIT_SEQ_RE.sub(_expand_digits, text)

    # Transliterate Latin words
    parts: list[str] = []
    last_end = 0
    for m in _LATIN_WORD_RE.finditer(text):
        parts.append(text[last_end:m.start()])
        parts.append(_transliterate_word(m.group()))
        last_end = m.end()
    parts.append(text[last_end:])

    return re.sub(r" {2,}", " ", "".join(parts))
