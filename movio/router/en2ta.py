"""English/Romanized-Tamil → Tamil script preprocessing for IndicF5.

Three-tier fully local pipeline (zero network calls):
  1. Override dict — function words + common loanwords where phonetic mapping fails
  2. CMU Dict + ARPAbet→Tamil — 123K English words, <0.1ms/word
  3. IndicXlit via CTranslate2 — Indian names, Tanglish, unknown words, ~1ms/word
"""

from __future__ import annotations
import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Tier 1: Override dictionary ──────────────────────────────────────────────
# Function words where phonetic mapping gives unnatural results, plus a few
# high-frequency loanwords with conventional Tamil spellings.

_OVERRIDES: dict[str, str] = {
    # Function words
    "is": "இஸ்", "are": "ஆர்", "was": "வாஸ்", "were": "வேர்",
    "has": "ஹாஸ்", "have": "ஹேவ்", "had": "ஹாட்",
    "not": "நாட்", "but": "பட்", "and": "அண்ட்", "or": "ஆர்",
    "for": "ஃபார்", "from": "ஃப்ரம்", "with": "வித்",
    "the": "த", "a": "எ", "an": "ஆன்",
    "to": "டு", "in": "இன்", "on": "ஆன்", "at": "அட்", "by": "பை", "of": "ஆஃப்",
    "your": "யுவர்", "you": "யூ", "my": "மை", "our": "அவர்",
    "his": "ஹிஸ்", "her": "ஹர்", "its": "இட்ஸ்",
    "this": "திஸ்", "that": "தட்", "it": "இட்",
    "will": "வில்", "can": "கேன்", "do": "டூ",
    "no": "நோ", "yes": "யெஸ்", "ok": "ஓகே", "okay": "ஓகே",
    "please": "ப்ளீஸ்", "thank": "தேங்க்", "thanks": "தேங்க்ஸ்", "sorry": "சாரி",
    # Loanwords with conventional Tamil spellings (CMU phonetic mapping gets these wrong)
    "booking": "புக்கிங்", "booked": "புக்ட்",
    "airport": "ஏர்போர்ட்", "station": "ஸ்டேஷன்",
    "phone": "ஃபோன்", "mobile": "மொபைல்", "message": "மெசேஜ்",
    "share": "ஷேர்", "payment": "பேமெண்ட்",
    "cancel": "கேன்சல்", "confirm": "கன்பர்ம்", "confirmed": "கன்பர்ம்ட்",
    "taxi": "டாக்சி", "cab": "கேப்", "ready": "ரெடி", "waiting": "வெயிட்டிங்",
    "pickup": "பிக்கப்", "drop": "ட்ராப்",
    "fare": "ஃபேர்", "sir": "சார்", "madam": "மேடம்",
    "arrive": "அரைவ்", "arrived": "அரைவ்ட்", "arrives": "அரைவ்ஸ்",
    "service": "சர்வீஸ்", "minutes": "மினிட்ஸ்", "minute": "மினிட்",
    "hours": "அவர்ஸ்", "hour": "அவர்",
    "kilometer": "கிலோமீட்டர்", "kilometers": "கிலோமீட்டர்ஸ்", "km": "கிலோமீட்டர்",
    "license": "லைசென்ஸ்", "licence": "லைசென்ஸ்",
    "number": "நம்பர்", "numbers": "நம்பர்ஸ்",
    "vehicle": "வெஹிக்கிள்", "car": "கார்", "auto": "ஆட்டோ",
    "driver": "ட்ரைவர்", "driver's": "ட்ரைவர்ஸ்",
    "otp": "ஓ டீ பீ", "pin": "பின்", "code": "கோட்",
    "location": "லொகேஷன்", "route": "ரூட்", "traffic": "ட்ராஃபிக்",
    "toll": "டோல்", "gate": "கேட்",
    "rupees": "ரூபீஸ்", "rupee": "ரூபீ",
    "cash": "கேஷ்", "online": "ஆன்லைன்", "paid": "பெய்ட்",
    "am": "ஏ எம்", "pm": "பி எம்", "a.m.": "ஏ எம்", "p.m.": "பி எம்",
    "oh": "ஓ", "point": "பாயிண்ட்",
    # English numbers for natural speech & time in cab apps
    "zero": "ஸீரோ", "one": "வன்", "two": "டூ", "three": "த்ரீ",
    "four": "ஃபோர்", "five": "ஃபைவ்", "six": "சிக்ஸ்", "seven": "செவன்",
    "eight": "எயிட்", "nine": "நைன்", "ten": "டென்",
    "eleven": "இலெவன்", "twelve": "ட்வெல்வ்", "thirteen": "தர்ட்டீன்",
    "fourteen": "ஃபோர்ட்டீன்", "fifteen": "ஃபிஃப்ட்டீன்", "sixteen": "சிக்ஸ்டீன்",
    "seventeen": "செவன்ட்டீன்", "eighteen": "எய்ட்டீன்", "nineteen": "நைன்ட்டீன்",
    "twenty": "ட்வென்ட்டி", "thirty": "தேர்ட்டி", "forty": "ஃபோர்ட்டி",
    "fifty": "ஃபிஃப்ட்டி", "sixty": "சிக்ஸ்டி", "seventy": "செவன்ட்டி",
    "eighty": "எய்ட்டி", "ninety": "நைன்ட்டி",
    "hundred": "ஹண்ட்ரட்", "thousand": "தௌசண்ட்",
}

# Digits → Tamil loanword forms
_DIGIT_TAMIL = {
    "0": "ஸீரோ", "1": "வன்", "2": "டூ", "3": "த்ரீ", "4": "ஃபோர்",
    "5": "ஃபைவ்", "6": "சிக்ஸ்", "7": "செவன்", "8": "எயிட்", "9": "நைன்",
}

# Tamil number words for time (7:30 → ஏழு முப்பது)
_TAMIL_NUMS = {
    0: "சுழியம்", 1: "ஒன்று", 2: "இரண்டு", 3: "மூன்று", 4: "நான்கு",
    5: "ஐந்து", 6: "ஆறு", 7: "ஏழு", 8: "எட்டு", 9: "ஒன்பது", 10: "பத்து",
    11: "பதினொன்று", 12: "பன்னிரண்டு", 13: "பதிமூன்று", 14: "பதினான்கு",
    15: "பதினைந்து", 16: "பதினாறு", 17: "பதினேழு", 18: "பதினெட்டு",
    19: "பத்தொன்பது", 20: "இருபது", 21: "இருபத்தொன்று", 22: "இருபத்திரண்டு",
    23: "இருபத்துமூன்று", 24: "இருபத்தினான்கு", 25: "இருபத்தைந்து",
    30: "முப்பது", 35: "முப்பத்தைந்து", 40: "நாற்பது", 45: "நாற்பத்தைந்து",
    50: "ஐம்பது", 55: "ஐம்பத்தைந்து",
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

_DIGIT_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"
}

# Indian vehicle registration plates (e.g. TN82CS1312, KA 01 AB 1234, DL3CAA1111)
_RAW_PLATE_RE = re.compile(
    r"\b([A-Za-z]{2})[\s-]?(\d{1,2})[\s-]?([A-Za-z]{1,3})[\s-]?(\d{4})\b"
)

# Spelled plate from normalizer: "T N eight two C S one three one two"
_SPELLED_PLATE_RE = re.compile(
    r"\b([A-Za-z])\s+([A-Za-z])\s+((?:[a-z]+|\d)\s+(?:[a-z]+|\d))\s+([A-Za-z](?:\s+[A-Za-z]){0,2})\s+((?:[a-z]+|\d)\s+(?:[a-z]+|\d)\s+(?:[a-z]+|\d)\s+(?:[a-z]+|\d))\b",
    re.IGNORECASE
)


def _format_vehicle_plate(text: str) -> str:
    """Format license plates with distinct letter pauses and paired numbers.

    e.g. TN82CS1312 -> T - N, eight - two, C - S, one - three, one - two
    """
    def raw_repl(m):
        state = " - ".join(c.upper() for c in m.group(1))
        rto_digits = [_DIGIT_WORDS.get(d, d) for d in m.group(2)]
        rto = " - ".join(rto_digits)
        series = " - ".join(c.upper() for c in m.group(3))
        reg_digits = [_DIGIT_WORDS.get(d, d) for d in m.group(4)]
        reg = f"{reg_digits[0]} - {reg_digits[1]}, {reg_digits[2]} - {reg_digits[3]}"
        return f"{state}, {rto}, {series}, {reg}"

    text = _RAW_PLATE_RE.sub(raw_repl, text)

    def spelled_repl(m):
        state = f"{m.group(1).upper()} - {m.group(2).upper()}"
        rto = " - ".join(m.group(3).split())
        series = " - ".join(c.upper() for c in m.group(4).split())
        reg_words = m.group(5).split()
        reg = f"{reg_words[0]} - {reg_words[1]}, {reg_words[2]} - {reg_words[3]}"
        return f"{state}, {rto}, {series}, {reg}"

    text = _SPELLED_PLATE_RE.sub(spelled_repl, text)
    return text


_EN_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen"
]
_EN_TENS = ["", "", "twenty", "thirty", "forty", "fifty"]


def _english_num_word(n: int) -> str:
    if n < 20:
        return _EN_ONES[n]
    t, o = divmod(n, 10)
    return _EN_TENS[t] + (f" {_EN_ONES[o]}" if o else "")


_TIME_RE = re.compile(
    r"\b(\d{1,2}):(\d{2})(?:\s?(A\.?M\.?|P\.?M\.?|am|pm))?\b",
    re.IGNORECASE
)


def _format_time_english(h: int, m: int, suffix: str | None = None) -> str:
    """Format time in natural English (e.g. 7:30 am -> seven thirty AM)."""
    h12 = ((h - 1) % 12) + 1
    h_str = _EN_ONES[h12]
    if m == 0:
        m_str = ""
    elif m < 10:
        m_str = f" oh {_EN_ONES[m]}"
    else:
        m_str = f" {_english_num_word(m)}"
    s_upper = (suffix or "").upper().replace(".", "")
    if "P" in s_upper:
        period = " PM"
    elif "A" in s_upper:
        period = " AM"
    else:
        period = ""
    return f"{h_str}{m_str}{period}".strip()


# Regex patterns
_DIGIT_SEQ_RE = re.compile(r"(?<![:\d])(\d{2,6})(?![:\.]\d)")
_LATIN_WORD_RE = re.compile(r"[A-Za-z]+")


# ── Tier 2: CMU Dict + ARPAbet → Tamil ──────────────────────────────────────

_ARPA_CONSONANT: dict[str, str] = {
    "B": "ப", "CH": "ச", "D": "ட", "DH": "த",
    "F": "ஃப", "G": "க", "HH": "ஹ", "JH": "ஜ",
    "K": "க", "L": "ல", "M": "ம", "N": "ந",
    "NG": "ங", "P": "ப", "R": "ர", "S": "ஸ",
    "SH": "ஷ", "T": "ட", "TH": "த", "V": "வ",
    "W": "வ", "Y": "ய", "Z": "ஸ", "ZH": "ஷ",
}

_ARPA_VOWEL: dict[str, tuple[str, str]] = {
    "AA": ("ஆ", "ா"), "AE": ("ஆ", "ா"), "AH": ("அ", ""),
    "AO": ("ஆ", "ா"), "AW": ("அவ்", "வ்"), "AY": ("ஐ", "ை"),
    "EH": ("எ", "ெ"), "ER": ("அர்", "ர்"), "EY": ("ஏ", "ே"),
    "IH": ("இ", "ி"), "IY": ("ஈ", "ீ"), "OW": ("ஓ", "ோ"),
    "OY": ("ஓய்", "ோய்"), "UH": ("உ", "ு"), "UW": ("ஊ", "ூ"),
}

_cmu_dict: dict[str, list] | None = None
_cmu_loaded = False


def _get_cmu() -> dict[str, list] | None:
    global _cmu_dict, _cmu_loaded
    if _cmu_loaded:
        return _cmu_dict
    _cmu_loaded = True
    try:
        import nltk
        try:
            nltk.data.find("corpora/cmudict")
        except LookupError:
            nltk.download("cmudict", quiet=True)
        from nltk.corpus import cmudict
        _cmu_dict = cmudict.dict()
        logger.info("CMU dict loaded: %d words", len(_cmu_dict))
    except Exception as e:
        logger.warning("CMU dict unavailable: %s", e)
    return _cmu_dict


def _arpabet_to_tamil(phones: list[str]) -> str:
    out: list[str] = []
    pending_consonant: str | None = None

    def flush():
        nonlocal pending_consonant
        if pending_consonant is not None:
            out.append(pending_consonant + "்")
            pending_consonant = None

    for i, ph in enumerate(phones):
        bare = ph.rstrip("012")
        if bare in _ARPA_CONSONANT:
            flush()
            c = _ARPA_CONSONANT[bare]
            # N at word-final or before another consonant → ன் (alveolar), not ந (dental)
            if bare == "N":
                next_bare = phones[i + 1].rstrip("012") if i + 1 < len(phones) else None
                if next_bare is None or next_bare in _ARPA_CONSONANT:
                    c = "ன"
            pending_consonant = c
        elif bare in _ARPA_VOWEL:
            indep, dep = _ARPA_VOWEL[bare]
            if pending_consonant is not None:
                out.append(pending_consonant + dep)
                pending_consonant = None
            else:
                out.append(indep)
        else:
            flush()
    flush()
    return "".join(out)


def _cmu_transliterate(word: str) -> str | None:
    cmu = _get_cmu()
    if cmu is None:
        return None
    phones = cmu.get(word.lower())
    if not phones:
        return None
    result = _arpabet_to_tamil(phones[0])
    return result if result.strip() else None


# ── Tier 3: IndicXlit via CTranslate2 ───────────────────────────────────────

_xlit_translator = None
_xlit_loaded = False
_xlit_src_vocab: dict[str, int] = {}
_xlit_tgt_vocab: list[str] = []


_XLIT_MODEL_DIR = str(Path(__file__).resolve().parent.parent.parent / "models" / "indicxlit_ct2")


def _load_xlit():
    global _xlit_translator, _xlit_loaded, _xlit_src_vocab, _xlit_tgt_vocab
    if _xlit_loaded:
        return _xlit_translator
    _xlit_loaded = True
    try:
        import ctranslate2
        import json

        ct2_dir = _XLIT_MODEL_DIR
        if not Path(ct2_dir).exists():
            from huggingface_hub import snapshot_download
            repo = snapshot_download("Singla0009/all-indic-transliteration")
            ct2_dir = f"{repo}/indicxlit_ct2_fp32"

        _xlit_translator = ctranslate2.Translator(
            ct2_dir, device="cpu", inter_threads=1,
        )

        with open(f"{ct2_dir}/source_vocabulary.json") as f:
            _xlit_src_vocab = json.load(f)
        with open(f"{ct2_dir}/target_vocabulary.json") as f:
            _xlit_tgt_vocab = json.load(f)

        logger.info("IndicXlit CTranslate2 loaded from %s", ct2_dir)
    except Exception as e:
        logger.warning("IndicXlit CTranslate2 unavailable: %s", e)
        _xlit_translator = None
    return _xlit_translator


def _xlit_transliterate(word: str) -> str | None:
    translator = _load_xlit()
    if translator is None:
        return None
    try:
        # IndicXlit expects character-level input with language tag
        src_chars = list(word.lower())
        lang_tag = "__ta__"
        src_tokens = [lang_tag] + src_chars

        results = translator.translate_batch(
            [src_tokens], beam_size=4, max_decoding_length=64,
        )
        if results and results[0].hypotheses:
            out_tokens = results[0].hypotheses[0]
            result = "".join(out_tokens)
            return result if result.strip() else None
    except Exception as e:
        logger.debug("IndicXlit failed for %r: %s", word, e)
    return None


# ── Core word transliterator ─────────────────────────────────────────────────

def _is_abbreviation(word: str) -> bool:
    return word.isupper() and len(word) >= 2 and word.isalpha()


_cache: dict[str, str] = {}


def _transliterate_word(word: str) -> str:
    lower = word.lower()

    if lower in _cache:
        return _cache[lower]

    upper = word.upper()
    if len(word) == 1 and upper in _LETTER_NAMES:
        return _LETTER_NAMES[upper]

    # 1. Abbreviations — spell out letter by letter with natural spacing
    if _is_abbreviation(word):
        return " ".join(_LETTER_NAMES.get(c, c) for c in word)

    # 2. Override dict — function words + known loanwords
    if lower in _OVERRIDES:
        return _OVERRIDES[lower]

    # 3. CMU Dict + phoneme mapping — 123K English words
    result = _cmu_transliterate(word)

    # 4. IndicXlit CTranslate2 — Indian names, Tanglish, unknown words
    if result is None:
        result = _xlit_transliterate(word)

    # 5. Pass through
    if result is None:
        result = word

    _cache[lower] = result
    return result


_ALL_DIGIT_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "பூஜ்யம்", "ஒன்று", "இரண்டு", "மூன்று", "நான்கு", "ஐந்து", "ஆறு", "ஏழு", "எட்டு", "ஒன்பது",
    "ஸீரோ", "வன்", "டூ", "த்ரீ", "ஃபோர்", "ஃபைவ்", "சிக்ஸ்", "செவன்", "எயிட்", "நைன்",
}


def _format_natural_digits(text: str) -> str:
    """Format consecutive digit sequences into natural human prosodic cadence.

    Instead of robotic comma inflections on every single digit (which sounds like
    a robotic phone tree), this groups numbers into natural human breath units:
      - 4-digit OTPs: 'four - eight, three - two' (2 pairs with subtle micro-gap and mid pause)
      - 6-digit codes: '1 - 2, 3 - 4, 5 - 6'
      - 10-digit phones: '9 - 8, 7 - 6, 5 - 4, 3 - 2, 1 - 0'
    This gives clear separation between digits (no blurred 'two in three') while
    sounding fluid, natural, and human.
    """
    words = text.split()
    if not words:
        return text

    punct_chars = ".,;:!?—…।\n\"'()[]{}"
    out: list[str] = []
    i = 0
    while i < len(words):
        w = words[i]
        clean = w.strip(punct_chars).lower()
        if clean in _ALL_DIGIT_WORDS or clean.isdigit():
            run = [w]
            j = i + 1
            while j < len(words):
                next_w = words[j]
                next_clean = next_w.strip(punct_chars).lower()
                if bool(re.search(r"[,;:!?.।]$", run[-1])):
                    break
                if next_clean in _ALL_DIGIT_WORDS or next_clean.isdigit():
                    run.append(next_w)
                    j += 1
                else:
                    break

            if len(run) == 1:
                out.append(run[0])
            elif len(run) == 4:
                d1 = run[0].strip(punct_chars)
                d2 = run[1].strip(punct_chars)
                d3 = run[2].strip(punct_chars)
                d4 = run[3].rstrip(punct_chars)
                trail_punct = run[3][len(d4):]
                out.append(f"{d1} - {d2}, {d3} - {d4}{trail_punct}")
            elif len(run) == 6:
                d1 = run[0].strip(punct_chars)
                d2 = run[1].strip(punct_chars)
                d3 = run[2].strip(punct_chars)
                d4 = run[3].strip(punct_chars)
                d5 = run[4].strip(punct_chars)
                d6 = run[5].rstrip(punct_chars)
                trail_punct = run[5][len(d6):]
                out.append(f"{d1} - {d2}, {d3} - {d4}, {d5} - {d6}{trail_punct}")
            else:
                parts = []
                for k in range(0, len(run), 2):
                    pair = run[k:k+2]
                    clean_pair = [p.strip(punct_chars) for p in pair]
                    parts.append(" - ".join(clean_pair))
                last_punct = run[-1][len(run[-1].rstrip(punct_chars)):]
                out.append(", ".join(parts) + last_punct)
            i = j
        else:
            out.append(w)
            i += 1
    return " ".join(out)


# ── Sentence-level entry point ───────────────────────────────────────────────

def transliterate_english_to_tamil(text: str) -> str:
    """Convert all Latin-script words and digit codes to Tamil script.

    Tamil-script tokens pass through unchanged. Vehicle plates (TN82CS1312),
    times (7:30 AM), and OTP codes are given crisp, natural prosodic cadence
    with human-sounding pauses.
    """
    # 1. Expand vehicle license plates first so letters/digits are not mangled
    text = _format_vehicle_plate(text)

    # 2. Expand times in English (e.g. 7:30 am -> seven thirty AM)
    def _expand_time(m: re.Match) -> str:
        h, mn = int(m.group(1)), int(m.group(2))
        suffix = m.group(3)
        return _format_time_english(h, mn, suffix)

    text = _TIME_RE.sub(_expand_time, text)

    def _expand_digits(m: re.Match) -> str:
        return " ".join(_DIGIT_TAMIL[d] for d in m.group(1))

    text = _DIGIT_SEQ_RE.sub(_expand_digits, text)
    text = _format_natural_digits(text)

    parts: list[str] = []
    last_end = 0
    for m in _LATIN_WORD_RE.finditer(text):
        parts.append(text[last_end:m.start()])
        parts.append(_transliterate_word(m.group()))
        last_end = m.end()
    parts.append(text[last_end:])

    joined = re.sub(r" {2,}", " ", "".join(parts))
    return _format_natural_digits(joined)
