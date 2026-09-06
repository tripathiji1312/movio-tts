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
    # Function words where CMU dict gives wrong vowels for Indian English
    "a": "எ", "the": "த", "of": "ஆப்", "for": "போர்", "to": "டு",
    "our": "அவர்", "your": "யுவர்",
    # Abbreviations / units that need expansion (not transliteration)
    "am": "ஏ எம்", "pm": "பி எம்", "a.m.": "ஏ எம்", "p.m.": "பி எம்",
    "otp": "ஓ டீ பீ", "gps": "ஜீ பீ எஸ்",
    "km": "கிலோமீட்டர்", "hr": "அவர்", "hrs": "அவர்ஸ்",
    # Words where auto pipeline produces broken output (hiatus, wrong phonemes)
    "hour": "அவர்", "hours": "அவர்ஸ்",
    "flower": "பிளவர்", "around": "அரவுண்ட்",
    "kilometer": "கிலோமீட்டர்", "kilometers": "கிலோமீட்டர்ஸ்",
    "vehicle": "வெஹிக்கிள்", "waiting": "வெயிட்டிங்",
    # Transport domain — conventional Tamil loanword spellings
    "booking": "புக்கிங்", "booked": "புக்டு",
    "airport": "ஏர்போர்ட்", "cab": "கேப்", "taxi": "டாக்சி",
    "fare": "பேர்", "pickup": "பிக்அப்", "drop": "டிராப்",
    "driver": "டிரைவர்", "traffic": "டிராபிக்",
    "signal": "சிக்னல்", "junction": "ஜங்ஷன்",
    "confirm": "கன்பர்ம்", "confirmed": "கன்பர்ம்டு",
    "cancel": "கேன்சல்", "cancelled": "கேன்சல்டு",
    "arrive": "அரைவ்", "arrived": "அரைவ்டு",
    "auto": "ஆட்டோ", "sir": "சார்", "madam": "மேடம்",
    "rupees": "ரூபீஸ்", "rupee": "ரூபீ", "rupaai": "ரூபாய்",
    "cash": "கேஷ்",
    "delay": "டிலே", "delayed": "டிலேட்",
    "express": "எக்ஸ்பிரஸ்", "platform": "பிளாட்பார்ம்",
    "flyover": "பிளைஓவர்", "highway": "ஹைவே",
    "transport": "டிரான்ஸ்போர்ட்", "transfer": "டிரான்ஸ்பர்",
    "via": "வயா", "station": "ஸ்டேஷன்",
    # Brand/service names where both CMU and IndicXlit get it wrong
    "uber": "ஊபர்", "ola": "ஓலா", "rapido": "ராபிடோ",
    # English numbers — consistent pronunciation for digit sequences
    "zero": "ஸீரோ", "one": "வன்", "two": "டூ", "three": "திரீ",
    "four": "போர்", "five": "பைவ்", "six": "சிக்ஸ்", "seven": "செவன்",
    "eight": "எயிட்", "nine": "நைன்", "ten": "டென்",
    "eleven": "இலெவன்", "twelve": "டுவெல்வ்", "thirteen": "தர்டீன்",
    "fourteen": "போர்டீன்", "fifteen": "பிப்டீன்", "sixteen": "சிக்ஸ்டீன்",
    "seventeen": "செவன்டீன்", "eighteen": "எய்டீன்", "nineteen": "நைன்டீன்",
    "twenty": "டுவென்டி", "thirty": "தர்டி", "forty": "போர்டி",
    "fifty": "பிப்டி", "sixty": "சிக்ஸ்டி", "seventy": "செவன்டி",
    "eighty": "எய்டி", "ninety": "நைன்டி",
    "hundred": "ஹன்ட்ரட்", "thousand": "தவுசண்ட்",
}

# Digits → Tamil loanword forms
_DIGIT_TAMIL = {
    "0": "ஸீரோ", "1": "வன்", "2": "டூ", "3": "திரீ", "4": "போர்",
    "5": "பைவ்", "6": "சிக்ஸ்", "7": "செவன்", "8": "எயிட்", "9": "நைன்",
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
    "F": "எப்", "G": "ஜீ", "H": "எச்", "I": "ஐ", "J": "ஜே",
    "K": "கே", "L": "எல்", "M": "எம்", "N": "என்", "O": "ஓ",
    "P": "பீ", "Q": "க்யூ", "R": "ஆர்", "S": "எஸ்", "T": "டீ",
    "U": "யூ", "V": "வீ", "W": "டபிள்யூ", "X": "எக்ஸ்",
    "Y": "வை", "Z": "ஸெட்",
}

_DIGIT_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"
}

_SYMBOL_NAMES: dict[str, str] = {
    "@": "அட்",
    "#": "ஹாஷ்",
    "$": "டாலர்",
    "%": "சதவீதம்",
    "&": "அண்ட்",
    "*": "ஸ்டார்",
    "+": "பிளஸ்",
    "=": "ஈக்வல்ஸ்",
    "/": "ஸ்லாஷ்",
    "\\": "பேக்ஸ்லாஷ்",
    "|": "பைப்",
    "~": "டில்டா",
    "^": "கேரட்",
    "<": "லெஸ் தான்",
    ">": "கிரேட்டர் தான்",
    "_": "அண்டர்ஸ்கோர்",
    "{": "ஓபன் பிரேஸ்",
    "}": "குளோஸ் பிரேஸ்",
    "[": "ஓபன் பிராக்கெட்",
    "]": "குளோஸ் பிராக்கெட்",
    "©": "காபிரைட்",
    "®": "ரெஜிஸ்டர்ட்",
    "™": "டிரேட்மார்க்",
    "°": "டிகிரீ",
    "₹": "ரூபாய்",
    "€": "யூரோ",
    "£": "பவுண்ட்",
    "¥": "யென்",
    "§": "செக்ஷன்",
    "¶": "பாரா",
    "→": "",
    "←": "",
    "↔": "",
    "•": "",
    "—": "",
    "–": "",
    "…": "",
    "\"": "",
    "'": "",
    "'": "",
    "'": "",
    "“": "",
    "”": "",
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
    """Format license plates with period-separated groups for clear TTS pauses."""
    def raw_repl(m):
        state = " ".join(c.upper() for c in m.group(1))
        rto = " ".join(_DIGIT_WORDS.get(d, d) for d in m.group(2))
        series = " ".join(c.upper() for c in m.group(3))
        reg_digits = [_DIGIT_WORDS.get(d, d) for d in m.group(4)]
        reg = f"{reg_digits[0]} {reg_digits[1]}, {reg_digits[2]} {reg_digits[3]}"
        return f"{state}, {rto}, {series}, {reg}"

    text = _RAW_PLATE_RE.sub(raw_repl, text)

    def spelled_repl(m):
        state = f"{m.group(1).upper()} {m.group(2).upper()}"
        rto = " ".join(m.group(3).split())
        series = " ".join(c.upper() for c in m.group(4).split())
        reg_words = m.group(5).split()
        reg = f"{reg_words[0]} {reg_words[1]}, {reg_words[2]} {reg_words[3]}"
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
    "F": "ப", "G": "க", "HH": "ஹ", "JH": "ஜ",
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


_VIRAMA = "்"
_TAMIL_CONSONANTS = set("கஙசஜஞடணதநபமயரலவழளறனஸஷஹ")

_STOPS = {"க", "ச", "ட", "த", "ப", "ஜ"}
_SIBILANTS = {"ஸ", "ஷ", "ஹ"}
_LIQUIDS = {"ர", "ல", "ள"}
_NASALS = {"ங", "ஞ", "ண", "ந", "ம", "ன"}

_VALID_CLUSTERS = {
    "க்க", "ச்ச", "ட்ட", "த்த", "ப்ப", "ற்ற",
    "ன்ன", "ண்ண", "ம்ம", "ல்ல", "ள்ள",
    "ங்க", "ஞ்ச", "ண்ட", "ந்த", "ம்ப", "ன்ற",
}


def _is_foreign_cluster(c1: str, c2: str) -> bool:
    trigram = c1 + _VIRAMA + c2
    if trigram in _VALID_CLUSTERS:
        return False
    if c1 == c2:
        return False
    if c1 in _STOPS and c2 in _LIQUIDS:
        return True
    return False


def _break_clusters(tamil: str) -> str:
    """Insert helping vowels to break foreign consonant clusters.

    Only breaks clusters that don't exist in Tamil phonology (like ப்ர, க்ல,
    ஸ்ட). Preserves valid Tamil clusters (geminates like க்க, nasal+stop
    like ம்ப, liquid+consonant like ர்க).

    Uses ி (i-matra) for stop+liquid clusters (பிர, கில, டிர) — matches how
    Tamil speakers actually pronounce English loanwords. Uses ி for sibilant
    clusters too (ஸிட, ஷிர).
    """
    if _VIRAMA not in tamil:
        return tamil

    chars = list(tamil)
    result = []
    i = 0
    while i < len(chars):
        c = chars[i]
        if c == _VIRAMA and i > 0 and i + 1 < len(chars):
            prev_c = chars[i - 1]
            next_c = chars[i + 1]
            if prev_c in _TAMIL_CONSONANTS and next_c in _TAMIL_CONSONANTS:
                if _is_foreign_cluster(prev_c, next_c):
                    result.append("ி")
                    if next_c == "ல" and prev_c in _STOPS:
                        chars[i + 1] = "ள"
                else:
                    result.append(c)
            else:
                result.append(c)
        else:
            result.append(c)
        i += 1
    return "".join(result)


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
    raw = "".join(out)
    return _break_clusters(raw)


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
            result = result.replace("ஃப", "ப").replace("ஃ", "")
            return result if result.strip() else None
    except Exception as e:
        logger.debug("IndicXlit failed for %r: %s", word, e)
    return None


# ── Core word transliterator ─────────────────────────────────────────────────

def _is_abbreviation(word: str) -> bool:
    return word.isupper() and len(word) >= 2 and word.isalpha()


_cache: dict[str, str] = {}


def _is_proper_noun(word: str) -> bool:
    return len(word) > 1 and word[0].isupper() and not word.isupper()


def _transliterate_word(word: str) -> str:
    lower = word.lower()

    if lower in _cache:
        return _cache[lower]

    # 1. Override dict — function words + known loanwords (before letter names)
    if lower in _OVERRIDES:
        return _OVERRIDES[lower]

    upper = word.upper()
    if len(word) == 1 and upper in _LETTER_NAMES:
        return _LETTER_NAMES[upper]

    # 2. Abbreviations — spell out letter by letter with natural spacing
    if _is_abbreviation(word):
        return " ".join(_LETTER_NAMES.get(c, c) for c in word)

    # 3. Proper nouns (e.g. Chennai, Rajesh) → IndicXlit first, CMU fallback
    if _is_proper_noun(word):
        result = _xlit_transliterate(word)
        if result is None:
            result = _cmu_transliterate(word)
    else:
        # 4. Common words → CMU dict first, IndicXlit fallback
        result = _cmu_transliterate(word)
        if result is None:
            result = _xlit_transliterate(word)

    # 5. Apply cluster-breaking to all results
    if result is not None and result != word:
        result = _break_clusters(result)

    # 6. Pass through
    if result is None:
        result = word

    _cache[lower] = result
    return result


_ALL_DIGIT_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "பூஜ்யம்", "ஒன்று", "இரண்டு", "மூன்று", "நான்கு", "ஐந்து", "ஆறு", "ஏழு", "எட்டு", "ஒன்பது",
    "ஸீரோ", "வன்", "டூ", "திரீ", "போர்", "பைவ்", "சிக்ஸ்", "செவன்", "எயிட்", "நைன்",
}


def _format_natural_digits(text: str) -> str:
    """Group consecutive digit words into pairs with comma pauses.

    Commas give IndicF5 a light breath pause without triggering chunk splits
    (periods cause the chunker to break mid-sequence, creating laggy gaps).
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
                if bool(re.search(r"[.;:!?।]$", run[-1])):
                    break
                if next_clean in _ALL_DIGIT_WORDS or next_clean.isdigit():
                    run.append(next_w)
                    j += 1
                else:
                    break

            if len(run) <= 2:
                out.extend(run)
            else:
                for k in range(0, len(run), 2):
                    pair = run[k:k+2]
                    stripped = [p.strip(punct_chars) for p in pair]
                    group = " ".join(stripped)
                    if k + 2 < len(run):
                        group += ","
                    else:
                        trail = run[-1][len(run[-1].rstrip(punct_chars)):]
                        group += trail
                    out.append(group)
            i = j
        else:
            out.append(w)
            i += 1
    return " ".join(out)


# ── Sentence-level entry point ───────────────────────────────────────────────

def _expand_symbols(text: str) -> str:
    """Replace symbols with their Tamil spoken equivalents.

    Repeated identical symbols (like * * * * *) are collapsed to a single
    expansion to avoid robotic repetition.
    """
    text = re.sub(r"([^\w\s])([\s]*\1)+", r"\1", text)
    out = []
    for ch in text:
        if ch in _SYMBOL_NAMES:
            replacement = _SYMBOL_NAMES[ch]
            if replacement:
                if out and out[-1] != " ":
                    out.append(" ")
                out.append(replacement)
                out.append(" ")
        else:
            out.append(ch)
    return "".join(out)


def transliterate_english_to_tamil(text: str) -> str:
    """Convert all Latin-script words and digit codes to Tamil script.

    Tamil-script tokens pass through unchanged. Vehicle plates (TN82CS1312),
    times (7:30 AM), and OTP codes are given crisp, natural prosodic cadence
    with human-sounding pauses.
    """
    # 0. Expand ordinals (1st→first, 2nd→second, etc.)
    _ORDINAL_MAP = {
        "1st": "first", "2nd": "second", "3rd": "third", "4th": "fourth",
        "5th": "fifth", "6th": "sixth", "7th": "seventh", "8th": "eighth",
        "9th": "ninth", "10th": "tenth", "11th": "eleventh", "12th": "twelfth",
        "13th": "thirteenth", "14th": "fourteenth", "15th": "fifteenth",
        "16th": "sixteenth", "17th": "seventeenth", "18th": "eighteenth",
        "19th": "nineteenth", "20th": "twentieth", "21st": "twenty first",
        "30th": "thirtieth", "31st": "thirty first",
    }
    def _expand_ordinal(m: re.Match) -> str:
        return _ORDINAL_MAP.get(m.group(0).lower(), m.group(0))
    text = re.sub(r"\b\d{1,2}(?:st|nd|rd|th)\b", _expand_ordinal, text, flags=re.IGNORECASE)

    # 0b. Expand contractions
    _CONTRACTIONS = {
        "don't": "do not", "can't": "cannot", "won't": "will not",
        "wouldn't": "would not", "shouldn't": "should not", "couldn't": "could not",
        "isn't": "is not", "aren't": "are not", "wasn't": "was not",
        "weren't": "were not", "hasn't": "has not", "haven't": "have not",
        "hadn't": "had not", "doesn't": "does not", "didn't": "did not",
        "i'm": "I am", "i've": "I have", "i'll": "I will", "i'd": "I would",
        "he's": "he is", "she's": "she is", "it's": "it is",
        "we're": "we are", "they're": "they are", "you're": "you are",
        "we've": "we have", "they've": "they have", "you've": "you have",
        "we'll": "we will", "they'll": "they will", "you'll": "you will",
        "let's": "let us", "that's": "that is", "who's": "who is",
        "what's": "what is", "there's": "there is", "here's": "here is",
    }
    def _expand_contraction(m: re.Match) -> str:
        return _CONTRACTIONS.get(m.group(0).lower(), m.group(0))
    text = re.sub(r"\b\w+'\w+\b", _expand_contraction, text, flags=re.IGNORECASE)

    # 1. Expand symbols to spoken Tamil words
    text = _expand_symbols(text)

    # 2. Expand vehicle license plates first so letters/digits are not mangled
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
