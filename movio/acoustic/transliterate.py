"""English-to-Tamil transliteration for VITS synthesis.

Since the Tamil VITS model only has Tamil characters in its vocabulary,
English words must be transliterated to Tamil script before synthesis.
This uses a basic phonetic mapping that covers common English sounds.

For proper nouns and domain terms, the gazetteer provides exact mappings.
"""

import re

EN_TO_TA = {
    "a": "அ", "aa": "ஆ", "i": "இ", "ee": "ஈ", "u": "உ", "oo": "ஊ",
    "e": "எ", "ai": "ஐ", "o": "ஒ", "au": "ஔ",
    "ka": "க", "kha": "க", "ga": "க", "gha": "க",
    "cha": "ச", "ja": "ஜ", "jha": "ஜ",
    "ta": "ட", "da": "ட", "tha": "த", "dha": "த",
    "na": "ந", "pa": "ப", "ba": "ப", "pha": "ப",
    "ma": "ம", "ya": "ய", "ra": "ர", "la": "ல",
    "va": "வ", "wa": "வ", "sha": "ஷ", "sa": "ச", "ha": "ஹ",
}

CONSONANT_MAP = {
    "k": "க்", "g": "க்", "c": "க்", "q": "க்",
    "ch": "ச்", "j": "ஜ்", "s": "ச்", "z": "ச்",
    "t": "ட்", "d": "ட்", "th": "த்", "dh": "த்",
    "n": "ந்", "p": "ப்", "b": "ப்", "f": "ப்",
    "m": "ம்", "y": "ய்", "r": "ர்", "l": "ல்",
    "v": "வ்", "w": "வ்", "sh": "ஷ்", "h": "ஹ்",
    "x": "க்ச்",
}

VOWEL_MAP = {
    "a": "அ", "e": "எ", "i": "இ", "o": "ஒ", "u": "உ",
    "aa": "ஆ", "ee": "ஈ", "oo": "ஊ", "ai": "ஐ", "au": "ஔ",
    "ou": "ஔ", "ei": "ஏ",
}

VOWEL_SIGN_MAP = {
    "a": "ா", "e": "ெ", "i": "ி", "o": "ொ", "u": "ு",
    "aa": "ா", "ee": "ீ", "oo": "ூ", "ai": "ை", "au": "ௌ",
    "ou": "ௌ", "ei": "ே",
}

COMMON_WORDS = {
    "your": "யுவர்",
    "otp": "ஓடிபி",
    "is": "இஸ்",
    "four": "போர்",
    "eight": "எயிட்",
    "three": "த்ரீ",
    "two": "டூ",
    "one": "வன்",
    "five": "பைவ்",
    "six": "சிக்ஸ்",
    "seven": "செவன்",
    "nine": "நைன்",
    "zero": "சீரோ",
    "please": "ப்ளீஸ்",
    "share": "ஷேர்",
    "it": "இட்",
    "with": "வித்",
    "the": "த",
    "driver": "ட்ரைவர்",
    "booking": "புக்கிங்",
    "confirm": "கன்பர்ம்",
    "name": "நேம்",
    "fare": "பேர்",
    "for": "போர்",
    "point": "பாயிண்ட்",
    "hundred": "ஹன்ட்ரட்",
    "thousand": "தௌசண்ட்",
    "rupees": "ரூபீஸ்",
    "kilometers": "கிலோமீட்டர்ஸ்",
    "minutes": "மினிட்ஸ்",
    "arriving": "அரைவிங்",
    "arrive": "அரைவ்",
    "pickup": "பிக்அப்",
    "ready": "ரெடி",
    "cab": "கேப்",
    "trip": "ட்ரிப்",
    "cancel": "கேன்சல்",
    "location": "லொகேஷன்",
    "destination": "டெஸ்டினேஷன்",
    "payment": "பேமெண்ட்",
    "cash": "கேஷ்",
    "online": "ஆன்லைன்",
    "waiting": "வெயிட்டிங்",
    "call": "கால்",
    "number": "நம்பர்",
    "vehicle": "வெஹிக்கிள்",
    "ten": "டென்",
    "twenty": "ட்வென்டி",
    "thirty": "தர்ட்டி",
    "forty": "போர்ட்டி",
    "fifty": "பிப்டி",
    "sixty": "சிக்ஸ்டி",
    "seventy": "செவன்டி",
    "eighty": "எயிட்டி",
    "ninety": "நைன்டி",
    "the": "த",
    "and": "அண்ட்",
    "will": "வில்",
    "not": "நாட்",
    "now": "நவ்",
    "from": "ப்ரம்",
    "to": "டு",
    "at": "அட்",
    "in": "இன்",
    "on": "ஆன்",
    "unga": "உங்க",
    "enna": "என்ன",
    "rajesh": "ராஜேஷ்",
    "kumar": "குமார்",
    "chennai": "சென்னை",
    "central": "சென்ட்ரல்",
    "airport": "ஏர்போர்ட்",
    "station": "ஸ்டேஷன்",
}

LATIN_RE = re.compile(r"[a-zA-Z]+")

LETTER_NAMES_TA = {
    "a": "ஏ", "b": "பீ", "c": "சீ", "d": "டீ", "e": "ஈ",
    "f": "எஃப்", "g": "ஜீ", "h": "எச்", "i": "ஐ", "j": "ஜே",
    "k": "கே", "l": "எல்", "m": "எம்", "n": "என்", "o": "ஓ",
    "p": "பீ", "q": "க்யூ", "r": "ஆர்", "s": "எஸ்", "t": "டீ",
    "u": "யூ", "v": "வீ", "w": "டபிள்யூ", "x": "எக்ஸ்",
    "y": "வை", "z": "ஜெட்",
}


def transliterate_word(word: str) -> str:
    """Transliterate a single English word to Tamil script."""
    lower = word.lower()
    # Single letter → spell out the letter name
    if len(lower) == 1 and lower in LETTER_NAMES_TA:
        return LETTER_NAMES_TA[lower]
    if lower in COMMON_WORDS:
        return COMMON_WORDS[lower]
    return _phonetic_transliterate(lower)


def _phonetic_transliterate(word: str) -> str:
    """Basic phonetic transliteration for unknown words."""
    result = []
    i = 0
    while i < len(word):
        # Try two-char consonant clusters first
        if i + 1 < len(word):
            pair = word[i:i+2]
            if pair in CONSONANT_MAP:
                # Check if followed by a vowel
                vowel = _get_vowel(word, i + 2)
                if vowel:
                    base = CONSONANT_MAP[pair].rstrip("்")
                    result.append(base + VOWEL_SIGN_MAP.get(vowel, ""))
                    i += 2 + len(vowel)
                else:
                    result.append(CONSONANT_MAP[pair])
                    i += 2
                continue

        ch = word[i]
        if ch in CONSONANT_MAP:
            vowel = _get_vowel(word, i + 1)
            if vowel:
                base = CONSONANT_MAP[ch].rstrip("்")
                result.append(base + VOWEL_SIGN_MAP.get(vowel, ""))
                i += 1 + len(vowel)
            else:
                result.append(CONSONANT_MAP[ch])
                i += 1
        elif ch in VOWEL_MAP:
            vowel = _get_vowel(word, i)
            if vowel:
                result.append(VOWEL_MAP.get(vowel, VOWEL_MAP.get(ch, "")))
                i += len(vowel)
            else:
                i += 1
        else:
            i += 1

    return "".join(result)


def _get_vowel(word: str, pos: int) -> str | None:
    """Try to match a vowel at the given position (longest match first)."""
    if pos >= len(word):
        return None
    for length in (2, 1):
        if pos + length <= len(word):
            candidate = word[pos:pos+length]
            if candidate in VOWEL_MAP or candidate in VOWEL_SIGN_MAP:
                return candidate
    return None


def transliterate_english_segments(text: str) -> str:
    """Replace English words in text with Tamil transliterations.

    Tamil script portions are left unchanged. Only Latin-alphabet segments
    are transliterated. Special tokens like <cs> are stripped.
    """
    # Remove <cs> boundary tokens (they're routing metadata, not speech)
    text = re.sub(r"<cs>", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    def repl(m):
        word = m.group(0)
        return transliterate_word(word)

    return LATIN_RE.sub(repl, text)
