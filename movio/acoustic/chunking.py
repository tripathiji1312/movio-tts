import re

TA_CLAUSE_MARKS = ".,;:!?—…।\n"
CLAUSE_SPLIT_RE = re.compile(r"([^\.,;:!?।\n]+[\.,;:!?।\n]*)\s*")

TA_VOWEL_SIGNS = re.compile(r"[\u0BBE-\u0BCD\u0BD7]")
TA_VOWELS = re.compile(r"[\u0B85-\u0B94]")

DIGIT_WORDS = {
    "பூஜ்யம்", "ஒன்று", "இரண்டு", "மூன்று", "நான்கு", "ஐந்து", "ஆறு", "ஏழு", "எட்டு", "ஒன்பது",
    "ஸீரோ", "வன்", "டூ", "த்ரீ", "ஃபோர்", "ஃபைவ்", "சிக்ஸ்", "செவன்", "எயிட்", "நைன்",
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "பத்து", "நூறு", "ஆயிரம்", "ten", "hundred", "thousand", "rupees", "ரூபாய்",
    "இருபது", "முப்பது", "நாற்பது", "ஐம்பது", "twenty", "thirty", "forty", "fifty"
}

SPLIT_BEFORE = {
    "மற்றும்", "ஆனால்", "எனவே", "ஆகையால்", "பிறகு", "அப்புறம்", "இதை", "அதை",
    "and", "but", "so", "because", "then", "or", "while", "also", "please", "kindly"
}


PUNCT_CHARS = ".,;:!?—…।\n\"'()[]{}"


def is_digit_word(w: str) -> bool:
    clean = w.strip(PUNCT_CHARS).lower()
    return clean in DIGIT_WORDS or clean.isdigit()


def count_syllables_ta(word: str) -> int:
    n = len(TA_VOWELS.findall(word))
    n += len(TA_VOWEL_SIGNS.findall(re.sub(r"[\u0B85-\u0B94]", "", word)))
    return max(n, 1)


def count_syllables_en(word: str) -> int:
    groups = re.findall(r"[aeiouy]+", word.lower())
    return max(len(groups), 1)


def _word_syl(w: str) -> int:
    clean = w.strip(PUNCT_CHARS)
    if not clean:
        return 0
    return count_syllables_ta(clean) if re.search(r"[\u0B80-\u0BFF]", clean) else count_syllables_en(clean)


def _count_text_syl(words: list[str]) -> int:
    return sum(_word_syl(w) for w in words)


def split_at_clauses(text: str) -> list[str]:
    parts = [p.strip() for p in CLAUSE_SPLIT_RE.split(text) if p and p.strip()]
    merged: list[str] = []
    buf = ""
    for part in parts:
        buf = (buf + " " + part).strip()
        if re.search(rf"[{re.escape(TA_CLAUSE_MARKS)}]\s*$", part) or len(buf.split()) >= 12:
            merged.append(buf)
            buf = ""
    if buf:
        if merged and len(buf.split()) < 3 and not re.search(r"[\.!?।]\s*$", merged[-1]):
            merged[-1] = (merged[-1].rstrip(".,;:!?") + " " + buf).strip()
        else:
            merged.append(buf)
    return merged


def chunk_text(text: str, min_syl: int = 10, max_syl: int = 24) -> list[str]:
    """Smart prosody-aware chunker for natural streaming TTS.

    Principles:
    1. Complete sentences under max_syl are preserved intact so the neural
       model maintains natural intonation, pitch contour, and breath cadence.
    2. Short introductory clauses (< min_syl // 2 syllables, e.g. 'வணக்கம்,')
       are merged with subsequent text rather than producing choppy 1-word chunks.
    3. Digit sequences (OTPs, phone numbers) are strictly protected from
       being split mid-number.
    4. Overlong compound sentences (> max_syl) split only at clause punctuation
       or discourse conjunctions.
    """
    text = text.strip()
    if not text:
        return []

    # 1. Break into sentence units by terminal punctuation (. ! ? । \n)
    raw_sents = [s.strip() for s in re.split(r"([\.!\?।\n]+)", text) if s and s.strip()]
    sentences: list[str] = []
    buf = ""
    for s in raw_sents:
        if re.match(r"^[\.!\?।\n]+$", s):
            if buf:
                sentences.append((buf + s).strip())
                buf = ""
            elif sentences:
                sentences[-1] = (sentences[-1] + s).strip()
        else:
            if buf:
                sentences.append(buf.strip())
            buf = s
    if buf:
        sentences.append(buf.strip())

    chunks: list[str] = []
    for sent in sentences:
        words = sent.split()
        if not words:
            continue
        sent_syl = _count_text_syl(words)

        # Whole sentence fits comfortably in a single breath (<= max_syl)
        if sent_syl <= max_syl:
            # If the previous chunk is very small (< min_syl // 2), merge with it
            if chunks and _count_text_syl(chunks[-1].split()) < max(4, min_syl // 2):
                chunks[-1] = (chunks[-1] + " " + sent).strip()
            else:
                chunks.append(sent)
            continue

        # Overlong sentence: split at clause punctuation or conjunctions
        current: list[str] = []
        current_syl = 0

        for i, w in enumerate(words):
            syl = _word_syl(w)
            current.append(w)
            current_syl += syl

            is_clause_end = bool(re.search(r"[,;:—]\s*$", w))
            next_word = words[i + 1] if (i + 1 < len(words)) else ""
            is_split_before_next = next_word.lower().rstrip(".,;!?") in SPLIT_BEFORE
            near_hyphen = (w in ("-", "—", "–") or next_word in ("-", "—", "–") or w.endswith("-") or next_word.startswith("-"))
            in_digit_seq = (is_digit_word(w) and (is_digit_word(next_word) or next_word in ("-", "—", "–"))) or (w in ("-", "—", "–") and i > 0 and is_digit_word(words[i-1]))

            # Split when min_syl reached, at natural boundary, never mid-digit or near hyphen
            if current_syl >= min_syl and (is_clause_end or is_split_before_next or current_syl >= max_syl):
                if not in_digit_seq and not near_hyphen:
                    chunks.append(" ".join(current))
                    current = []
                    current_syl = 0

        if current:
            if chunks and current_syl < max(4, min_syl // 2):
                chunks[-1] = (chunks[-1] + " " + " ".join(current)).strip()
            else:
                chunks.append(" ".join(current))

    # Clean pass: merge any tiny leftovers
    merged: list[str] = []
    for c in chunks:
        if merged and _count_text_syl(c.split()) < max(4, min_syl // 2):
            merged[-1] = (merged[-1] + " " + c).strip()
        else:
            merged.append(c)

    return merged
