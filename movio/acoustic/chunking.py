import re

TA_VOWEL_SIGNS = re.compile(r"[ா-்ௗ]")
TA_VOWELS = re.compile(r"[அ-ஔ]")

PUNCT_CHARS = ".,;:!?—…।\n\"'()[]{}"

DIGIT_WORDS = {
    "ஸீரோ", "வன்", "டூ", "திரீ", "ஃபோர்", "ஃபைவ்", "சிக்ஸ்", "செவன்",
    "எயிட்", "நைன்", "டென்", "இலெவன்", "டுவெல்வ்",
    "ஏ", "பீ", "சீ", "டீ", "ஈ", "எஃப்", "ஜீ", "எச்", "ஐ", "ஜே",
    "கே", "எல்", "எம்", "என்", "ஓ", "ஆர்", "எஸ்",
    "யூ", "வீ", "எக்ஸ்", "வை", "ஸெட்",
}


def count_syllables_ta(word: str) -> int:
    n = len(TA_VOWELS.findall(word))
    n += len(TA_VOWEL_SIGNS.findall(re.sub(r"[அ-ஔ]", "", word)))
    return max(n, 1)


def count_syllables_en(word: str) -> int:
    groups = re.findall(r"[aeiouy]+", word.lower())
    return max(len(groups), 1)


def _word_syl(w: str) -> int:
    clean = w.strip(PUNCT_CHARS)
    if not clean:
        return 0
    return count_syllables_ta(clean) if re.search(r"[஀-௿]", clean) else count_syllables_en(clean)


def _count_text_syl(words: list[str]) -> int:
    return sum(_word_syl(w) for w in words)


def _is_digit_segment(text: str) -> bool:
    words = text.split()
    if not words:
        return False
    digit_count = sum(1 for w in words if w.strip(PUNCT_CHARS) in DIGIT_WORDS)
    return digit_count >= len(words) * 0.5


def chunk_text(text: str, min_syl: int = 10, max_syl: int = 28) -> list[str]:
    """Clause-level chunker for natural IndicF5 speech.

    Splits at every clause boundary (comma, period, semicolon) and merges
    intelligently: digit segments get paired (not fully merged) for clarity,
    non-digit segments merge up to min_syl to avoid choppy tiny chunks.
    """
    text = text.strip()
    if not text:
        return []

    raw = re.split(r"(?<=[,;.!?।])\s+", text)
    raw = [s.strip() for s in raw if s.strip()]

    _SPLIT_WORDS = {
        "மற்றும்", "ஆனால்", "எனவே", "ஆகையால்", "பிறகு", "அப்புறம்",
        "and", "but", "so", "because", "then", "or", "also", "please",
    }
    segments: list[str] = []
    for seg in raw:
        words = seg.split()
        syl = _count_text_syl(words)
        if syl <= max_syl:
            segments.append(seg)
            continue
        buf: list[str] = []
        buf_syl = 0
        for w in words:
            ws = _word_syl(w)
            if buf_syl >= min_syl and w.lower().rstrip(PUNCT_CHARS) in _SPLIT_WORDS:
                segments.append(" ".join(buf))
                buf = [w]
                buf_syl = ws
            else:
                buf.append(w)
                buf_syl += ws
                if buf_syl >= max_syl:
                    segments.append(" ".join(buf))
                    buf = []
                    buf_syl = 0
        if buf:
            segments.append(" ".join(buf))

    # Merge pass: pair adjacent digit segments, merge tiny segments with neighbors.
    # Digit segments pair greedily as long as combined stays under max_syl.
    # Non-digit tiny segments merge into previous non-digit chunk.
    merged: list[str] = []
    i = 0
    while i < len(segments):
        seg = segments[i]
        syl = _count_text_syl(seg.split())
        is_digit = _is_digit_segment(seg)

        if is_digit and syl < min_syl:
            combined = seg
            combined_syl = syl
            # Stop merging at sentence boundaries (period/! /?)
            ends_sentence = bool(re.search(r"[.!?।]\s*$", seg))
            j = i + 1
            while j < len(segments) and not ends_sentence:
                next_syl = _count_text_syl(segments[j].split())
                if combined_syl + next_syl <= max_syl and _is_digit_segment(segments[j]):
                    combined = combined + " " + segments[j]
                    combined_syl += next_syl
                    ends_sentence = bool(re.search(r"[.!?।]\s*$", segments[j]))
                    j += 1
                else:
                    break
            merged.append(combined)
            i = j
        elif merged and syl < max(3, min_syl // 2) and not _is_digit_segment(merged[-1]):
            merged[-1] = merged[-1] + " " + seg
            i += 1
        else:
            merged.append(seg)
            i += 1

    return merged if merged else [text]
