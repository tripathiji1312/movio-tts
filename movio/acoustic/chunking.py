import re

TA_CLAUSE_MARKS = "।,;:!?—…"
CLAUSE_SPLIT_RE = re.compile(r"([^\.,;:!?।]+[\.,;:!?।]*)\s*")

TA_VOWEL_SIGNS = re.compile(r"[\u0BBE-\u0BCD\u0BD7]")
TA_VOWELS = re.compile(r"[\u0B85-\u0B94]")


def count_syllables_ta(word: str) -> int:
    n = len(TA_VOWELS.findall(word))
    n += len(TA_VOWEL_SIGNS.findall(re.sub(r"[\u0B85-\u0B94]", "", word)))
    return max(n, 1)


def count_syllables_en(word: str) -> int:
    groups = re.findall(r"[aeiouy]+", word.lower())
    return max(len(groups), 1)


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
        if merged and len(buf.split()) < 3:
            merged[-1] = (merged[-1].rstrip(".,;:!?") + " " + buf).strip()
        else:
            merged.append(buf)
    return merged


def chunk_text(text: str, min_syl: int = 8, max_syl: int = 14) -> list[str]:
    """Split text into prosody-preserving chunks of ~8-14 syllables.

    Clause boundaries are respected first; over-long clauses are then
    split at word boundaries nearest the target syllable budget.
    """
    chunks: list[str] = []
    for clause in split_at_clauses(text):
        words = clause.split()
        if not words:
            continue
        current: list[str] = []
        current_syl = 0
        clause_syl = sum(
            count_syllables_ta(w) if re.search(r"[\u0B80-\u0BFF]", w) else count_syllables_en(w)
            for w in words
        )
        target = max(min_syl, clause_syl // max(1, round(clause_syl / max_syl)) or 1)
        for w in words:
            syl = (
                count_syllables_ta(w)
                if re.search(r"[\u0B80-\u0BFF]", w)
                else count_syllables_en(w)
            )
            current.append(w)
            current_syl += syl
            if current_syl >= target:
                chunks.append(" ".join(current))
                current, current_syl = [], 0
        if current:
            if chunks and current_syl < min_syl // 2:
                chunks[-1] = chunks[-1] + " " + " ".join(current)
            else:
                chunks.append(" ".join(current))
    return chunks
