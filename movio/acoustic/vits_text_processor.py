"""Text-to-token processor for VITS Tamil ONNX inference.

Exact vocab from samprabin/tamil_vits (56 tokens):
  0: pad (_)
  1-6: punctuations (.,!?; and space)
  7-54: Tamil characters (vowels, consonants, vowel signs, virama)
  55: blank (<BLNK>)

With add_blank=True, blank (id=55) is inserted between every character token.
"""

VOCAB = {
    "_": 0,
    ".": 1, ",": 2, "!": 3, "?": 4, ";": 5, " ": 6,
    "ஂ": 7, "ஃ": 8,
    "அ": 9, "ஆ": 10, "இ": 11, "ஈ": 12, "உ": 13, "ஊ": 14,
    "எ": 15, "ஏ": 16, "ஐ": 17, "ஒ": 18, "ஓ": 19, "ஔ": 20,
    "க": 21, "ங": 22, "ச": 23, "ஜ": 24, "ஞ": 25,
    "ட": 26, "ண": 27, "த": 28, "ந": 29, "ன": 30,
    "ப": 31, "ம": 32, "ய": 33, "ர": 34, "ற": 35,
    "ல": 36, "ள": 37, "ழ": 38, "வ": 39,
    "ஷ": 40, "ஸ": 41, "ஹ": 42,
    "ா": 43, "ி": 44, "ீ": 45, "ு": 46, "ூ": 47,
    "ெ": 48, "ே": 49, "ை": 50, "ொ": 51, "ோ": 52, "ௌ": 53,
    "்": 54,
}

BLANK_ID = 55


def text_to_sequence(text: str, config_path: str | None = None, add_blank: bool = True) -> list[int]:
    """Convert Tamil text to token IDs for VITS ONNX inference.

    Characters not in the vocab are silently skipped.
    With add_blank=True, inserts BLANK (id=55) between every token.
    """
    sequence = []
    for ch in text:
        if ch in VOCAB:
            sequence.append(VOCAB[ch])

    if not sequence:
        return [BLANK_ID]

    if add_blank:
        interspersed = [BLANK_ID] * (len(sequence) * 2 + 1)
        interspersed[1::2] = sequence
        sequence = interspersed

    return sequence


def get_vocab_size() -> int:
    return 56
