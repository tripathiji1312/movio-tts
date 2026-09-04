import pytest
from movio.acoustic.chunking import chunk_text
from movio.textnorm.domain_rules import DomainRuleEngine


class TestSmartChunkingAndOTP:
    def test_otp_single_sentence_not_split(self):
        chunks = chunk_text("யுவர் ஓ டீ பீ இஸ் ஃபோர் எயிட் த்ரீ டூ.")
        assert len(chunks) == 1
        assert "ஃபோர் எயிட் த்ரீ டூ" in chunks[0]

    def test_otp_two_sentences_split_at_period(self):
        text = "யுவர் ஓ டீ பீ இஸ் ஃபோர் எயிட் த்ரீ டூ. ப்ளீஸ் ஷேர் இட் வித் த ட்ரைவர்."
        chunks = chunk_text(text)
        assert len(chunks) == 2
        assert "ஃபோர் எயிட் த்ரீ டூ." in chunks[0]
        assert "ப்ளீஸ் ஷேர் இட்" in chunks[1]

    def test_greeting_merged_with_otp_clause(self):
        text = "வணக்கம், உங்கள் ஓ டீ பீ நான்கு எட்டு மூன்று இரண்டு. ஓட்டுநரிடம் சொல்லுங்கள்."
        chunks = chunk_text(text)
        assert len(chunks) == 2
        # Greeting should not be a standalone 3-syllable fragment
        assert chunks[0].startswith("வணக்கம், உங்கள் ஓ டீ பீ")

    def test_spaced_otp_normalization(self):
        eng = DomainRuleEngine(language="en")
        norm = eng.normalize("the otp is 483    2")
        assert "four eight three two" in norm
        assert "forty" not in norm
        assert "hundred" not in norm

    def test_spaced_otp_normalization_ta(self):
        ta = DomainRuleEngine(language="ta")
        norm = ta.normalize("உங்கள் ஓடிபி 483   2")
        assert "நான்கு எட்டு மூன்று இரண்டு" in norm

    def test_digit_sequence_never_split_across_chunks(self):
        # Long sentence with an 8-digit phone number in the middle
        text = (
            "உங்கள் புதிய சவாரி உறுதியானது ஓட்டுநரின் தொலைபேசி எண் "
            "ஒன்பது எட்டு ஏழு ஆறு ஐந்து நான்கு மூன்று இரண்டு ஒன்று பூஜ்ஜியம் உடனடியாக தொடர்பு கொள்ளவும்."
        )
        chunks = chunk_text(text, min_syl=10, max_syl=24)
        # All original words preserved
        assert sum(len(c.split()) for c in chunks) == len(text.split())
