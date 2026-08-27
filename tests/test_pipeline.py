import asyncio

import numpy as np
import pytest

from movio.acoustic.chunking import chunk_text
from movio.cache.audio_cache import AudioCache, MemoryCache
from movio.textnorm.domain_rules import DomainRuleEngine, english_number, tamil_number
from movio.router.lid import HeuristicLID


class TestChunking:
    def test_clause_split(self):
        text = "உங்கள் OTP 4821. ஓட்டுநர் வருகிறார்!"
        chunks = chunk_text(text)
        assert len(chunks) >= 1
        assert all(c.strip() for c in chunks)

    def test_long_text_multiple_chunks(self):
        text = "இது மிகவும் நீண்ட ஒரு சொல்லாகும், இதில் பல சொற்கள் உள்ளன; எனவே பல துண்டுகளாகப் பிரிக்கப்படும். ஒவ்வொரு துண்டும் சரியான அளவு."
        chunks = chunk_text(text)
        assert len(chunks) >= 2

    def test_no_loss_of_words(self):
        text = "Your driver is arriving now"
        joined = " ".join(chunk_text(text))
        assert "driver" in joined and "arriving" in joined


class TestDomainRules:
    def test_tamil_digits_translated(self):
        eng = DomainRuleEngine(language="ta")
        out = eng.normalize("௧௦௦")
        assert "நூறு" in out and not any(ch.isdigit() for ch in out)

    def test_otp_digits(self):
        eng = DomainRuleEngine(language="en")
        assert eng.expand_otp("4821") == "four eight two one"

    def test_currency_en(self):
        eng = DomainRuleEngine(language="en")
        assert "rupees" in eng.normalize("Rs. 250")

    def test_english_number(self):
        assert english_number(42) == "forty-two"
        assert english_number(1000) == "one thousand"

    def test_booking_id_spelling(self):
        eng = DomainRuleEngine(language="en")
        out = eng.normalize("Your booking ID is TN45AB1234.")
        assert "T N four five A B one two three four" in out

    def test_phone_digitwise_en(self):
        eng = DomainRuleEngine(language="en")
        assert "nine eight seven six five four three two one zero" in eng.normalize(
            "9876543210"
        )

    def test_time_pm_not_duplicated(self):
        eng = DomainRuleEngine(language="en")
        assert eng.normalize("arrive at 7:30 PM.") == "arrive at seven thirty PM."
        assert eng.normalize("pickup at 9:00 AM") == "pickup at nine AM"

    def test_time_tamil(self):
        ta = DomainRuleEngine(language="ta")
        out = ta.normalize("கேப் 7:30 வரும்")
        assert "ஏழு" in out and "முப்பது" in out and "நிமிடம்" in out

    def test_date_slash_en(self):
        eng = DomainRuleEngine(language="en")
        out = eng.normalize("ride on 25/08/2026")
        assert "August twenty-five" in out and "twenty twenty-six" in out

    def test_distance_and_decimal(self):
        eng = DomainRuleEngine(language="en")
        assert "four point five kilometers" in eng.normalize("4.5 km away")

    def test_currency_ta(self):
        ta = DomainRuleEngine(language="ta")
        assert "ரூபாய்" in ta.normalize("Rs. 250")


class TestTamilNumbers:
    @pytest.mark.parametrize(
        "n,expected",
        [(5, "ஐந்து"), (10, "பத்து"), (21, None), (100, "நூறு")],
    )
    def test_basic(self, n, expected):
        if expected:
            assert tamil_number(n) == expected

    def test_tens_join(self):
        out = tamil_number(21)
        assert out == "இருபத்தொன்று"
        assert tamil_number(45) == "நாற்பத்ஐந்து"


class TestHeuristicLID:
    def setup_method(self):
        self.lid = HeuristicLID()

    def test_native_tamil(self):
        assert self.lid.classify("உங்கள்") == "ta_native"

    def test_romanized(self):
        assert self.lid.classify("unga") == "ta_roman"
        assert self.lid.classify("iruku") == "ta_roman"

    def test_english(self):
        assert self.lid.classify("pickup") == "en"

    def test_batch(self):
        labels = self.lid.classify_batch(["unga", "pickup", "எங்கே"])
        assert labels == ["ta_roman", "en", "ta_native"]


class TestCache:
    @pytest.mark.asyncio
    async def test_memory_roundtrip(self):
        cache = MemoryCache(ttl_seconds=60)
        await cache.set("k", b"v")
        assert await cache.get("k") == b"v"

    @pytest.mark.asyncio
    async def test_audio_cache_memory_backend(self):
        config = {
            "pipeline": {"enable_cache": True},
            "cache": {"backend": "memory", "max_audio_bytes_mb": 8},
        }
        ac = AudioCache(config)
        pcm = np.zeros(100, dtype="<i2").tobytes()
        await ac.set("hello world", "voice_a", 12, pcm)
        got = await ac.get("hello world", "voice_a", 12)
        assert got == pcm

    @pytest.mark.asyncio
    async def test_disabled(self):
        config = {"pipeline": {"enable_cache": False}, "cache": {}}
        ac = AudioCache(config)
        await ac.set("x", "y", 1, b"data")
        assert await ac.get("x", "y", 1) is None
