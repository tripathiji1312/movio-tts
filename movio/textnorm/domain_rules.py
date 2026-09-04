import json
import re
from pathlib import Path

TA_DIGITS = str.maketrans("௦௧௨௩௪௫௬௭௮௯", "0123456789")

TA_NUMBER_WORDS = {
    0: "பூஜ்யம்", 1: "ஒன்று", 2: "இரண்டு", 3: "மூன்று", 4: "நான்கு",
    5: "ஐந்து", 6: "ஆறு", 7: "ஏழு", 8: "எட்டு", 9: "ஒன்பது",
    10: "பத்து", 11: "பதினொன்று", 12: "பனிரண்டு", 13: "பதின்மூன்று",
    14: "பதினான்கு", 15: "பதினைந்து", 16: "பதினாறு", 17: "பதினேழு",
    18: "பதினெட்டு", 19: "பத்தொன்பது", 20: "இருபது", 30: "முப்பது",
    40: "நாற்பது", 50: "ஐம்பது", 60: "அறுபது", 70: "எழுபது",
    80: "எண்பது", 90: "தொண்ணூறு", 100: "நூறு", 1000: "ஆயிரம்",
}

EN_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
           "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
           "sixteen", "seventeen", "eighteen", "nineteen"]
EN_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
           "eighty", "ninety"]


def _digits_of(n: int) -> list[int]:
    return [int(d) for d in str(n)]


def tamil_number(n: int) -> str:
    if n < 0:
        return "மைனஸ் " + tamil_number(-n)
    if n == 0:
        return TA_NUMBER_WORDS[0]
    if n <= 19:
        return TA_NUMBER_WORDS[n]
    if n < 100:
        tens = (n // 10) * 10
        ones = n % 10
        if ones == 0:
            return TA_NUMBER_WORDS[n]
        return _tens_join(tens, ones)
    if n < 1000:
        hundreds = n // 100
        rest = n % 100
        hun = f"{tamil_number(hundreds)} நூறு" if hundreds > 1 else "நூறு"
        return f"{hun} {tamil_number(rest)}".strip() if rest else hun
    if n < 100000:
        thousands = n // 1000
        rest = n % 1000
        th = f"{tamil_number(thousands)} ஆயிரம்"
        return f"{th} {tamil_number(rest)}".strip() if rest else th
    digits = "".join(str(d) for d in _digits_of(n))
    return " ".join(TA_NUMBER_WORDS[int(d)] for d in digits)


def _tens_join(tens: int, ones: int) -> str:
    stem = {
        20: ("இருபத்", ""), 30: ("முப்பத்", ""), 40: ("நாற்பத்", ""),
        50: ("ஐம்பத்", ""), 60: ("அறுபத்", ""), 70: ("எழுபத்", ""),
        80: ("எண்பத்", ""), 90: ("தொண்ணூற்", ""),
    }[tens]
    one_words = {1: "தொன்று", 2: "இரண்டு", 3: "மூன்று", 4: "நான்கு", 5: "ஐந்து",
                 6: "ஆறு", 7: "ஏழு", 8: "எட்டு", 9: "ஒன்பது"}
    return stem[0] + one_words[ones]


def _digit_seq_tamil(n: int) -> str:
    return " ".join(TA_NUMBER_WORDS[int(d)] for d in str(n))


def english_number(n: int) -> str:
    if n < 0:
        return "minus " + english_number(-n)
    if n < 20:
        return EN_ONES[n]
    if n < 100:
        t, o = divmod(n, 10)
        return EN_TENS[t] + ("-" + EN_ONES[o] if o else "")
    if n < 1000:
        h, rest = divmod(n, 100)
        out = EN_ONES[h] + " hundred"
        if rest:
            out += " " + english_number(rest)
        return out
    if n < 1_000_000:
        th, rest = divmod(n, 1000)
        out = english_number(th) + " thousand"
        if rest:
            out += " " + english_number(rest)
        return out


DIGIT_RE = re.compile(r"\d+")
TIME_RE = re.compile(r"\b(\d{1,2}):(\d{2})(?:\s?(A\.?M\.?|P\.?M\.?))?\b", re.IGNORECASE)
OTP_RE = re.compile(r"\b(\d{4,8})\b")
PHONE_RE = re.compile(r"\b(?:\+?91[- ]?)?([6-9]\d{4})[- ]?(\d{5})\b")
VEHICLE_RE = re.compile(
    r"\b([A-Z]{2}[\s-]?\d{1,2}[\s-]?[A-Z]{1,3}[\s-]?(\d{4}))\b"
)
BOOKING_ID_RE = re.compile(r"\b([A-Z]{1,3}\d{6,12})\b")
DATE_SLASH_RE = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b")
DATE_TEXT_RE = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+"
    r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?(?:,?\s+(\d{4}))?\b",
    re.IGNORECASE,
)
DISTANCE_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s?(km|kms|kilometers?|metres?|meters?|m)\b", re.IGNORECASE)
MONTHS_EN = ["January", "February", "March", "April", "May", "June", "July",
             "August", "September", "October", "November", "December"]
MONTHS_TA = ["ஜனவரி", "பிப்ரவரி", "மார்ச்", "ஏப்ரல்", "மே", "ஜூன்",
             "ஜூலை", "ஆகஸ்ட்", "செப்டம்பர்", "அக்டோபர்", "நவம்பர்", "டிசம்பர்"]


class DomainRuleEngine:
    """Deterministic transport-domain verbalization rules.

    Handles OTP/booking-ID digit-wise reading, phone numbers, vehicle plates,
    time, dates, distances/units, currency, and Tamil native digits — the
    known IndicF5 number-pronunciation edge cases.
    """

    def __init__(self, language: str = "ta"):
        self.language = language

    def normalize(self, text: str) -> str:
        text = text.translate(TA_DIGITS)
        text = self._expand_booking_ids(text)
        text = self._expand_vehicle(text)
        text = self._expand_phone(text)
        text = self._expand_time(text)
        text = self._expand_dates(text)
        text = self._expand_distances(text)
        text = self._expand_currency(text)
        text = self._expand_numbers(text)
        return re.sub(r"\s+", " ", text).strip()

    def _lang_is_ta(self) -> bool:
        return self.language == "ta"

    def _number_fn(self):
        return tamil_number if self._lang_is_ta() else english_number

    def _spell_alnum(self, token: str) -> str:
        numfn = self._number_fn()
        parts = []
        for ch in token:
            if ch.isalpha():
                parts.append(ch.upper())
            elif ch.isdigit():
                parts.append(numfn(int(ch)))
        return " ".join(parts)

    def _expand_booking_ids(self, text: str) -> str:
        return BOOKING_ID_RE.sub(lambda m: self._spell_alnum(m.group(1)), text)

    def _expand_time(self, text: str) -> str:
        numfn = self._number_fn()
        h12 = lambda h: ((h - 1) % 12) + 1

        def repl(m):
            h, mi = int(m.group(1)), int(m.group(2))
            if not (0 <= h <= 23 and 0 <= mi <= 59):
                return m.group(0)
            suffix = (m.group(3) or "").upper()
            explicit_pm = suffix.startswith("P")
            explicit_am = suffix.startswith("A")
            if explicit_am or explicit_pm:
                period = "AM" if explicit_am else "PM"
                minute_part = f" {english_number(mi)}" if mi else ""
                return f"{english_number(h12(h))}{minute_part} {period}"
            if self._lang_is_ta():
                period = "காலை" if h < 12 else "மதியம்" if h < 16 else "மாலை" if h < 20 else "இரவு"
                minute_part = f" {numfn(mi)} நிமிடம்" if mi else ""
                # Check if the period word already precedes the time in the text
                start = m.start()
                prefix = text[max(0, start - 10):start]
                if period in prefix:
                    return f"{numfn(h12(h))} மணி{minute_part}"
                return f"{period} {numfn(h12(h))} மணி{minute_part}"
            minute_part = f" {numfn(mi)}" if mi else ""
            period = "AM" if h < 12 else "PM"
            return f"{numfn(h12(h))}{minute_part} {period}"

        return TIME_RE.sub(repl, text)

    def _year_words(self, y: int) -> str:
        numfn = self._number_fn()
        if 1000 <= y <= 2099:
            first, second = divmod(y, 100)
            return f"{numfn(first)} {numfn(second)}"
        return " ".join(numfn(int(c)) for c in str(y))

    def _expand_dates(self, text: str) -> str:
        numfn = self._number_fn()

        def slash_repl(m):
            d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if y < 100:
                y += 2000
            if not (1 <= d <= 31 and 1 <= mo <= 12):
                return m.group(0)
            month = MONTHS_TA[mo - 1] if self._lang_is_ta() else MONTHS_EN[mo - 1]
            year_words = self._year_words(y)
            return f"{month} {numfn(d)}, {year_words}"

        out = DATE_SLASH_RE.sub(slash_repl, text)

        def text_repl(m):
            d, mon_name, y = int(m.group(1)), m.group(2).capitalize(), m.group(3)
            idx = next((i for i, mn in enumerate(MONTHS_EN) if mn.startswith(mon_name)), None)
            if idx is None or not (1 <= d <= 31):
                return m.group(0)
            month = MONTHS_TA[idx] if self._lang_is_ta() else MONTHS_EN[idx]
            year_part = ""
            if y:
                year_words = self._year_words(int(y))
                year_part = f", {year_words}"
            return f"{month} {numfn(d)}{year_part}"

        return DATE_TEXT_RE.sub(text_repl, out)

    def _expand_distances(self, text: str) -> str:
        numfn = self._number_fn()
        units_en = {"km": "kilometers", "kms": "kilometers", "kilometer": "kilometers",
                    "kilometers": "kilometers", "m": "meters", "meter": "meters",
                    "metre": "meters", "metres": "meters"}
        units_ta = {"km": "கிலோமீட்டர்", "kms": "கிலோமீட்டர்", "kilometer": "கிலோமீட்டர்",
                    "kilometers": "கிலோமீட்டர்", "m": "மீட்டர்", "meter": "மீட்டர்",
                    "metre": "மீட்டர்", "metres": "மீட்டர்"}
        table = units_ta if self._lang_is_ta() else units_en

        def repl(m):
            value_str, unit = m.group(1), m.group(2).lower()
            unit_word = table.get(unit, unit)
            if "." in value_str:
                whole, frac = value_str.split(".")
                point = "புள்ளி" if self._lang_is_ta() else "point"
                frac_words = " ".join(numfn(int(c)) for c in frac)
                return f"{numfn(int(whole))} {point} {frac_words} {unit_word}"
            return f"{numfn(int(value_str))} {unit_word}"

        return DISTANCE_RE.sub(repl, text)

    def _expand_phone(self, text: str) -> str:
        numfn = self._number_fn()

        def repl(m):
            digits = m.group(1) + m.group(2)
            return " ".join(numfn(int(d)) for d in digits)

        return PHONE_RE.sub(repl, text)

    def _expand_vehicle(self, text: str) -> str:
        def _spell_vehicle(token: str) -> str:
            parts = []
            for ch in token:
                if ch.isalpha():
                    parts.append(ch.upper())
                elif ch.isdigit():
                    parts.append(english_number(int(ch)))
            return " ".join(parts)

        return VEHICLE_RE.sub(lambda m: _spell_vehicle(re.sub(r"[\s-]", "", m.group(1))), text)

    def _expand_currency(self, text: str) -> str:
        numfn = self._number_fn()
        rupee_re = re.compile(r"(?:₹|Rs\.?|ரூ)\s?(\d+(?:,\d+)*)")

        def repl(m):
            amount = int(m.group(1).replace(",", ""))
            unit = "ரூபாய்" if self._lang_is_ta() else "rupees"
            return f"{numfn(amount)} {unit}"

        return rupee_re.sub(repl, text)

    def _expand_numbers(self, text: str) -> str:
        numfn = self._number_fn()

        # Pre-process OTP / PIN / passcode contexts with spaced digits (e.g. "otp is 483   2" or "OTP: 4 8 3 2")
        otp_context_re = re.compile(
            r"\b(otp(?:\s+is|\s*:|\s*-)?|pin(?:\s+is|\s*:|\s*-)?|code(?:\s+is|\s*:|\s*-)?|passcode(?:\s+is|\s*:|\s*-)?|"
            r"ஓடிபி(?:\s+எண்|\s*:|\s*-)?|கடவுச்சொல்(?:\s+எண்|\s*:|\s*-)?)\s+((?:\d\s*){3,8})\b",
            re.IGNORECASE,
        )

        def _otp_repl(m):
            prefix = m.group(1)
            raw_digits = re.sub(r"\D", "", m.group(2))
            return f"{prefix} " + " ".join(numfn(int(d)) for d in raw_digits)

        text = otp_context_re.sub(_otp_repl, text)

        def repl(m):
            digits = m.group(0)
            value = int(digits.lstrip("0") or "0")
            # OTP / ID style codes (4-8 digits) are always read digit-by-digit
            if 4 <= len(digits) <= 8:
                return " ".join(numfn(int(d)) for d in digits)
            return numfn(value)

        return DIGIT_RE.sub(repl, text)

    def expand_otp(self, otp: str) -> str:
        numfn = self._number_fn()
        return " ".join(numfn(int(d)) for d in otp if d.isdigit())

    def to_json(self) -> str:
        return json.dumps({"language": self.language})
