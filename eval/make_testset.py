"""Tanglish / transport-domain evaluation test set generator.

Builds a reproducible ~200-utterance test set covering:
  - Tamil Unicode + Latin English code-mix
  - Romanized Tamil + English
  - Proper-noun mixing
  - Structured entities (OTP, booking ID, phone, time, price, distance)

Deterministic given the seed; write once to eval/testsets/ and freeze.
"""

import argparse
import csv
import random
from pathlib import Path

TEMPLATES = [
    # (template, lang_tag)
    ("உங்கள் pickup location எங்கே?", "ta_en_mix"),
    ("Unga pickup location enga?", "ta_roman_en"),
    ("Your cab will arrive in {minutes} minutes.", "en"),
    ("உங்கள் கேப் {minutes} நிமிடத்தில் வரும்.", "ta"),
    ("Your OTP is {otp}. Please share it with the driver.", "en"),
    ("உங்கள் OTP {otp}, ஓட்டுநரிடம் சொல்லுங்கள்.", "ta"),
    ("Unga OTP {otp}, driver kitta sollunga.", "ta_roman"),
    ("Your booking ID is {booking_id}.", "en"),
    ("Booking ID {booking_id}, சரியா?", "ta_en_mix"),
    ("உங்கள் ride {place}-ல இருக்கா? {minutes} நிமிடத்தில் வந்துடுவான்.", "ta_en_mix"),
    ("Driver will reach {place} by {time_12}.", "en"),
    ("கட்டணம் Rs. {price}, தூரம் {distance} km.", "ta"),
    ("The fare is Rs. {price} for {distance} km.", "en"),
    ("Fare Rs. {price} aa, ok va?", "ta_roman"),
    ("Please confirm your phone number {phone}.", "en"),
    ("உங்கள் phone number {phone} தானே?", "ta_en_mix"),
    ("Cancel pannalama? Cancellation fee Rs. {fee} irukkum.", "ta_roman"),
    ("உங்கள் {vehicle_type} வந்துடுச்சு, {place} gate-ல காத்திருங்க.", "ta_en_mix"),
]

PLACES = ["Chennai Central", "Tambaram", "Adyar", "Velachery", "OMR",
          "Koyambedu", "Mylapore", "Guindy", "Egmore", "Sholinganallur"]
VEHICLE_TYPES = ["auto", "sedan", "SUV", "bike"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/testsets/tanglish_transport_200.tsv")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = []
    for i in range(args.n):
        tpl, tag = TEMPLATES[i % len(TEMPLATES)]
        text = tpl.format(
            minutes=rng.choice([2, 3, 5, 8, 10, 12, 15]),
            otp=f"{rng.randint(0, 9999):04d}",
            booking_id=f"TN{rng.randint(10, 99)}AB{rng.randint(1000, 9999)}",
            place=rng.choice(PLACES),
            time_12=f"{rng.randint(1, 12)}:{rng.choice(['00', '15', '30', '45'])} PM",
            price=rng.choice([120, 150, 180, 210, 250, 320, 450]),
            distance=rng.choice(["2.5", "4", "6.5", "9", "12"]),
            phone=f"9{rng.randint(700000000, 899999999)}",
            fee=rng.choice([25, 50]),
            vehicle_type=rng.choice(VEHICLE_TYPES),
        )
        rows.append({"id": f"utt_{i:04d}", "text": text, "category": tag})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "text", "category"], delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    from collections import Counter

    print(f"{len(rows)} utterances -> {out}")
    for cat, n in Counter(r["category"] for r in rows).most_common():
        print(f"  {cat}: {n}")


if __name__ == "__main__":
    main()
