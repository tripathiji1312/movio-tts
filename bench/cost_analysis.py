"""Cost analysis — problem-statement deliverable 6.

Combines measured throughput (bench/results.json) with an infrastructure
price profile to produce the hardware -> concurrency -> throughput ->
latency -> cost/minute relationship at 1, 5, 10, 15, 20 concurrent.

Usage:
  python bench/cost_analysis.py --results bench/results.json \
      --profile a100 --out bench/cost.json
"""

import argparse
import json
from pathlib import Path

PROFILES = {
    # $/hour on-demand list prices; adjust to your cloud contract.
    "a100": {
        "label": "1x A100 40GB + 8 vCPU pool",
        "usd_per_hour": 3.00,
        "note": "Blueprint §3.7 footprint: 3 IndicF5 replicas + CPU LID/xlit + Redis",
    },
    "l4": {
        "label": "2x L4 24GB + 8 vCPU pool",
        "usd_per_hour": 1.60,
        "note": "Overflow tier from blueprint §3.7",
    },
    "t4": {
        "label": "2x T4 16GB + 8 vCPU pool (budget)",
        "usd_per_hour": 0.90,
        "note": "Lower quality ceiling under load; validate p99 first",
    },
    "custom": {
        "label": "Custom",
        "usd_per_hour": 0.0,
        "note": "Set --usd-per-hour",
    },
}

COMMERCIAL_API_RANGE = (0.015, 0.040)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="bench/results.json")
    ap.add_argument("--profile", default="a100", choices=list(PROFILES))
    ap.add_argument("--usd-per-hour", type=float, default=None)
    ap.add_argument("--cache-hit-frac", type=float, default=0.35,
                    help="fraction of production requests served from cache")
    ap.add_argument("--out", default="bench/cost.json")
    args = ap.parse_args()

    results = json.loads(Path(args.results).read_text())
    profile = PROFILES[args.profile]
    usd_h = args.usd_per_hour if args.usd_per_hour is not None else profile["usd_per_hour"]

    rows = []
    for key, data in sorted(results.items()):
        if not key.startswith("concurrency_"):
            continue
        conc = int(key.split("_")[1])
        amp_h = data.get("audio_min_per_hour_at_this_load") or 0
        if amp_h <= 0 or usd_h <= 0:
            continue
        cost_min = usd_h / amp_h
        effective_cost = cost_min * (1 - args.cache_hit_frac)
        rows.append({
            "concurrency": conc,
            "audio_min_per_hour": amp_h,
            "ttfa_p99_ms": data["ttfa_p99_ms"],
            "usd_per_audio_min_raw": round(cost_min, 5),
            f"usd_per_audio_min_with_{int(args.cache_hit_frac*100)}pct_cache": round(effective_cost, 5),
            "vs_commercial_x_cheaper": round(
                COMMERCIAL_API_RANGE[0] / max(cost_min, 1e-9), 1
            ),
        })

    report = {
        "profile": profile["label"],
        "usd_per_hour": usd_h,
        "commercial_api_range_usd_per_min": list(COMMERCIAL_API_RANGE),
        "rows": rows,
        "note": profile["note"],
    }
    Path(args.out).write_text(json.dumps(report, indent=2))

    cache_key = f"usd_per_audio_min_with_{int(args.cache_hit_frac*100)}pct_cache"
    print(f"{'conc':>5} {'audio-min/h':>12} {'TTFA p99':>10} {'$/min':>9} {'w/ cache':>9} {'vs API':>8}")
    for r in rows:
        print(
            f"{r['concurrency']:>5} {r['audio_min_per_hour']:>12} "
            f"{r['ttfa_p99_ms']:>8.0f}ms {r['usd_per_audio_min_raw']:>9.4f} "
            f"{r[cache_key]:>9.4f} {r['vs_commercial_x_cheaper']:>7.1f}x"
        )
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
