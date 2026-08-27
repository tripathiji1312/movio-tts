"""03 — Build the F5-style training dataset.

Converts the manifest from 02 into the format expected by the trainer:
each training item pairs (text, ref_audio, ref_text, target_audio) —
exactly IndicF5's zero-shot conditioning — with a sampled reference clip
per epoch step. Writes an Arrow dataset + a flat metadata.csv for the
F5-TTS finetuning path.
"""

import argparse
import csv
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/kaggle/working/data")
    ap.add_argument("--sample-rate", type=int, default=24000)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    train_csv = data_dir / "train.csv"
    ref_csv = data_dir / "ref_pool.csv"
    if not train_csv.exists():
        raise SystemExit("Run 02_prepare_corpus.py first.")

    import pandas as pd

    train = pd.read_csv(train_csv)
    refs = pd.read_csv(ref_csv).to_dict("records") if ref_csv.exists() else []

    # Flat metadata.csv for F5-TTS finetune path: audio|text
    meta_path = data_dir / "metadata.csv"
    with open(meta_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="|")
        for _, row in train.iterrows():
            w.writerow([Path(row["audio"]).name, row["text"]])

    rng = random.Random(42)

    def add_ref(row):
        if refs:
            r = rng.choice(refs)
            row["ref_audio"] = r["audio"]
            row["ref_text"] = r["text"]
        else:
            row["ref_audio"] = ""
            row["ref_text"] = ""
        return row

    from datasets import Audio, Dataset, Value

    ds = Dataset.from_pandas(train.apply(add_ref, axis=1))
    ds = ds.cast_column("audio", Audio(sampling_rate=args.sample_rate))
    if refs:
        ds = ds.cast_column("ref_audio", Audio(sampling_rate=args.sample_rate))
    ds.save_to_disk(str(data_dir / "hf_dataset"))

    print(f"Arrow dataset: {len(ds)} items -> {data_dir / 'hf_dataset'}")
    print(f"metadata.csv -> {meta_path}")


if __name__ == "__main__":
    main()
