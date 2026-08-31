"""Inspect IndicF5 HF repo — find its actual tokenizer/vocab.

Run: HF_TOKEN=hf_xxx uv run python scripts/inspect_indicf5_repo.py
"""
from huggingface_hub import list_repo_files, hf_hub_download

print("=== Files in ai4bharat/IndicF5 ===")
files = list(list_repo_files("ai4bharat/IndicF5"))
for f in files:
    print(" ", f)

print("\n=== config.json ===")
p = hf_hub_download("ai4bharat/IndicF5", "config.json")
print(open(p).read())

print("\n=== Any vocab/tokenizer files? ===")
vocab_files = [f for f in files if any(x in f.lower() for x in ["vocab", "token", "char", "text"])]
for f in vocab_files:
    p = hf_hub_download("ai4bharat/IndicF5", f)
    content = open(p, encoding="utf-8", errors="replace").read()
    print(f"\n--- {f} ---")
    print(content[:2000])

print("\n=== Any .py files? ===")
py_files = [f for f in files if f.endswith(".py")]
for f in py_files:
    p = hf_hub_download("ai4bharat/IndicF5", f)
    content = open(p, encoding="utf-8", errors="replace").read()
    print(f"\n--- {f} ---")
    print(content[:3000])
