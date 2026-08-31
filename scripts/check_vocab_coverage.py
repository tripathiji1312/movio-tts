"""Check which Tamil chars are already in the F5-TTS base vocab."""
import sys
from pathlib import Path
import subprocess

sys.path.insert(0, str(Path(__file__).parent.parent))

# Find vocab.txt from pip-installed f5-tts
result = subprocess.run(
    ["find", str(Path(sys.executable).parent.parent), "-name", "vocab.txt", "-path", "*/f5_tts/*"],
    capture_output=True, text=True
)
candidates = [p for p in result.stdout.strip().splitlines() if "examples" in p or "infer" in p]
print("Found vocab candidates:", candidates)

if not candidates:
    # Try site-packages directly
    result2 = subprocess.run(
        ["find", str(Path(sys.executable).parent.parent / "lib"), "-name", "vocab.txt"],
        capture_output=True, text=True
    )
    candidates = result2.stdout.strip().splitlines()
    print("All vocab.txt files found:", candidates)
    sys.exit(1)

base_vocab_path = candidates[0]
base_vocab = [l.rstrip("\n") for l in open(base_vocab_path, encoding="utf-8")]
base_set = set(base_vocab)

print(f"\nF5-TTS base vocab: {base_vocab_path}")
print(f"Base vocab size: {len(base_vocab)}")

# Tamil Unicode block U+0B80–U+0BFF
tamil_in_base = [c for c in base_vocab if '஀' <= c <= '௿']
print(f"\nTamil chars already in base vocab: {len(tamil_in_base)}")
if tamil_in_base:
    print("".join(tamil_in_base))
else:
    print("NONE — base vocab has zero Tamil characters")

ft_vocab = [l.rstrip("\n") for l in open("models/indicf5_tanglish/vocab.txt", encoding="utf-8")]
new_chars = [c for c in ft_vocab if c not in base_set]
new_tamil = [c for c in new_chars if '஀' <= c <= '௿']

print(f"\nChars we added during training: {len(new_chars)}")
print(f"Of those, Tamil Unicode: {len(new_tamil)}")

print(f"\n{'='*50}")
if not new_tamil:
    print("No new Tamil chars — retrain with base vocab only, zero new embeddings.")
else:
    print(f"{len(new_tamil)} Tamil chars were added as NEW embeddings (randomly initialized).")
    print("These random embeddings explain the Malayalam/Hindi output.")
    print("\nMissing Tamil chars (codepoint | char | name):")
    import unicodedata
    for c in sorted(new_tamil, key=ord):
        try:
            name = unicodedata.name(c)
        except ValueError:
            name = "unknown"
        print(f"  U+{ord(c):04X}  {c}  {name}")
