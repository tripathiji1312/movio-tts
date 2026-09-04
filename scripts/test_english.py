"""Test IndicF5 with English, Romanized Tanglish, and Mixed text.
Run in another terminal: HF_TOKEN=$HF_TOKEN uv run python scripts/test_english.py
Requires HF_TOKEN env (gated ai4bharat/IndicF5) — never hardcode tokens.
"""
import sys, os
if not os.getenv("HF_TOKEN"):
    raise SystemExit("HF_TOKEN env is required (gated ai4bharat/IndicF5)")

project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "third_party", "IndicF5"))
for k in list(sys.modules):
    if k.startswith("f5_tts"):
        del sys.modules[k]

from f5_tts.model.backbones.dit import DiT
from f5_tts.infer.utils_infer import load_model, load_vocoder, preprocess_ref_audio_text, infer_process
from huggingface_hub import hf_hub_download
from movio.router.en2ta import transliterate_english_to_tamil
import soundfile as sf

print("Loading model...")
ckpt_path = hf_hub_download("ai4bharat/IndicF5", "model.safetensors")
model = load_model(DiT, dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
    ckpt_path=ckpt_path, mel_spec_type="vocos", vocab_file="models/indicf5_tanglish/vocab.txt",
    ode_method="midpoint", device="cpu")
model.eval()
vocoder = load_vocoder(vocoder_name="vocos", is_local=False, device="cpu")

ref_audio, ref_text = preprocess_ref_audio_text(
    "config/voices/ta_female_neutral/ref.wav",
    "ஆனா நீங்க இப்போதான் மொத தடவையா இன்டர்நெட் யூஸ் பண்றீங்க அப்படின்னா இதை முழுசா கத்துக்க கொஞ்சம் நாள் ஆகும்.")

tests = [
    ("english", "Your OTP is 4832. Please share it with the driver."),
    ("tanglish_roman", "Unga booking confirm aayiduchu. Driver name Rajesh."),
    ("mixed", "காலை 7:30-க்கு pickup ready-யா இருங்க."),
    ("tamil", "உங்கள் கேப் ஐந்து நிமிடத்தில் வரும்."),
]

for name, text in tests:
    tamil_text = transliterate_english_to_tamil(text)
    print(f"\n--- {name}")
    print(f"  IN:  {text}")
    print(f"  OUT: {tamil_text}")
    audio, sr, _ = infer_process(ref_audio, ref_text, tamil_text,
        model, vocoder, device="cpu", nfe_step=24, cfg_strength=2.0, speed=1.0)
    path = f"/tmp/test_{name}.wav"
    sf.write(path, audio, 24000)
    print(f"  {len(audio)/24000:.2f}s -> {path}")

print("\nDone! Play with: aplay /tmp/test_english.wav etc.")
