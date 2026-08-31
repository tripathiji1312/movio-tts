---
datasets:
- ai4bharat/indicvoices_r
- ai4bharat/Rasa
language:
- as
- bn
- gu
- mr
- hi
- kn
- ml
- or
- pa
- ta
- te
pipeline_tag: text-to-speech
license: mit
---
# **IndicF5: High-Quality Text-to-Speech for Indian Languages**


We release **IndicF5**, a **near-human polyglot** **Text-to-Speech (TTS)** model trained on **1417 hours** of high-quality speech from **[Rasa](https://huggingface.co/datasets/ai4bharat/Rasa), [IndicTTS](https://www.iitm.ac.in/donlab/indictts/database), [LIMMITS](https://sites.google.com/view/limmits24/), and [IndicVoices-R](https://huggingface.co/datasets/ai4bharat/indicvoices_r)**.  

IndicF5 supports **11 Indian languages**:  
**Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu.**  

---

## 🚀 Installation
```bash
conda create -n indicf5 python=3.10 -y
conda activate indicf5
pip install git+https://github.com/ai4bharat/IndicF5.git
```


## 🎙 Usage

To generate speech, you need to provide **three inputs**:
1. **Text to synthesize** – The content you want the model to speak.
2. **A reference prompt audio** – An example speech clip that guides the model’s prosody and speaker characteristics.
3. **Text spoken in the reference prompt audio** – The transcript of the reference prompt audio.


```python
from transformers import AutoModel
import numpy as np
import soundfile as sf

# Load IndicF5 from Hugging Face
repo_id = "ai4bharat/IndicF5"
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)

# Generate speech
audio = model(
    "नमस्ते! संगीत की तरह जीवन भी खूबसूरत होता है, बस इसे सही ताल में जीना आना चाहिए.",
    ref_audio_path="prompts/PAN_F_HAPPY_00001.wav",
    ref_text="ਭਹੰਪੀ ਵਿੱਚ ਸਮਾਰਕਾਂ ਦੇ ਭਵਨ ਨਿਰਮਾਣ ਕਲਾ ਦੇ ਵੇਰਵੇ ਗੁੰਝਲਦਾਰ ਅਤੇ ਹੈਰਾਨ ਕਰਨ ਵਾਲੇ ਹਨ, ਜੋ ਮੈਨੂੰ ਖੁਸ਼ ਕਰਦੇ  ਹਨ।"
)

# Normalize and save output
if audio.dtype == np.int16:
    audio = audio.astype(np.float32) / 32768.0
sf.write("namaste.wav", np.array(audio, dtype=np.float32), samplerate=24000)
print("Audio saved succesfully.")
```

You can find example prompt audios used [here](https://huggingface.co/ai4bharat/IndicF5/tree/main/prompts).

## Terms of Use
By using this model, you agree to only clone voices for which you have explicit permission. Unauthorized voice cloning is strictly prohibited. Any misuse of this model is the responsibility of the user.

## References

We would like to extend our gratitude to the authors of  **[F5-TTS](https://github.com/SWivid/F5-TTS)** for their invaluable contributions and inspiration to this work. Their efforts have played a crucial role in advancing  the field of text-to-speech synthesis.


## 📖 Citation
If you use **IndicF5** in your research or projects, please consider citing it:

### 🔹 BibTeX
```bibtex
@misc{AI4Bharat_IndicF5_2025,
  author       = {Praveen S V and Srija Anand and Soma Siddhartha and Mitesh M. Khapra},
  title        = {IndicF5: High-Quality Text-to-Speech for Indian Languages},
  year         = {2025},
  url          = {https://github.com/AI4Bharat/IndicF5},
}