import io
import time
from contextlib import contextmanager

import numpy as np


@contextmanager
def stopwatch():
    start = time.perf_counter()
    elapsed = {}
    try:
        yield elapsed
    finally:
        elapsed["ms"] = (time.perf_counter() - start) * 1000.0


def float_to_pcm16(audio: np.ndarray) -> bytes:
    audio = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    return (audio * 32767.0).astype("<i2").tobytes()


def pcm16_to_float(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def crossfade(a: np.ndarray, b: np.ndarray, n_samples: int) -> np.ndarray:
    if n_samples <= 0 or len(a) < n_samples or len(b) < n_samples:
        return np.concatenate([a, b])
    ramp = np.linspace(0.0, 1.0, n_samples, dtype=np.float32)
    head = a[:-n_samples]
    mixed = a[-n_samples:] * (1.0 - ramp) + b[:n_samples] * ramp
    return np.concatenate([head, mixed, b[n_samples:]])


def ms_to_samples(ms: float, sample_rate: int) -> int:
    return int(round(ms * sample_rate / 1000.0))


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio.astype(np.float32)
    import librosa

    return librosa.resample(
        np.asarray(audio, dtype=np.float32), orig_sr=orig_sr, target_sr=target_sr
    )


def wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    import soundfile as sf

    sf.write(buf, audio, samplerate=sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()
