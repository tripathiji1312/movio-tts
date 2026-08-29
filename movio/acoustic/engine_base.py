import abc
from typing import Iterator

import numpy as np


class AcousticEngine(abc.ABC):
    """Common interface for all TTS acoustic backends."""

    @property
    @abc.abstractmethod
    def sample_rate(self) -> int: ...

    @property
    @abc.abstractmethod
    def is_ready(self) -> bool: ...

    @abc.abstractmethod
    def synthesize(self, text: str) -> np.ndarray:
        """Full utterance synthesis. Returns float32 audio [-1, 1]."""
        ...

    def synthesize_stream(self, text: str, chunk_words: int = 6) -> Iterator[np.ndarray]:
        """Yield audio in chunks for streaming. Default: split text, synth each."""
        words = text.split()
        if not words:
            return
        chunks = []
        for i in range(0, len(words), chunk_words):
            chunk = " ".join(words[i : i + chunk_words])
            if chunk.strip():
                chunks.append(chunk)
        if not chunks:
            chunks = [text]
        for chunk in chunks:
            audio = self.synthesize(chunk)
            if audio is not None and len(audio) > 0:
                yield audio
