"""Small PCM16 WAV helpers used by reproducible CLIs."""

from __future__ import annotations

import wave
from pathlib import Path


def read_wav_mono_float(path: str | Path, expected_sample_rate: int = 16000) -> list[float]:
    """Read uncompressed PCM16 WAV and average channels to mono float32 range."""

    with wave.open(str(path), "rb") as reader:
        channels = reader.getnchannels()
        sample_width = reader.getsampwidth()
        sample_rate = reader.getframerate()
        compression = reader.getcomptype()
        frames = reader.readframes(reader.getnframes())
    if sample_width != 2 or compression != "NONE":
        raise ValueError("only uncompressed PCM16 WAV is supported")
    if sample_rate != expected_sample_rate:
        raise ValueError(f"expected {expected_sample_rate} Hz, found {sample_rate} Hz")
    samples = [int.from_bytes(frames[index : index + 2], "little", signed=True) / 32768.0 for index in range(0, len(frames), 2)]
    if channels == 1:
        return samples
    return [sum(samples[index : index + channels]) / channels for index in range(0, len(samples), channels)]
