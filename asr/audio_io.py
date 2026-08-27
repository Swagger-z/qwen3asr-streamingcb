"""Small PCM16 WAV helpers used by reproducible CLIs."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def read_wav_mono_array(path: str | Path, expected_sample_rate: int = 16000) -> np.ndarray:
    """Read an uncompressed PCM16 WAV into a contiguous mono ``float32`` array."""

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
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32)
    samples *= 1.0 / 32768.0
    if channels > 1:
        if samples.size % channels:
            raise ValueError("interleaved WAV sample count is not divisible by channel count")
        samples = samples.reshape(-1, channels).mean(axis=1, dtype=np.float32)
    return np.ascontiguousarray(samples)


def read_wav_mono_float(path: str | Path, expected_sample_rate: int = 16000) -> list[float]:
    """Read uncompressed PCM16 WAV and average channels to mono float32 range."""

    return read_wav_mono_array(path, expected_sample_rate).tolist()
