"""Tests for vectorized PCM16 WAV loading."""

from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from asr.audio_io import read_wav_mono_array, read_wav_mono_float


class AudioIOTests(unittest.TestCase):
    def test_vectorized_stereo_loader_preserves_public_list_api(self) -> None:
        stereo = np.asarray(
            [[-32768, 32767], [0, 16384], [8192, -8192]],
            dtype="<i2",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stereo.wav"
            with wave.open(str(path), "wb") as writer:
                writer.setnchannels(2)
                writer.setsampwidth(2)
                writer.setframerate(16000)
                writer.writeframes(stereo.tobytes())
            array = read_wav_mono_array(path)
            values = read_wav_mono_float(path)
        expected = stereo.astype(np.float32).mean(axis=1) / 32768.0
        self.assertEqual(array.dtype, np.float32)
        self.assertTrue(array.flags.c_contiguous)
        np.testing.assert_allclose(array, expected, atol=1e-7)
        np.testing.assert_allclose(values, expected, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
