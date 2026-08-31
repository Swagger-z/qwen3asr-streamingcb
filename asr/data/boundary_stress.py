"""Construct controlled hotword/chunk-boundary WAV variants."""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .timed_entities import validate_boundary


@dataclass(frozen=True)
class BoundaryCondition:
    """Named placement condition for one keyword span."""

    name: str
    kind: str
    value: float


DEFAULT_CONDITIONS = (
    BoundaryCondition("center", "center", 0.5),
    BoundaryCondition("b-400", "before", 0.400),
    BoundaryCondition("b-200", "before", 0.200),
    BoundaryCondition("b-100", "before", 0.100),
    BoundaryCondition("cross-25", "cross", 0.25),
    BoundaryCondition("cross-50", "cross", 0.50),
    BoundaryCondition("cross-75", "cross", 0.75),
)


def silence_for_condition(
    word_start_sec: float,
    word_end_sec: float,
    chunk_sec: float,
    condition: BoundaryCondition,
) -> float:
    """Return the smallest non-negative leading-silence duration."""

    if not 0 <= word_start_sec < word_end_sec or chunk_sec <= 0:
        raise ValueError("invalid word span or chunk duration")
    duration = word_end_sec - word_start_sec
    if condition.kind == "center":
        landmark = (word_start_sec + word_end_sec) / 2
        phase = chunk_sec / 2
    elif condition.kind == "before":
        landmark = (word_start_sec + word_end_sec) / 2
        phase = chunk_sec - condition.value
    elif condition.kind == "cross":
        # ``value`` is the fraction of keyword duration after the boundary.
        landmark = word_start_sec + duration * (1 - condition.value)
        phase = 0.0
    else:
        raise ValueError(f"unknown boundary condition kind: {condition.kind}")
    cycle = 0
    while phase + cycle * chunk_sec < landmark:
        cycle += 1
    return phase + cycle * chunk_sec - landmark


def prepend_pcm16_silence(source: str | Path, target: str | Path, silence_sec: float) -> None:
    """Prepend silence to a mono/stereo PCM16 WAV without resampling."""

    with wave.open(str(source), "rb") as reader:
        params = reader.getparams()
        frames = reader.readframes(reader.getnframes())
    if params.sampwidth != 2 or params.comptype != "NONE":
        raise ValueError("boundary builder requires uncompressed PCM16 WAV")
    silence_frames = round(params.framerate * silence_sec)
    silence = b"\x00" * silence_frames * params.nchannels * params.sampwidth
    output = Path(target)
    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "wb") as writer:
        writer.setparams(params)
        writer.writeframes(silence + frames)


def build_variants(
    source: str | Path,
    output_dir: str | Path,
    word_start_sec: float,
    word_end_sec: float,
    chunk_sec: float,
    conditions: Iterable[BoundaryCondition] = DEFAULT_CONDITIONS,
) -> list[dict[str, float | str]]:
    """Create all controlled variants and return manifest records."""

    source_path = Path(source).resolve()
    records = []
    for condition in conditions:
        silence = silence_for_condition(word_start_sec, word_end_sec, chunk_sec, condition)
        # Actual WAV padding is quantized to PCM samples, not arbitrary floats.
        with wave.open(str(source_path), "rb") as reader:
            silence = round(silence * reader.getframerate()) / reader.getframerate()
        target = (Path(output_dir) / f"{source_path.stem}.{condition.name}.wav").resolve()
        records.append(
            {
                "source": str(target),
                "audio": str(target),
                "boundary_group": condition.name,
                "boundary_chunk_sec": chunk_sec,
                "leading_silence_sec": silence,
                "hotword_start_sec": word_start_sec + silence,
                "hotword_end_sec": word_end_sec + silence,
                "word_start_sec": word_start_sec + silence,
                "word_end_sec": word_end_sec + silence,
            }
        )
    for record in records:
        validate_boundary(record)
    for record in records:
        prepend_pcm16_silence(source_path, record["audio"], record["leading_silence_sec"])
    return records
