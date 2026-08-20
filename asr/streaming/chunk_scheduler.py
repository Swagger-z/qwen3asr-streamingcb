"""Chunk scheduling utilities.

TODO: implement waveform chunk packetization with absolute offsets.
"""

from dataclasses import dataclass


@dataclass
class ChunkPacket:
    """Placeholder chunk packet metadata."""

    chunk_id: int
    wav_start_sample: int
    wav_end_sample: int


class ChunkScheduler:
    """Placeholder chunk scheduler.

    TODO: implement streaming chunk indexing and boundary tracking.
    """

    def push(self, audio_chunk):
        """Accept a chunk and return chunk metadata.

        Args:
            audio_chunk: Waveform chunk-like object.

        Returns:
            Placeholder `ChunkPacket`.
        """
        raise NotImplementedError('TODO: implement ChunkScheduler.push')
