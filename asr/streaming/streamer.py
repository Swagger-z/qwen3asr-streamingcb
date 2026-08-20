"""Streaming session runtime.

TODO: implement step/flush/reset orchestration for streaming ASR.
"""


class StreamingSession:
    """Placeholder streaming session.

    TODO: wire chunk scheduler, feature buffer, model forward, decoder, and commit logic.
    """

    def step(self, audio_chunk):
        """Process one incoming chunk.

        Args:
            audio_chunk: Waveform chunk-like object.

        Returns:
            Placeholder partial result object.
        """
        raise NotImplementedError('TODO: implement StreamingSession.step')

    def flush(self):
        """Finalize pending streaming outputs.

        Returns:
            Placeholder final result object.
        """
        raise NotImplementedError('TODO: implement StreamingSession.flush')

    def reset(self):
        """Reset all streaming state for a new utterance."""
        raise NotImplementedError('TODO: implement StreamingSession.reset')
