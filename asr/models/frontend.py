"""Audio frontend module.

TODO: implement waveform-to-feature extraction with explicit shape contracts.
"""


class AudioFrontend:
    """Placeholder audio frontend.

    TODO: implement feature extraction for offline and streaming paths.
    """

    def __call__(self, wav, wav_lens):
        """Run frontend on waveform batch.

        Args:
            wav: Waveform tensor-like object with shape [B, T_wav].
            wav_lens: Length tensor-like object with shape [B].

        Returns:
            Placeholder tuple `(feat, feat_lens)`.
        """
        raise NotImplementedError('TODO: implement AudioFrontend.__call__')
