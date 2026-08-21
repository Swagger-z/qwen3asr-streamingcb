"""Time-subsampling module.

TODO: implement configurable subsampling and projection.
"""


class Subsampling:
    """Placeholder subsampling module.

    TODO: implement time reduction from feature frames to encoder frames.
    """

    def __call__(self, feat, feat_lens):
        """Run subsampling.

        Args:
            feat: Feature tensor-like object with shape [B, T_feat, F].
            feat_lens: Length tensor-like object with shape [B].

        Returns:
            Placeholder tuple `(x, enc_lens)`.
        """
        raise NotImplementedError('TODO: implement Subsampling.__call__')
