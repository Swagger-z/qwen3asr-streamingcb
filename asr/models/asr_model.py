"""ASR model composition module.

TODO: compose frontend, subsampling, encoder, and CTC head.
"""


class ASRModel:
    """Placeholder ASR model wrapper.

    TODO: implement offline and streaming forward contracts from docs/spec.md.
    """

    def forward_offline(self, batch):
        """Run offline forward pass.

        Args:
            batch: Input batch object.

        Returns:
            Placeholder dict with logits and lengths.
        """
        raise NotImplementedError('TODO: implement ASRModel.forward_offline')

    def forward_streaming(self, chunk_batch, state):
        """Run streaming forward pass.

        Args:
            chunk_batch: Streaming chunk batch object.
            state: Encoder streaming state object.

        Returns:
            Placeholder dict with logits, lengths, and updated state.
        """
        raise NotImplementedError('TODO: implement ASRModel.forward_streaming')
