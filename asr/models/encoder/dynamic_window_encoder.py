"""Dynamic-window encoder module.

TODO: implement Qwen3-ASR-inspired dynamic attention-window encoder.
"""


class DynamicWindowEncoder:
    """Placeholder dynamic-window encoder.

    TODO: implement unified offline and streaming encoder behavior.
    """

    def forward_offline(self, x, enc_lens):
        """Run offline encoder forward.

        Args:
            x: Input tensor-like object with shape [B, T_enc, D].
            enc_lens: Length tensor-like object with shape [B].

        Returns:
            Placeholder tuple `(h, out_lens)`.
        """
        raise NotImplementedError('TODO: implement DynamicWindowEncoder.forward_offline')

    def forward_streaming(self, x_chunk, state):
        """Run streaming encoder forward.

        Args:
            x_chunk: Input tensor-like object with shape [1, T_chunk_enc, D].
            state: Streaming cache state.

        Returns:
            Placeholder tuple `(h_new, out_lens, next_state)`.
        """
        raise NotImplementedError('TODO: implement DynamicWindowEncoder.forward_streaming')
