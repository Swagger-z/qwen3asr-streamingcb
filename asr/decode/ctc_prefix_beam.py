"""Prefix beam search CTC decoder.

TODO: implement LM-free prefix beam search decoding.
"""


def decode_prefix_beam(log_probs, logit_lens, state=None):
    """Decode with prefix beam search.

    Args:
        log_probs: Tensor-like object with shape [B, T, V].
        logit_lens: Length tensor-like object with shape [B].
        state: Optional incremental decoding state.

    Returns:
        Placeholder hypotheses and next state.
    """
    raise NotImplementedError('TODO: implement decode_prefix_beam')
