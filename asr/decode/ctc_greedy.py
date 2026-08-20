"""Greedy CTC decoder.

TODO: implement blank removal and repeat-collapse decoding.
"""


def decode_greedy(log_probs, logit_lens, state=None):
    """Decode with greedy CTC.

    Args:
        log_probs: Tensor-like object with shape [B, T, V].
        logit_lens: Length tensor-like object with shape [B].
        state: Optional incremental decoding state.

    Returns:
        Placeholder hypotheses and next state.
    """
    raise NotImplementedError('TODO: implement decode_greedy')
