"""Training losses.

TODO: implement CTC loss wrapper with explicit tensor shape checks.
"""


def compute_ctc_loss(logits, logit_lens, targets, target_lens, blank_id):
    """Compute CTC loss from model outputs and targets.

    Args:
        logits: Tensor-like object with shape [B, T, V].
        logit_lens: Length tensor-like object with shape [B].
        targets: Concatenated target token ids.
        target_lens: Target length tensor-like object with shape [B].
        blank_id: CTC blank token id.

    Returns:
        Placeholder scalar loss.
    """
    raise NotImplementedError('TODO: implement compute_ctc_loss')
