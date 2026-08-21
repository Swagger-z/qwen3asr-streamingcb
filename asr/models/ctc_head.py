"""CTC head module.

TODO: implement frame-wise projection to vocabulary logits.
"""


class CTCHead:
    """Placeholder CTC head.

    TODO: implement linear projection from encoder dimension D to vocab V.
    """

    def __call__(self, h):
        """Project encoder states to CTC logits.

        Args:
            h: Encoder hidden tensor-like object with shape [B, T, D].

        Returns:
            Placeholder logits tensor-like object with shape [B, T, V].
        """
        raise NotImplementedError('TODO: implement CTCHead.__call__')
