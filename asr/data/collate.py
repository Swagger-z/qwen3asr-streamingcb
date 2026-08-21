"""Batch collation utilities.

TODO: implement padding and length collation for waveform/text batches.
"""


def collate_batch(samples):
    """Collate dataset samples into a batch.

    Args:
        samples: Sequence of sample mappings.

    Returns:
        Placeholder batch mapping.
    """
    raise NotImplementedError('TODO: implement collate_batch')
