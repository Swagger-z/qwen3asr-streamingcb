"""Tokenizer abstraction.

TODO: implement character-level tokenizer baseline and pluggable interfaces.
"""


class Tokenizer:
    """Placeholder tokenizer interface.

    TODO: add encode/decode and vocab serialization logic.
    """

    @property
    def blank_id(self):
        """Return CTC blank token id."""
        raise NotImplementedError('TODO: implement Tokenizer.blank_id')

    @property
    def vocab_size(self):
        """Return vocabulary size including blank token."""
        raise NotImplementedError('TODO: implement Tokenizer.vocab_size')

    def encode(self, text):
        """Encode text into token ids."""
        raise NotImplementedError('TODO: implement Tokenizer.encode')

    def decode(self, token_ids):
        """Decode token ids into text."""
        raise NotImplementedError('TODO: implement Tokenizer.decode')
