"""Streaming cache state definitions.

TODO: define cache dataclasses and invariant checks.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class EncoderStreamState:
    """Placeholder encoder stream state.

    TODO: add per-layer K/V caches and timeline bookkeeping fields.
    """

    processed_enc_frames: int = 0
    last_chunk_id: int = -1
    note: Optional[str] = None
