"""Multi-path sparse token-trie logit biasing."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from .catalog import HotwordCatalog, TokenizerLike


class SparseTokenBias:
    """Pure, request-local calculation of sparse next-token bonuses."""

    def __init__(
        self,
        catalog: HotwordCatalog,
        tokenizer: TokenizerLike,
        enabled_ids: Iterable[str] | None = None,
        base_bonus: float = 2.0,
        max_bonus: float = 4.0,
    ) -> None:
        self.catalog = catalog
        self.tokenizer = tokenizer
        self.enabled_ids = set(enabled_ids) if enabled_ids is not None else set(catalog.entries)
        self.base_bonus = float(base_bonus)
        self.max_bonus = float(max_bonus)
        self.trie = catalog.token_trie(tokenizer, self.enabled_ids)

    def update_catalog(self, catalog: HotwordCatalog, enabled_ids: Iterable[str]) -> None:
        """Rebuild token paths after a catalog or session-filter update."""

        self.catalog = catalog
        self.enabled_ids = set(enabled_ids)
        self.trie = catalog.token_trie(self.tokenizer, self.enabled_ids)

    def biases(
        self,
        output_token_ids: Sequence[int],
        confidence_by_id: Mapping[str, float],
    ) -> dict[int, float]:
        """Return bonuses only for direct children of the current trie state."""

        if self.base_bonus == 0 or not confidence_by_id:
            return {}
        node = 0
        for token_id in output_token_ids:
            node = self.trie.step(node, int(token_id))
        result: dict[int, float] = {}
        for child_token, child_node in self.trie.nodes[node].children.items():
            best = 0.0
            for hotword_id in self.trie.candidate_ids(child_node, limit=256):
                confidence = max(0.0, min(1.0, float(confidence_by_id.get(hotword_id, 0.0))))
                weighted = self.base_bonus * confidence * self.catalog.weight(hotword_id)
                best = max(best, weighted)
            if best > 0:
                result[int(child_token)] = min(self.max_bonus, best)
        return result

    def processor(self, confidence_by_id: Mapping[str, float]) -> "VLLMTokenTrieLogitsProcessor":
        """Create an isolated processor snapshot for one refresh request."""

        return VLLMTokenTrieLogitsProcessor(self, dict(confidence_by_id))


class VLLMTokenTrieLogitsProcessor:
    """Callable compatible with vLLM request-level processor signatures."""

    def __init__(self, bias: SparseTokenBias, confidence_by_id: Mapping[str, float]) -> None:
        self.bias = bias
        self.confidence_by_id = dict(confidence_by_id)

    def __call__(self, *args: Any) -> Any:
        """Apply sparse in-place bonuses and return the logits object."""

        if len(args) == 2:
            output_ids, logits = args
        elif len(args) == 3:
            _, output_ids, logits = args
        else:
            raise TypeError("expected (output_ids, logits) or (prompt_ids, output_ids, logits)")
        for token_id, bonus in self.bias.biases(output_ids, self.confidence_by_id).items():
            logits[token_id] = logits[token_id] + bonus
        return logits

    def is_argmax_invariant(self) -> bool:
        """Return false because biasing may change greedy output."""

        return False
