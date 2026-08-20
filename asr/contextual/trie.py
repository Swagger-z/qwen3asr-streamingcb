"""Memory-conscious token trie with Aho-Corasick failure links."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Hashable, Iterable, Sequence


@dataclass
class TrieNode:
    """Internal trie node."""

    children: dict[Hashable, int] = field(default_factory=dict)
    failure: int = 0
    terminal_ids: set[str] = field(default_factory=set)
    output_ids: set[str] = field(default_factory=set)
    depth: int = 0


class TokenTrie:
    """Trie for string, phoneme, or tokenizer-token paths.

    Descendant candidate IDs are resolved lazily and capped, avoiding the
    quadratic memory behavior of materializing a full descendant set at every
    node for 100k-entry catalogs.
    """

    def __init__(self) -> None:
        self.nodes: list[TrieNode] = [TrieNode()]
        self._min_path_lengths: dict[str, int] = {}
        self._candidate_cache: dict[tuple[int, int], tuple[str, ...]] = {}
        self._built = False

    def add(self, path: Sequence[Hashable], hotword_id: str) -> None:
        """Insert a non-empty symbol path for ``hotword_id``."""

        if not path:
            raise ValueError("trie paths must be non-empty")
        node_index = 0
        for symbol in path:
            next_index = self.nodes[node_index].children.get(symbol)
            if next_index is None:
                next_index = len(self.nodes)
                self.nodes[node_index].children[symbol] = next_index
                self.nodes.append(TrieNode(depth=self.nodes[node_index].depth + 1))
            node_index = next_index
        self.nodes[node_index].terminal_ids.add(hotword_id)
        old = self._min_path_lengths.get(hotword_id)
        self._min_path_lengths[hotword_id] = len(path) if old is None else min(old, len(path))
        self._candidate_cache.clear()
        self._built = False

    def build_failure_links(self) -> None:
        """Build Aho-Corasick failure and output links."""

        queue: deque[int] = deque()
        root = self.nodes[0]
        root.output_ids = set(root.terminal_ids)
        for child in root.children.values():
            self.nodes[child].failure = 0
            queue.append(child)

        while queue:
            current = queue.popleft()
            node = self.nodes[current]
            node.output_ids = node.terminal_ids | self.nodes[node.failure].output_ids
            for symbol, child in node.children.items():
                fallback = node.failure
                while fallback and symbol not in self.nodes[fallback].children:
                    fallback = self.nodes[fallback].failure
                self.nodes[child].failure = self.nodes[fallback].children.get(symbol, 0)
                queue.append(child)
        self._built = True

    def step(self, node_index: int, symbol: Hashable) -> int:
        """Advance one AC state, following failure links as needed."""

        if not self._built:
            self.build_failure_links()
        while node_index and symbol not in self.nodes[node_index].children:
            node_index = self.nodes[node_index].failure
        return self.nodes[node_index].children.get(symbol, 0)

    def child(self, node_index: int, symbol: Hashable) -> int | None:
        """Return a direct trie child without using failure links."""

        return self.nodes[node_index].children.get(symbol)

    def output_ids(self, node_index: int) -> tuple[str, ...]:
        """Return terminal IDs matched at an AC state."""

        if not self._built:
            self.build_failure_links()
        return tuple(sorted(self.nodes[node_index].output_ids))

    def candidate_ids(self, node_index: int, limit: int = 256) -> tuple[str, ...]:
        """Return up to ``limit`` terminal descendants of a prefix node."""

        key = (node_index, limit)
        cached = self._candidate_cache.get(key)
        if cached is not None:
            return cached
        found: list[str] = []
        seen: set[str] = set()
        stack = [node_index]
        while stack and len(found) < limit:
            current = stack.pop()
            for hotword_id in sorted(self.nodes[current].terminal_ids):
                if hotword_id not in seen:
                    seen.add(hotword_id)
                    found.append(hotword_id)
                    if len(found) >= limit:
                        break
            children = self.nodes[current].children
            stack.extend(reversed([children[key] for key in sorted(children, key=str)]))
        result = tuple(found)
        self._candidate_cache[key] = result
        return result

    def min_path_length(self, hotword_id: str) -> int:
        """Return the shortest inserted path length for an entry."""

        return self._min_path_lengths[hotword_id]

    def iter_paths(self) -> Iterable[tuple[tuple[Hashable, ...], str]]:
        """Yield every terminal path, primarily for tests and diagnostics."""

        stack: list[tuple[int, tuple[Hashable, ...]]] = [(0, ())]
        while stack:
            node_index, path = stack.pop()
            for hotword_id in sorted(self.nodes[node_index].terminal_ids):
                yield path, hotword_id
            for symbol, child in self.nodes[node_index].children.items():
                stack.append((child, path + (symbol,)))
