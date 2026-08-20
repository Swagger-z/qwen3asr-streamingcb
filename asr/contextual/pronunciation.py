"""Optional Mandarin pinyin and English ARPAbet path generation."""

from __future__ import annotations

import itertools
import re
from typing import Iterable, Sequence

from .types import HotwordEntry


_ENGLISH_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


class PinyinArpabetProvider:
    """Generate language-tagged pronunciation paths.

    Mandarin characters become initial/final-with-tone pairs via ``pypinyin``;
    English words use all CMUdict pronunciations up to ``max_paths``. Missing
    dependencies fail explicitly so paper runs cannot silently fall back to
    character matching.
    """

    def __init__(self, max_paths: int = 8) -> None:
        try:
            import cmudict
            from pypinyin import Style, pinyin
        except ImportError as exc:
            raise ImportError("install pypinyin and cmudict for phoneme catalog construction") from exc
        self._cmu = cmudict.dict()
        self._style = Style
        self._pinyin = pinyin
        self.max_paths = max_paths

    def __call__(self, entry: HotwordEntry, text: str) -> Iterable[Sequence[str]]:
        """Return up to ``max_paths`` pronunciation variants."""

        explicit = entry.metadata.get("pronunciations")
        if explicit:
            if isinstance(explicit, str):
                explicit = [explicit]
            for path in explicit:
                yield tuple(path.split()) if isinstance(path, str) else tuple(str(item) for item in path)
            return

        units: list[list[tuple[str, ...]]] = []
        position = 0
        for match in _ENGLISH_WORD.finditer(text):
            if match.start() > position:
                units.extend(self._mandarin_units(text[position : match.start()]))
            word = match.group(0).lower()
            pronunciations = self._cmu.get(word)
            if not pronunciations:
                units.append([tuple(f"en:char:{char}" for char in word)])
            else:
                units.append([tuple(f"en:{phone}" for phone in pronunciation) for pronunciation in pronunciations])
            position = match.end()
        if position < len(text):
            units.extend(self._mandarin_units(text[position:]))
        if not units:
            return
        count = 0
        for combination in itertools.product(*units):
            yield tuple(symbol for group in combination for symbol in group)
            count += 1
            if count >= self.max_paths:
                break

    def _mandarin_units(self, text: str) -> list[list[tuple[str, ...]]]:
        result: list[list[tuple[str, ...]]] = []
        for char in text:
            if char.isspace() or char in "-_/":
                continue
            initials = self._pinyin(char, style=self._style.INITIALS, strict=False, heteronym=True)[0]
            finals = self._pinyin(char, style=self._style.FINALS_TONE3, strict=False, heteronym=True)[0]
            variants = []
            for initial, final in itertools.product(initials, finals):
                symbols = []
                if initial:
                    symbols.append(f"zh:i:{initial}")
                symbols.append(f"zh:f:{final}")
                variants.append(tuple(symbols))
            result.append(variants or [(f"zh:char:{char}",)])
        return result
