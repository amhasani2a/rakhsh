"""Self-supervision: the brain invents its own training questions from any data.

No labels are needed – every chunk generates pseudo-queries of several "shapes"
(exact span, scattered keywords, damaged/paraphrase-like sentence, heading,
row-field lookup). The gate's labels are then derived from *outcomes*:
the cheapest level that actually retrieved the right chunk.
"""
from __future__ import annotations
import numpy as np

from ..ingest.chunk import Chunk
from .text import STOPWORDS, content_tokens, split_sentences, tokenize


class PseudoQueries:
    def __init__(self, chunks: list[Chunk], seed: int = 0):
        self.chunks, self.rng = chunks, np.random.default_rng(seed)

    def _span(self, toks):
        n = int(self.rng.integers(3, 8)); s = int(self.rng.integers(0, max(1, len(toks) - n)))
        return " ".join(toks[s:s + n])

    def _keywords(self, c: Chunk):
        toks = list(dict.fromkeys(content_tokens(c.text)))
        if not toks:
            return c.text[:40]
        k = min(len(toks), int(self.rng.integers(2, 5)))
        return " ".join(self.rng.choice(toks, k, replace=False))

    def _damaged(self, c: Chunk):
        sents = c.sentences or split_sentences(c.text)
        toks = tokenize(sents[int(self.rng.integers(len(sents)))]) if sents else tokenize(c.text)
        keep = [t for t in toks if self.rng.random() > 0.4 and (t not in STOPWORDS or self.rng.random() < 0.3)]
        if len(keep) > 3 and self.rng.random() < 0.5:
            i = int(self.rng.integers(len(keep) - 1)); keep[i], keep[i + 1] = keep[i + 1], keep[i]
        return " ".join(keep[:10]) or " ".join(toks[:5])

    def _row(self, c: Chunk):
        items = [(k, v) for k, v in (c.fields or {}).items() if str(v).strip()]
        if not items:
            return self._keywords(c)
        k = min(len(items), int(self.rng.integers(1, 4)))
        pick = [items[i] for i in self.rng.choice(len(items), k, replace=False)]
        return " ".join((f"{a} {b}" if self.rng.random() < 0.4 else str(b)) for a, b in pick)

    def one(self, i: int) -> str:
        c = self.chunks[i]
        if c.kind == "row":
            return self._row(c) if self.rng.random() < 0.7 else self._keywords(c)
        r = self.rng.random()
        if r < 0.3:
            return self._span(tokenize(c.text))
        if r < 0.6:
            return self._keywords(c)
        if r < 0.9 or not c.title:
            return self._damaged(c)
        return f"{c.title} {self._keywords(c)}"

    def sample(self, n: int) -> tuple[list[str], np.ndarray]:
        idx = self.rng.integers(len(self.chunks), size=n)
        return [self.one(int(i)) for i in idx], idx
