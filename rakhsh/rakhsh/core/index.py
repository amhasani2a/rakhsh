"""Retrieval back-ends: inverted index (sparse), flat dense index, BM25 baseline, RRF fusion."""
from __future__ import annotations
import math
from collections import Counter, defaultdict

import numpy as np

from .text import content_tokens


def _topk(scores: np.ndarray, k: int, positive_only: bool = True) -> list[tuple[int, float]]:
    k = min(k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [(int(i), float(scores[i])) for i in idx if scores[i] > 0 or not positive_only]


class InvertedIndex:
    """Term -> postings. Query cost scales with the number of spikes, not with |V|."""

    def __init__(self, doc_term: np.ndarray):
        self.n_docs = doc_term.shape[0]
        self.postings: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for t in np.nonzero(doc_term.any(0))[0]:
            docs = np.nonzero(doc_term[:, t])[0]
            self.postings[int(t)] = (docs, doc_term[docs, t])

    @classmethod
    def from_rows(cls, rows: list[tuple[np.ndarray, np.ndarray]]) -> "InvertedIndex":
        self = cls.__new__(cls); self.n_docs = len(rows)
        tmp: dict[int, tuple[list, list]] = defaultdict(lambda: ([], []))
        for d, (idx, val) in enumerate(rows):
            for t, v in zip(idx.tolist(), val.tolist()):
                tmp[t][0].append(d); tmp[t][1].append(v)
        self.postings = {t: (np.array(a), np.array(b, np.float32)) for t, (a, b) in tmp.items()}
        return self

    def scores(self, q: np.ndarray) -> np.ndarray:
        scores = np.zeros(self.n_docs, np.float32)
        for t in np.nonzero(q)[0]:
            p = self.postings.get(int(t))
            if p is not None:
                scores[p[0]] += q[t] * p[1]
        return scores

    def search(self, q: np.ndarray, k: int) -> list[tuple[int, float]]:
        scores = np.zeros(self.n_docs, np.float32)
        for t in np.nonzero(q)[0]:
            p = self.postings.get(int(t))
            if p is not None:
                scores[p[0]] += q[t] * p[1]
        return _topk(scores, k)


class DenseIndex:
    def __init__(self, vectors: np.ndarray):
        self.vectors = vectors.astype(np.float32)

    def search(self, q: np.ndarray, k: int) -> list[tuple[int, float]]:
        return _topk(self.vectors @ q, k, positive_only=False)


class BM25:
    """Classic lexical baseline (the 'old committee member')."""

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.docs = [Counter(content_tokens(t)) for t in texts]
        self.len = np.array([sum(d.values()) for d in self.docs], float)
        self.avg = self.len.mean(); self.k1, self.b = k1, b
        df = Counter(t for d in self.docs for t in d); n = len(texts)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.inv = defaultdict(list)
        for i, d in enumerate(self.docs):
            for t in d:
                self.inv[t].append(i)

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        s = np.zeros(len(self.docs))
        for t in set(content_tokens(query)):
            for i in self.inv.get(t, ()):
                tf = self.docs[i][t]
                s[i] += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return _topk(s, k)


def rrf(rankings: list[list[tuple[int, float]]], k: int, c: int = 60) -> list[tuple[int, float]]:
    """Reciprocal-rank fusion – parameter-free, no re-ranker model needed."""
    s: dict[int, float] = defaultdict(float)
    for r in rankings:
        for rank, (d, _) in enumerate(r):
            s[d] += 1.0 / (c + rank + 1)
    return sorted(s.items(), key=lambda x: -x[1])[:k]


def score_fusion(rankings: list[list[tuple[int, float]]], weights: list[float], k: int) -> list[tuple[int, float]]:
    """Max-normalised weighted score fusion of several first-stage rankings."""
    s: dict[int, float] = defaultdict(float)
    for r, w in zip(rankings, weights):
        if not r:
            continue
        top = max(abs(x[1]) for x in r) or 1.0
        for d, v in r:
            s[d] += w * v / top
    return sorted(s.items(), key=lambda x: -x[1])[:k]
