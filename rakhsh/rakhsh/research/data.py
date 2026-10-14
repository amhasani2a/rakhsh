"""Dataset loading and on-the-fly query augmentation for the three gate levels."""
from __future__ import annotations
import json
import pathlib
from dataclasses import dataclass

import numpy as np

from ..core.text import tokenize


@dataclass
class Doc:
    id: str
    title: str
    director: str
    year: int
    genre: str
    text: str

    @property
    def full_text(self) -> str:
        return f"{self.title} {self.director} {self.genre} {self.text}"


@dataclass
class Query:
    query: str
    doc_id: str
    level: int


def _jsonl(path):
    return [json.loads(l) for l in pathlib.Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def load_corpus(path) -> list[Doc]:
    return [Doc(**r) for r in _jsonl(path)]


def load_queries(path) -> list[Query]:
    return [Query(**r) for r in _jsonl(path)]


class QuerySampler:
    """Mixes three supervision signals so the gate learns *what kind* of query it sees:

    level 0 – navigational (title / title+director)      -> reflex path
    level 1 – keyword fragments copied from the document  -> sparse path
    level 2 – hand-written abstract descriptions (+ word-dropout) -> deep path
    """

    def __init__(self, docs: list[Doc], abstract: list[Query], seed: int = 0):
        self.docs, self.rng = docs, np.random.default_rng(seed)
        self.doc_idx = {d.id: i for i, d in enumerate(docs)}
        self.abstract = abstract

    def _title(self, i: int) -> str:
        d = self.docs[i]
        return self.rng.choice([d.title, f"{d.title} {d.director}", f"فیلم {d.title}", f"{d.title} {d.year}"])

    def _span(self, i: int) -> str:
        toks = tokenize(self.docs[i].text)
        n = int(self.rng.integers(2, 5)); s = int(self.rng.integers(0, max(1, len(toks) - n)))
        return " ".join(toks[s:s + n])

    def _abstract(self, q: Query) -> str:
        toks = q.query.split()
        if len(toks) > 3 and self.rng.random() < 0.4:
            toks.pop(int(self.rng.integers(len(toks))))
        return " ".join(toks)

    def sample(self, n: int) -> tuple[list[str], np.ndarray, np.ndarray]:
        texts, docs, levels = [], [], []
        for _ in range(n):
            r = self.rng.random()
            if r < 0.25:
                i = int(self.rng.integers(len(self.docs))); texts.append(self._title(i)); lv = 0
            elif r < 0.5:
                i = int(self.rng.integers(len(self.docs))); texts.append(self._span(i)); lv = 1
            else:
                q = self.abstract[int(self.rng.integers(len(self.abstract)))]
                i = self.doc_idx[q.doc_id]; texts.append(self._abstract(q)); lv = 2
            docs.append(i); levels.append(lv)
        return texts, np.array(docs), np.array(levels)
