"""Persian-aware normalisation, tokenisation, vocabulary and hashed featurisation."""
from __future__ import annotations
import math
import re
import zlib
from collections import Counter
from typing import Iterable

import numpy as np

from ..config import FeaturizerConfig

_CHAR_MAP = str.maketrans({
    "ي": "ی", "ى": "ی", "ك": "ک", "ة": "ه", "أ": "ا", "إ": "ا", "آ": "ا", "ؤ": "و",
    "\u200c": " ", "\u200f": " ", "‌": " ", "-": " ", "_": " ",
    **{chr(0x06F0 + i): str(i) for i in range(10)},   # Persian digits
    **{chr(0x0660 + i): str(i) for i in range(10)},   # Arabic digits
})
_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670\u0640]")
_TOKEN = re.compile(r"\w+", re.UNICODE)

STOPWORDS = frozenset("""
و در به از که این آن با را برای است هست بود شد شده می ها های یک تا بر یا هم نیز اش ای
فیلم فیلمی درباره داستان کسی چیزی خود او آنها ما شود شوند کند کنند کرده دارد دارند باید
پس روی زیر بین همه یکی دیگر ی سال اند چه چرا چطور کجا کی آیا چی رو میشه هست؟ بود؟ بودن
the a an of and or to in on at for is are was were be been it this that with as by from what
who when where why how which do does did not no yes i you he she we they me my your his her its
""".split())

_SENT = re.compile(r"(?<=[\.!؟\?;؛])\s+|\n+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text) if s and s.strip()]


def normalize(text: str) -> str:
    text = _DIACRITICS.sub("", text.translate(_CHAR_MAP))
    return text.lower()


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(normalize(text))


def content_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS and len(t) > 1]


class Vocab:
    """Output vocabulary of the sparse head (the 'spike' neurons) + doc-side lexical weights."""

    def __init__(self, terms: list[str], idf: np.ndarray):
        self.terms = terms
        self.index = {t: i for i, t in enumerate(terms)}
        self.idf = idf.astype(np.float32)

    def __len__(self) -> int:
        return len(self.terms)

    @classmethod
    def build(cls, docs: Iterable[str], max_size: int | None = None) -> "Vocab":
        docs = [set(content_tokens(d)) for d in docs]
        df = Counter(t for d in docs for t in d)
        terms = sorted(df) if not max_size or len(df) <= max_size else \
            sorted(t for t, _ in df.most_common(max_size))
        n = len(docs)
        idf = np.array([math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) for t in terms])
        return cls(terms, idf)

    def lexical(self, texts: list[str]) -> np.ndarray:
        """Doc-side sparse vectors: (1+log tf)*idf, L2-normalised. Shape (N, V)."""
        m = np.zeros((len(texts), len(self)), dtype=np.float32)
        for i, txt in enumerate(texts):
            for t, c in Counter(content_tokens(txt)).items():
                j = self.index.get(t)
                if j is not None:
                    m[i, j] = (1 + math.log(c)) * self.idf[j]
        m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-8
        return m

    def lexical_sparse(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        """Same weights as `lexical` but as (indices, values) – memory-safe for big corpora."""
        items = [(self.index[t], (1 + math.log(c)) * self.idf[self.index[t]])
                 for t, c in Counter(content_tokens(text)).items() if t in self.index]
        if not items:
            return np.zeros(0, np.int64), np.zeros(0, np.float32)
        idx, val = map(np.array, zip(*items))
        return idx.astype(np.int64), (val / (np.linalg.norm(val) + 1e-8)).astype(np.float32)

    def densify(self, rows: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        m = np.zeros((len(rows), len(self)), np.float32)
        for i, (idx, val) in enumerate(rows):
            m[i, idx] = val
        return m

    def to_dict(self) -> dict:
        return {"terms": self.terms, "idf": self.idf.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "Vocab":
        return cls(d["terms"], np.array(d["idf"]))


class HashedFeaturizer:
    """Open-vocabulary input layer: words + char n-grams hashed into a fixed space.

    Char n-grams make the model robust to Persian morphology (قتل / قاتل / قاتلی)
    and typos without any external tokenizer or pretrained model.
    """

    def __init__(self, cfg: FeaturizerConfig):
        self.cfg = cfg

    def _feats(self, text: str) -> Counter:
        c: Counter = Counter()
        for w in tokenize(text):
            if self.cfg.use_words:
                c["w:" + w] += 1
            padded = f"<{w}>"
            for n in self.cfg.char_ngrams:
                for i in range(max(1, len(padded) - n + 1)):
                    c["c:" + padded[i:i + n]] += 1
        return c

    def sparse(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        acc: dict[int, float] = {}
        for f, v in self._feats(text).items():
            j = zlib.crc32(f.encode("utf-8")) % self.cfg.dim
            acc[j] = acc.get(j, 0.0) + 1.0 + math.log(v)
        if not acc:
            return np.zeros(0, np.int64), np.zeros(0, np.float32)
        idx = np.fromiter(acc.keys(), np.int64); val = np.fromiter(acc.values(), np.float32)
        return idx, val / (np.linalg.norm(val) + 1e-8)

    def densify(self, rows) -> np.ndarray:
        x = np.zeros((len(rows), self.cfg.dim), np.float32)
        for i, (idx, val) in enumerate(rows):
            x[i, idx] = val
        return x

    def __call__(self, texts: list[str]) -> np.ndarray:
        x = np.zeros((len(texts), self.cfg.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for f, v in self._feats(t).items():
                x[i, zlib.crc32(f.encode("utf-8")) % self.cfg.dim] += 1.0 + math.log(v)
        x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
        return x
