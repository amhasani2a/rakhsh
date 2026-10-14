"""Extractive answering: the reply is always quoted from the user's own data."""
from __future__ import annotations
import re

import numpy as np

from ..ingest.chunk import Chunk
from .text import content_tokens, normalize, split_sentences

_WORD = re.compile(r"\w+", re.UNICODE)


def highlight(text: str, typed: set[str], expanded: set[str]) -> list[list]:
    """-> [[segment, 0|1|2]]   1 = typed word, 2 = expanded spike the user never wrote."""
    segs, last = [], 0
    for m in _WORD.finditer(text):
        w = normalize(m.group())
        tag = 1 if w in typed else 2 if w in expanded else 0
        if tag:
            if m.start() > last:
                segs.append([text[last:m.start()], 0])
            segs.append([m.group(), tag]); last = m.end()
    if last < len(text):
        segs.append([text[last:], 0])
    return segs


def best_window(chunk: Chunk, weights: dict[str, float], max_words: int = 70) -> str:
    sents = chunk.sentences or split_sentences(chunk.text) or [chunk.text]
    if len(sents) == 1:
        return sents[0]
    sc = np.array([sum(weights.get(t, 0.0) for t in set(content_tokens(s))) for s in sents])
    b = int(sc.argmax()); lo = hi = b; n = len(sents[b].split())
    while True:
        cand = [(sc[i], i) for i in (lo - 1, hi + 1) if 0 <= i < len(sents)]
        if not cand:
            break
        _, i = max(cand)
        w = len(sents[i].split())
        if n + w > max_words:
            break
        n += w; lo, hi = min(lo, i), max(hi, i)
    return " ".join(sents[lo:hi + 1])
