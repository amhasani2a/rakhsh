"""ReflexSearchAgent – one forward pass decides meaning, keywords AND search depth.

No loops, no tool-calling, no re-ranker: the hazard gate's jump picks a level
policy and the retrieval back-ends just execute it.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field

import numpy as np

from .cache import SemanticCache
from .config import Config
from .data import Doc
from ..core.index import DenseIndex, InvertedIndex, score_fusion
from ..core.model import ReflexNet
from ..core.text import HashedFeaturizer, Vocab, content_tokens


@dataclass
class Hit:
    doc_id: str
    title: str
    score: float


@dataclass
class SearchResult:
    query: str
    hits: list[Hit]
    level: int
    level_name: str
    k: int
    confidence: float
    stop_probs: list[float]
    spikes: list[tuple[str, float, bool]]   # (term, weight, appeared_in_query?)
    cache_hit: bool
    timings_ms: dict = field(default_factory=dict)

    def pretty(self) -> str:
        sp = ", ".join(f"{t}{'' if inq else '*'}:{w:.2f}" for t, w, inq in self.spikes[:8])
        lines = [f"🔎 {self.query}",
                 f"   gate → level {self.level} [{self.level_name}]  K={self.k}  conf={self.confidence:.2f}  "
                 f"P(stop)={[round(p, 2) for p in self.stop_probs]}  cache={'HIT' if self.cache_hit else 'miss'}",
                 f"   spikes (*=not in query): {sp}",
                 f"   time: " + "  ".join(f"{k}={v:.2f}ms" for k, v in self.timings_ms.items())]
        lines += [f"   {i+1}. {h.title}  ({h.score:.3f})" for i, h in enumerate(self.hits[:5])]
        return "\n".join(lines)


class ReflexSearchAgent:
    def __init__(self, model: ReflexNet, featurizer: HashedFeaturizer, vocab: Vocab, docs: list[Doc], cfg: Config):
        self.model, self.feat, self.vocab, self.docs, self.cfg = model, featurizer, vocab, docs, cfg
        texts = [d.full_text for d in docs]
        self.sparse_index = InvertedIndex(vocab.lexical(texts))
        self.dense_index = DenseIndex(model.infer(featurizer(texts), prune=False).dense)
        self.cache = SemanticCache(cfg.agent.cache_capacity, cfg.agent.cache_similarity)

    def encode(self, queries: list[str]):
        return self.model.infer(self.feat(queries), lex=self.vocab.lexical(queries))

    def search(self, query: str, k: int | None = None, use_cache: bool = True) -> SearchResult:
        t0 = time.perf_counter()
        out = self.encode([query])                                    # ← the single forward pass
        t1 = time.perf_counter()
        level = int(out.level[0]); conf = float(out.confidence[0])
        pol = self.cfg.agent.policies[level]
        k = k or (min(pol.k * 2, self.cfg.agent.max_k) if conf < self.cfg.agent.low_confidence else pol.k)
        dense, sparse = out.dense[0], out.sparse[0]

        hits, cache_hit = None, False
        key = f"{level}:{' '.join(content_tokens(query))}"
        if use_cache and pol.use_cache:
            hits = self.cache.get(key, dense); cache_hit = hits is not None
        if hits is None:
            rankings, weights, wd = [], [], self.cfg.agent.dense_weight
            if pol.use_sparse:
                rankings.append(self.sparse_index.search(sparse, max(k, 10))); weights.append(1 - wd)
            if pol.use_dense:
                rankings.append(self.dense_index.search(dense, max(k, 10))); weights.append(wd)
            ranked = rankings[0][:k] if len(rankings) == 1 else score_fusion(rankings, weights, k)
            hits = [Hit(self.docs[i].id, self.docs[i].title, s) for i, s in ranked]
            if use_cache and pol.use_cache:
                self.cache.put(key, dense, hits)
        t2 = time.perf_counter()

        q_terms = set(content_tokens(query))
        nz = np.nonzero(sparse)[0]; nz = nz[np.argsort(-sparse[nz])]
        spikes = [(self.vocab.terms[j], float(sparse[j]), self.vocab.terms[j] in q_terms) for j in nz]
        return SearchResult(query, hits[:k], level, pol.name, k, conf, out.stop_probs[0].tolist(), spikes,
                            cache_hit, {"forward": (t1 - t0) * 1e3, "retrieve": (t2 - t1) * 1e3})
