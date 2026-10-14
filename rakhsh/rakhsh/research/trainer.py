"""End-to-end training of all three heads with one shared forward pass per batch."""
from __future__ import annotations
import time

import numpy as np

from ..core import autograd as ag
from .config import Config
from .data import Doc, Query, QuerySampler
from ..core.model import ReflexNet
from ..core.nn import Adam
from ..core.text import HashedFeaturizer, Vocab


class Trainer:
    def __init__(self, cfg: Config, docs: list[Doc], train_queries: list[Query]):
        self.cfg, self.docs = cfg, docs
        self.feat = HashedFeaturizer(cfg.featurizer)
        self.vocab = Vocab.build([d.full_text for d in docs])
        self.model = ReflexNet(cfg.featurizer.dim, len(self.vocab), cfg.model, seed=cfg.train.seed)
        self.sampler = QuerySampler(docs, train_queries, seed=cfg.train.seed)
        self.doc_x = self.feat([d.full_text for d in docs])
        self.doc_lex = ag.Tensor(self.vocab.lexical([d.full_text for d in docs]))   # fixed, index-friendly
        self.opt = Adam(self.model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        self.rng = np.random.default_rng(cfg.train.seed)

    def step(self, it: int) -> dict:
        tc = self.cfg.train
        texts, y, levels = self.sampler.sample(tc.batch_size)
        xq = self.feat(texts)
        xq *= (self.rng.random(xq.shape) > tc.input_dropout) / (1 - tc.input_dropout)
        b = len(texts)
        # queries and documents share one forward pass through the same network
        dense, sparse, hz = self.model.forward(ag.Tensor(np.concatenate([xq, self.doc_x])))
        q_dense, d_dense = ag.slice_(dense, rows=slice(0, b)), ag.slice_(dense, rows=slice(b, None))
        q_exp, q_hz = ag.slice_(sparse, rows=slice(0, b)), ag.slice_(hz, rows=slice(0, b))
        q_sparse = ag.add(q_exp, self.cfg.model.lexical_skip * self.vocab.lexical(texts))   # lexical skip

        l_dense = ag.cross_entropy(ag.scale(q_dense @ d_dense.T, 1 / tc.dense_temperature), y)
        l_sparse = ag.cross_entropy(ag.scale(q_sparse @ self.doc_lex.T, tc.sparse_scale), y)
        l_flops = ag.sum_(ag.square(ag.mean(q_exp, axis=0)))
        l_gate = ag.hazard_nll(q_hz, levels)
        lam = tc.flops_lambda * min(1.0, (it / tc.flops_warmup)) ** 2
        loss = l_dense + l_sparse + ag.scale(l_flops, lam) + ag.scale(l_gate, tc.gate_lambda)

        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return dict(loss=float(loss.data), dense=float(l_dense.data), sparse=float(l_sparse.data),
                    flops=float(l_flops.data), gate=float(l_gate.data),
                    spikes=float((q_exp.data > 0).sum(1).mean()))

    def fit(self, verbose: bool = True) -> ReflexNet:
        t0 = time.time(); tc = self.cfg.train
        for it in range(1, tc.steps + 1):
            log = self.step(it)
            if verbose and (it % tc.log_every == 0 or it == 1):
                print(f"step {it:4d} | " + " ".join(f"{k}={v:.3f}" for k, v in log.items()) + f" | {time.time()-t0:.1f}s")
        return self.model
