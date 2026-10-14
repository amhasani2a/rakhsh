"""ReflexNet: one shared trunk + ONE fused head matmul -> dense, sparse spikes, hazard gate.

    x (hashed text) ──► trunk: tanh(x·W1+b1) ──► z = h·[W_dense | W_sparse | W_hazard] + b
                                                  │          │            │
                                     L2-normalise ┘   log1p(ReLU) ┘    sigmoid ┘
                                     (right hand)     "jump" spikes   (reflex brain)

The three heads are columns of a single weight matrix, so the whole
decision (meaning + keywords + how hard to search) is ONE forward pass.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from . import autograd as ag
from ..config import ModelConfig
from .nn import Linear, Module


@dataclass
class ReflexOutput:
    dense: np.ndarray        # (B, D)    unit-norm semantic vectors
    sparse: np.ndarray       # (B, V)    exact-zero except a few spikes
    hazard: np.ndarray       # (B, L-1)  P(stop at level l | reached l)
    stop_probs: np.ndarray   # (B, L)    P(stop exactly at level l)
    level: np.ndarray        # (B,)      level where the gate "fired"
    confidence: np.ndarray   # (B,)      stop_probs at the chosen level


class ReflexNet(Module):
    def __init__(self, in_dim: int, vocab_size: int, cfg: ModelConfig, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.cfg, self.V, self.D, self.G = cfg, vocab_size, cfg.dense_dim, cfg.levels - 1
        self.trunk = Linear(in_dim, cfg.hidden, rng)
        self.heads = Linear(cfg.hidden, self.D + self.V + self.G, rng)
        self.heads.b.data[self.D:self.D + self.V] = -0.1   # start quiet: neurons below threshold

    # ---------- training graph ----------
    def forward(self, x: ag.Tensor):
        z = self.heads(ag.tanh(self.trunk(x)))             # the single fused head projection
        D, V = self.D, self.V
        dense = ag.l2_normalize(ag.slice_(z, cols=slice(0, D)))
        sparse = ag.log1p(ag.relu(ag.slice_(z, cols=slice(D, D + V))))
        hazard_logits = ag.slice_(z, cols=slice(D + V, D + V + self.G))
        return dense, sparse, hazard_logits

    # ---------- fast inference (pure numpy, no graph) ----------
    def infer(self, x: np.ndarray, prune: bool = True, lex: np.ndarray | None = None) -> ReflexOutput:
        z = self.heads.np(np.tanh(self.trunk.np(x)))
        D, V = self.D, self.V
        dense = z[:, :D]; dense = dense / (np.linalg.norm(dense, axis=1, keepdims=True) + 1e-8)
        sparse = np.log1p(np.maximum(z[:, D:D + V], 0))
        if lex is not None:
            sparse = sparse + self.cfg.lexical_skip * lex
        if prune:
            sparse = self._jump(sparse)
        hazard = 1 / (1 + np.exp(-z[:, D + V:]))
        stop = self.stop_distribution(hazard)
        level = self._fire(hazard)
        return ReflexOutput(dense, sparse, hazard, stop, level, stop[np.arange(len(level)), level])

    def _jump(self, s: np.ndarray) -> np.ndarray:
        """Action-potential pruning: everything below a relative threshold is silenced to exactly 0."""
        out = np.where(s >= self.cfg.spike_rel_threshold * s.max(1, keepdims=True), s, 0.0)
        if out.shape[1] > self.cfg.max_spikes:
            kth = -np.partition(-out, self.cfg.max_spikes - 1, axis=1)[:, self.cfg.max_spikes - 1:self.cfg.max_spikes]
            out = np.where(out >= kth, out, 0.0)
        return out

    def _fire(self, hazard: np.ndarray) -> np.ndarray:
        """Hard jump: the first level whose hazard crosses the threshold locks the switch."""
        fired = hazard >= self.cfg.hazard_threshold
        return np.where(fired.any(1), fired.argmax(1), self.G).astype(int)

    @staticmethod
    def stop_distribution(hazard: np.ndarray) -> np.ndarray:
        survive = np.cumprod(np.concatenate([np.ones((len(hazard), 1)), 1 - hazard], 1), 1)
        return np.concatenate([hazard * survive[:, :-1], survive[:, -1:]], 1)

    def resize_vocab(self, old_terms: list[str], new_terms: list[str], seed: int = 0):
        """Grow/shrink the sparse 'spike' neurons when new data arrives, keeping learned synapses."""
        D, V, G = self.D, self.V, self.G
        W, b = self.heads.W.data, self.heads.b.data
        rng = np.random.default_rng(seed); lim = np.sqrt(6 / (W.shape[0] + len(new_terms)))
        Ws = rng.uniform(-lim, lim, (W.shape[0], len(new_terms))).astype(np.float32)
        bs = np.full(len(new_terms), -0.1, np.float32)
        old = {t: i for i, t in enumerate(old_terms)}
        for j, t in enumerate(new_terms):
            i = old.get(t)
            if i is not None:
                Ws[:, j] = W[:, D + i]; bs[j] = b[D + i]
        self.heads.W.data = np.concatenate([W[:, :D], Ws, W[:, D + V:]], 1)
        self.heads.b.data = np.concatenate([b[:D], bs, b[D + V:]])
        self.V = len(new_terms)

    def trunk_np(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(self.trunk.np(x))

    def gate_cols(self) -> slice:
        return slice(self.D + self.V, self.D + self.V + self.G)

    def n_params(self) -> int:
        return sum(p.data.size for p in self.parameters().values())
