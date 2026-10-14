"""Two-speed learning borrowed from neuroscience (fast, gradient-free parts).

* :class:`FastSynapses` – Hebbian / anti-Hebbian plasticity. One confirmation instantly
  wires the query's active spike-neurons to the right chunk; a rejection inhibits.
* :class:`ReflexMemory` – *episodic neurons*. Each confirmed episode is a neuron whose
  weights are the query's dense vector. When it fires (cosine >= threshold) and its
  accumulated, trust-weighted strength is high enough, the spinal reflex answers at
  level 0 – the gate is bypassed.

Slow (gradient) learning and sleep-consolidation live in :mod:`rakhsh.brain`.
"""
from __future__ import annotations

import numpy as np


class FastSynapses:
    def __init__(self):
        self.w: dict[int, dict[int, float]] = {}

    def __len__(self) -> int:
        return sum(len(v) for v in self.w.values())

    def update(self, chunk: int, q_sparse: np.ndarray, sign: float, lr: float) -> None:
        nz = np.nonzero(q_sparse)[0]
        if not len(nz):
            return
        q = q_sparse[nz] / (np.linalg.norm(q_sparse[nz]) + 1e-8)
        syn = self.w.setdefault(int(chunk), {})
        for t, v in zip(nz.tolist(), q.tolist()):
            syn[t] = float(np.clip(syn.get(t, 0.0) + sign * lr * v, -2.0, 2.0))

    def scores(self, q_sparse: np.ndarray, n: int) -> np.ndarray:
        out = np.zeros(n, np.float32)
        norm = np.linalg.norm(q_sparse) + 1e-8
        for c, syn in self.w.items():
            if c < n:
                out[c] = sum(q_sparse[t] * v for t, v in syn.items()) / norm
        return out

    def decay(self, factor: float, floor: float = 0.02) -> None:
        for c in list(self.w):
            self.w[c] = {t: v * factor for t, v in self.w[c].items() if abs(v * factor) > floor}
            if not self.w[c]:
                del self.w[c]

    def remap_terms(self, old_terms: list[str], new_terms: list[str]) -> None:
        new = {t: i for i, t in enumerate(new_terms)}
        self.w = {c: {new[old_terms[t]]: v for t, v in syn.items() if old_terms[t] in new}
                  for c, syn in self.w.items()}

    def remap_chunks(self, mapping: dict[int, int]) -> None:
        self.w = {mapping[c]: s for c, s in self.w.items() if c in mapping}

    # serialisation (JSON-safe)
    def to_json(self) -> dict:
        return {str(c): {str(t): v for t, v in s.items()} for c, s in self.w.items()}

    @classmethod
    def from_json(cls, d: dict) -> "FastSynapses":
        o = cls(); o.w = {int(c): {int(t): float(v) for t, v in s.items()} for c, s in d.items()}
        return o


class ReflexMemory:
    def __init__(self):
        self.texts: list[str] = []
        self.chunks: list[int] = []
        self.strength: list[float] = []
        self.keys = np.zeros((0, 0), np.float32)

    def __len__(self) -> int:
        return len(self.texts)

    def fire(self, dense: np.ndarray, threshold: float, min_strength: float = 1.0):
        """-> (chunk, similarity, index) if an episodic neuron fires, else None."""
        if not self.texts or self.keys.shape[0] != len(self.texts):
            return None
        sims = self.keys @ dense
        for j in np.argsort(-sims)[:3]:
            if sims[j] < threshold:
                break
            if self.strength[j] >= min_strength - 1e-9:
                return self.chunks[j], float(sims[j]), int(j)
        return None

    def reinforce(self, text: str, dense: np.ndarray | None, chunk: int, amount: float) -> None:
        for j, (t, c) in enumerate(zip(self.texts, self.chunks)):
            if t == text and c == chunk:
                self.strength[j] = min(self.strength[j] + amount, 5.0)
                if self.strength[j] <= 0:
                    self._drop(j)
                return
        if amount > 0 and dense is not None:
            self.texts.append(text); self.chunks.append(int(chunk)); self.strength.append(float(amount))
            self.keys = dense[None].astype(np.float32) if self.keys.size == 0 else np.vstack([self.keys, dense[None]])

    def _drop(self, j: int) -> None:
        for a in (self.texts, self.chunks, self.strength):
            a.pop(j)
        if self.keys.shape[0] > j:
            self.keys = np.delete(self.keys, j, 0)

    def reencode(self, encode) -> None:
        self.keys = encode(self.texts) if self.texts else np.zeros((0, 0), np.float32)

    def remap_chunks(self, mapping: dict[int, int]) -> None:
        keep = [j for j, c in enumerate(self.chunks) if c in mapping]
        self.texts = [self.texts[j] for j in keep]; self.strength = [self.strength[j] for j in keep]
        self.chunks = [mapping[self.chunks[j]] for j in keep]
        self.keys = self.keys[keep] if self.keys.shape[0] else self.keys

    def to_json(self) -> dict:
        return {"texts": self.texts, "chunks": self.chunks, "strength": self.strength}

    @classmethod
    def from_json(cls, d: dict) -> "ReflexMemory":
        o = cls(); o.texts, o.chunks, o.strength = list(d["texts"]), list(d["chunks"]), list(d["strength"])
        return o
