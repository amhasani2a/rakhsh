"""Semantic LRU cache: level-0 'reflex' answers without touching any index."""
from __future__ import annotations
from collections import OrderedDict

import numpy as np


class SemanticCache:
    def __init__(self, capacity: int = 256, threshold: float = 0.97):
        self.capacity, self.threshold = capacity, threshold
        self._store: OrderedDict[str, tuple[np.ndarray, list]] = OrderedDict()
        self.hits = self.misses = 0

    def get(self, key: str, vec: np.ndarray):
        if key in self._store:
            self._store.move_to_end(key); self.hits += 1
            return self._store[key][1]
        if self._store:
            keys = list(self._store)
            mat = np.stack([self._store[k][0] for k in keys])
            sims = mat @ vec; j = int(sims.argmax())
            if sims[j] >= self.threshold:
                self._store.move_to_end(keys[j]); self.hits += 1
                return self._store[keys[j]][1]
        self.misses += 1
        return None

    def put(self, key: str, vec: np.ndarray, value: list):
        self._store[key] = (vec, value); self._store.move_to_end(key)
        while len(self._store) > self.capacity:
            self._store.popitem(last=False)

    def clear(self):
        self._store.clear(); self.hits = self.misses = 0
