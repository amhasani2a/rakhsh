"""Decider – a general, few-shot, explainable *decision layer* (the 'hodhod' of Rakhsh).

An agent (or a human) defines labels and teaches with examples; the decider answers in a
single forward pass with a hard decision, a confidence, an abstain flag and word-level evidence.

Modes
  * ``categorical`` – labels are alternatives (e.g. tools: search / calculator / email).
  * ``ordinal``     – labels are an escalation ladder (e.g. no-llm < small-llm < large-llm);
                      trained as a discrete hazard, so when unsure it *escalates*, never under-shoots.

Learning is two-speed: every example becomes an episodic prototype immediately (instant
recall), and a small network is refit with a few gradient steps (generalisation).

Typical teacher–student loop::

    d = registry.decider("llm-router")
    r = d.decide(prompt)
    if r["abstained"]:                    # student unsure → ask the expensive teacher
        label = big_llm_choose(prompt)
        d.teach([{"text": prompt, "label": label}], actor="agent:teacher")
"""
from __future__ import annotations

import json
import os
import pathlib
import threading
import time

import numpy as np

from .config import DeciderConfig, TrustPolicy, from_dict, to_dict
from .core import autograd as ag
from .core.nn import Adam, Linear, Module
from .core.text import HashedFeaturizer, tokenize
from .errors import RakhshError


class _Net(Module):
    def __init__(self, f, h, out, seed):
        rng = np.random.default_rng(seed)
        self.l1 = Linear(f, h, rng); self.l2 = Linear(h, out, rng)

    def forward(self, x):
        return self.l2(ag.tanh(self.l1(x)))

    def hidden(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(self.l1.np(x))

    def logits(self, x: np.ndarray) -> np.ndarray:
        return self.l2.np(self.hidden(x))


class Decider:
    def __init__(self, name: str, labels: list[str], mode: str = "categorical", description: str = "",
                 cfg: DeciderConfig | None = None, trust: TrustPolicy | None = None, on_event=None):
        if len(labels) < 2 or len(set(labels)) != len(labels):
            raise RakhshError("a decider needs >= 2 distinct labels")
        if mode not in ("categorical", "ordinal"):
            raise RakhshError("mode must be 'categorical' or 'ordinal'")
        self.name, self.labels, self.mode, self.description = name, list(labels), mode, description
        self.cfg = cfg or DeciderConfig(); self.trust = trust or TrustPolicy()
        self.on_event = on_event or (lambda **kw: None)
        self.feat = HashedFeaturizer(self.cfg.featurizer)
        L = len(labels)
        self.net = _Net(self.cfg.featurizer.dim, self.cfg.hidden, L if mode == "categorical" else L - 1, self.cfg.seed)
        self.examples: list[dict] = []
        self.lock = threading.RLock()
        self.workdir: pathlib.Path | None = None
        self.stats = {"decisions": 0, "abstained": 0, "taught": 0, "fit_acc": None}
        self._protos = np.zeros((0, self.cfg.hidden), np.float32); self._proto_y = np.zeros(0, int)
        self._proto_w = np.zeros(0)

    # ------------------------------------------------------------------ learning
    def teach(self, examples: list[dict], actor: str = "human", refit: bool = True) -> dict:
        """examples: [{"text": ..., "label": ...}, ...]. Weighted by the actor's trust."""
        trust = self.trust.of(actor); added = 0
        with self.lock:
            for e in examples:
                if e["label"] not in self.labels:
                    raise RakhshError(f"unknown label {e['label']!r}; labels = {self.labels}")
                if not str(e.get("text", "")).strip():
                    continue
                self.examples.append({"text": str(e["text"]), "label": e["label"], "w": trust, "actor": actor, "t": time.time()})
                added += 1
            self.stats["taught"] += added
            acc = self.fit(self.cfg.online_steps if self.stats.get("fit_acc") is not None else self.cfg.fit_steps) if refit else None
        self.on_event(decider=self.name, actor=actor, action="teach", n=added, trust=trust)
        self.save()
        return {"ok": True, "added": added, "total": len(self.examples), "trust": trust, "fit_acc": acc}

    def fit(self, steps: int | None = None) -> float | None:
        with self.lock:
            if not self.examples:
                return None
            X = self.feat([e["text"] for e in self.examples])
            y = np.array([self.labels.index(e["label"]) for e in self.examples])
            w = np.array([e["w"] for e in self.examples], float)
            opt = Adam(self.net.parameters(), lr=self.cfg.lr, weight_decay=1e-4)
            for _ in range(steps or self.cfg.fit_steps):
                z = self.net.forward(ag.Tensor(X))
                loss = ag.cross_entropy(z, y, w) if self.mode == "categorical" else ag.hazard_nll(z, y, w)
                opt.zero_grad(); loss.backward(); opt.step()
            H = self.net.hidden(X)
            self._protos = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-8); self._proto_y = y; self._proto_w = w
            acc = float((self._probs(X).argmax(1) == y).mean()); self.stats["fit_acc"] = acc
            return acc

    # ------------------------------------------------------------------ inference
    def _net_probs(self, X):
        z = self.net.logits(X)
        if self.mode == "categorical":
            z = z - z.max(1, keepdims=True); p = np.exp(z); return p / p.sum(1, keepdims=True)
        h = 1 / (1 + np.exp(-z))
        surv = np.cumprod(np.concatenate([np.ones((len(h), 1)), 1 - h], 1), 1)
        return np.concatenate([h * surv[:, :-1], surv[:, -1:]], 1)

    def _probs(self, X):
        p = self._net_probs(X)
        if not len(self._proto_y):
            return p
        H = self.net.hidden(X); H = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-8)
        sims = H @ self._protos.T                                      # (B, n_examples)
        L = len(self.labels); best = np.full((len(X), L), -1.0)
        for c in range(L):
            m = self._proto_y == c
            if m.any():
                best[:, c] = (sims[:, m] * np.minimum(1, self._proto_w[m])[None]).max(1)
        z = best / self.cfg.proto_temperature; z -= z.max(1, keepdims=True)
        pp = np.exp(z) * (best > -1); pp /= pp.sum(1, keepdims=True) + 1e-12
        wgt = self.cfg.proto_weight * float(best.max(1).mean() > 0.5)
        return (1 - wgt) * p + wgt * pp

    def decide(self, text: str, explain: bool = True) -> dict:
        with self.lock:
            t0 = time.perf_counter()
            if not self.examples:
                return {"label": None, "abstained": True, "confidence": 0.0, "reason": "no examples yet – teach me",
                        "probs": {l: 1 / len(self.labels) for l in self.labels}, "evidence": []}
            p = self._probs(self.feat([text]))[0]
            j = int(p.argmax()); conf = float(p[j])
            abstained = conf < self.cfg.abstain_below
            label = self.labels[j]
            if abstained and self.mode == "ordinal":            # safe escalation when unsure:
                k = int(np.searchsorted(np.cumsum(p), 0.8))      # smallest level covering 80% mass
                label = self.labels[min(max(j, k), len(p) - 1)]
            ev = self._evidence(text, j) if explain else []
            self.stats["decisions"] += 1; self.stats["abstained"] += int(abstained)
            return {"label": label, "abstained": abstained, "confidence": round(conf, 4),
                    "probs": {l: round(float(x), 4) for l, x in zip(self.labels, p)},
                    "evidence": ev, "ms": round((time.perf_counter() - t0) * 1e3, 3)}

    def _evidence(self, text: str, j: int, top: int = 5) -> list[list]:
        """Occlusion attribution: how much does removing each word lower P(decision)?"""
        words = tokenize(text)
        if not words:
            return []
        uniq = list(dict.fromkeys(words))[:40]
        variants = [" ".join(w for w in words if w != u) for u in uniq]
        base = self._probs(self.feat([text]))[0, j]
        drop = base - self._probs(self.feat(variants))[:, j]
        order = np.argsort(-drop)[:top]
        return [[uniq[i], round(float(drop[i]), 4)] for i in order if drop[i] > 0.005]

    # ------------------------------------------------------------------ persistence
    def info(self) -> dict:
        with self.lock:
            counts = {l: sum(e["label"] == l for e in self.examples) for l in self.labels}
            return {"name": self.name, "labels": self.labels, "mode": self.mode, "description": self.description,
                    "examples": len(self.examples), "per_label": counts, "stats": dict(self.stats)}

    def save(self):
        if self.workdir is None:
            return
        with self.lock:
            d = self.workdir; d.mkdir(parents=True, exist_ok=True)
            meta = {"format": 1, "name": self.name, "labels": self.labels, "mode": self.mode,
                    "description": self.description, "config": to_dict(self.cfg), "stats": self.stats,
                    "examples": self.examples}
            tmp = d / "decider.json.tmp"; tmp.write_text(json.dumps(meta, ensure_ascii=False), "utf-8"); os.replace(tmp, d / "decider.json")
            t2 = d / "net.tmp.npz"; np.savez_compressed(t2, **self.net.state_dict()); os.replace(t2, d / "net.npz")

    @classmethod
    def load(cls, directory, **kw) -> "Decider":
        d = pathlib.Path(directory); meta = json.loads((d / "decider.json").read_text("utf-8"))
        o = cls(meta["name"], meta["labels"], meta["mode"], meta.get("description", ""),
                from_dict(DeciderConfig, meta["config"]), **kw)
        o.examples = meta["examples"]; o.stats.update(meta["stats"]); o.workdir = d
        if (d / "net.npz").exists():
            with np.load(d / "net.npz") as z:
                o.net.load_state_dict({k: z[k] for k in z.files})
        if o.examples:
            X = o.feat([e["text"] for e in o.examples]); H = o.net.hidden(X)
            o._protos = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-8)
            o._proto_y = np.array([o.labels.index(e["label"]) for e in o.examples]); o._proto_w = np.array([e["w"] for e in o.examples])
        return o
