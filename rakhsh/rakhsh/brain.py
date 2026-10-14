"""Brain – a self-training, feedback-learning reflex retrieval model for ANY data.

Lifecycle::

    ingest(file | text)   -> chunks (instantly searchable via incremental indexing)
    train()               -> self-supervised: invents its own questions, learns dense + spike
                             heads, then labels the hazard gate by *outcome*
    search()/ask()        -> ONE forward pass -> (episodic reflex | hazard gate) -> retrieval
    feedback()/teach()    -> trust-weighted Hebbian plasticity + episodic memory
                             + a few gradient steps + gate update
    sleep()               -> consolidation: replay experience into slow weights, decay fast ones

Thread-safety: every public method takes ``self.lock``; training runs on a deep copy
of the model in a background thread and is swapped in atomically.
"""
from __future__ import annotations

import copy
import logging
import threading
import time
from collections import deque
from typing import Callable

import numpy as np

from .config import BrainConfig, TrustPolicy
from .core import autograd as ag
from .core.answer import best_window, highlight
from .core.index import InvertedIndex
from .core.model import ReflexNet
from .core.nn import Adam
from .core.plasticity import FastSynapses, ReflexMemory
from .core.selfsup import PseudoQueries
from .core.text import HashedFeaturizer, Vocab, content_tokens
from .errors import NotFound, NotReady, RakhshError
from .ingest import Chunk, ingest

log = logging.getLogger("rakhsh")
LEVELS = [("reflex", "رفلکس (حافظه)"), ("spike", "اسپایک (کلیدواژه)"), ("deep", "عمیق (هیبرید)")]


def train_gate(model: ReflexNet, H: np.ndarray, labels: np.ndarray, steps: int, lr: float = 0.05,
               weights: np.ndarray | None = None) -> float:
    """Fit only the hazard-gate columns of the fused head (numpy Adam, discrete-hazard NLL)."""
    cols = model.gate_cols()
    W = model.heads.W.data[:, cols].astype(np.float64); b = model.heads.b.data[cols].astype(np.float64)
    G = W.shape[1]; ar = np.arange(G)[None]
    before = (ar < labels[:, None]).astype(float); at = (ar == labels[:, None]).astype(float)
    freq = np.bincount(labels, minlength=G + 1).astype(float)
    w = 1.0 / np.maximum(freq[labels], 1.0)              # class balance
    if weights is not None:
        w = w * weights
    w /= w.sum()
    mW, vW, mb, vb = np.zeros_like(W), np.zeros_like(W), np.zeros_like(b), np.zeros_like(b)
    for t in range(1, steps + 1):
        h = 1 / (1 + np.exp(-(H @ W + b)))
        g = (before * h + at * (h - 1)) * w[:, None]
        for p, d, m, v in ((W, H.T @ g + 1e-4 * W, mW, vW), (b, g.sum(0), mb, vb)):
            m[:] = 0.9 * m + 0.1 * d; v[:] = 0.999 * v + 0.001 * d * d
            p -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    model.heads.W.data[:, cols] = W.astype(np.float32); model.heads.b.data[cols] = b.astype(np.float32)
    h = 1 / (1 + np.exp(-(H @ W + b)))
    return float((model._fire(h) == labels).mean())


class Brain:
    def __init__(self, name: str = "default", cfg: BrainConfig | None = None,
                 trust: TrustPolicy | None = None, on_event: Callable[..., None] | None = None):
        self.name = name
        self.cfg = cfg or BrainConfig()
        self.trust = trust or TrustPolicy()
        self.on_event = on_event or (lambda **kw: None)
        self.feat = HashedFeaturizer(self.cfg.featurizer)
        self.lock = threading.RLock()
        self.rng = np.random.default_rng(self.cfg.seed)
        self._thread: threading.Thread | None = None
        self._pending = False
        self.msgs: dict[int, dict] = {}
        self._clear()

    def _clear(self):
        self.chunks: list[Chunk] = []
        self.sources: list[dict] = []
        self.vocab: Vocab | None = None
        self.model: ReflexNet | None = None
        self.n_indexed = 0
        self.feat_rows: list = []; self.lex_rows: list = []
        self.sparse_index: InvertedIndex | None = None
        self.dense_vecs: np.ndarray | None = None
        self.synapses = FastSynapses(); self.memory = ReflexMemory()
        self.replay: list[list] = []                 # [query, chunk, gate_label, weight]
        self.gate_set: tuple[list[str], np.ndarray] = ([], np.zeros(0, int))
        self._next_id = 1
        self.stats = {"questions": 0, "up": 0, "down": 0, "picks": 0, "taught": 0, "sleeps": 0,
                      "gate_acc": None, "train_seconds": 0.0, "trained_at": None, "recent": deque(maxlen=30)}
        self.status = {"state": "empty", "progress": 0.0, "message": "منتظر داده… / waiting for data"}

    # ================================================================ knowledge
    def ingest(self, name: str, data: bytes, train: bool = True, actor: str = "human") -> dict:
        chunks = ingest(name, data, self.cfg.chunk_words)
        if not chunks:
            raise RakhshError("هیچ متن قابل استفاده‌ای پیدا نشد / no usable text found")
        with self.lock:
            if name in {s["name"] for s in self.sources}:
                name = f"{name} ({len(self.sources) + 1})"
                for c in chunks:
                    c.source = name
            self.chunks.extend(chunks)
            info = {"name": name, "chunks": len(chunks), "kind": "rows" if chunks[0].kind == "row" else "text",
                    "words": sum(len(c.text.split()) for c in chunks), "added_by": actor, "added_at": time.time()}
            self.sources.append(info)
            self._index_incremental()
        self.on_event(brain=self.name, actor=actor, action="ingest", source=name, chunks=len(chunks))
        if train:
            self.train_async()
        return info

    def ingest_text(self, name: str, text: str, **kw) -> dict:
        return self.ingest(name if "." in name else name + ".txt", text.encode("utf-8"), **kw)

    def remove_source(self, name: str, actor: str = "human") -> None:
        with self.lock:
            if name not in {s["name"] for s in self.sources}:
                raise NotFound(f"source not found: {name}")
            keep = [i for i, c in enumerate(self.chunks) if c.source != name]
            mapping = {old: new for new, old in enumerate(keep)}
            self.chunks = [self.chunks[i] for i in keep]
            self.sources = [s for s in self.sources if s["name"] != name]
            self.replay = [[q, mapping[c], l, w] for q, c, l, w in self.replay if c in mapping]
            self.synapses.remap_chunks(mapping); self.memory.remap_chunks(mapping)
            if self.model is not None and self.chunks:
                self.feat_rows = [self.feat_rows[i] for i in keep if i < len(self.feat_rows)]
                self.lex_rows = [self.lex_rows[i] for i in keep if i < len(self.lex_rows)]
                self.dense_vecs = self.dense_vecs[[i for i in keep if i < len(self.dense_vecs)]]
                self.n_indexed = len(self.feat_rows)
                self.sparse_index = InvertedIndex.from_rows(self.lex_rows)
            if not self.chunks:
                self._clear(); return
        self.on_event(brain=self.name, actor=actor, action="remove_source", source=name)
        self.train_async()

    def _index_incremental(self) -> None:
        """Make newly added chunks searchable immediately with the current model (no retrain)."""
        if self.model is None or self.n_indexed >= len(self.chunks):
            return
        new = self.chunks[self.n_indexed:]
        texts = [c.model_text for c in new]
        fr = [self.feat.sparse(t) for t in texts]; lr = [self.vocab.lexical_sparse(t) for t in texts]
        self.feat_rows += fr; self.lex_rows += lr
        self.dense_vecs = np.concatenate([self.dense_vecs, self._encode_chunks(self.model, fr)])
        self.sparse_index = InvertedIndex.from_rows(self.lex_rows)
        self.n_indexed = len(self.chunks)

    # ================================================================ training
    def train_async(self) -> None:
        with self.lock:
            if self._thread and self._thread.is_alive():
                self._pending = True; return
            self._thread = threading.Thread(target=self._train_loop, name=f"rakhsh-train-{self.name}", daemon=True)
            self._thread.start()

    def train(self) -> None:
        """Blocking training (convenience for scripts/tests)."""
        self.train_async(); self.wait()

    def wait(self, timeout: float | None = None) -> bool:
        t0 = time.time()
        while self._thread and self._thread.is_alive():
            self._thread.join(0.1)
            if timeout and time.time() - t0 > timeout:
                return False
        return True

    @property
    def training(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def _train_loop(self):
        while True:
            self._pending = False
            try:
                self._train()
            except Exception as e:     # surfaced in status, logged with traceback
                log.exception("training failed")
                self.status = {"state": "error", "progress": 0.0, "message": f"خطا در آموزش / training error: {e}"}
                return
            if not self._pending:
                break

    def _set(self, state, progress, message):
        self.status = {"state": state, "progress": float(progress), "message": message}

    def _encode_chunks(self, model: ReflexNet, rows) -> np.ndarray:
        out = [model.infer(self.feat.densify(rows[s:s + 512]), prune=False).dense for s in range(0, len(rows), 512)]
        return np.concatenate(out) if out else np.zeros((0, model.D), np.float32)

    def _encode_queries(self, model: ReflexNet, vocab: Vocab, texts: list[str]):
        return model.infer(self.feat(texts), lex=vocab.lexical(texts))

    def _step(self, model, vocab, opt, pq, feat_rows, lex_rows, lam, replay=None, replay_frac=0.0):
        """One contrastive step: queries + their target chunks + random negatives, one shared forward."""
        c = self.cfg; b = c.batch; N = len(feat_rows)
        texts, pos = pq.sample(b); texts = list(texts); pos = pos.copy()
        weights = np.ones(b)
        if replay and replay_frac > 0:
            for j, r in enumerate(self.rng.integers(len(replay), size=max(1, int(b * replay_frac)))):
                texts[j], pos[j], weights[j] = replay[r][0], replay[r][1], 1.0 + 2.0 * replay[r][3]
        docs = np.unique(np.concatenate([pos, self.rng.integers(N, size=c.negatives)]))
        col = {d: i for i, d in enumerate(docs.tolist())}
        y = np.array([col[int(p)] for p in pos])
        xq = self.feat(texts)
        xq *= (self.rng.random(xq.shape) > c.input_dropout) / (1 - c.input_dropout)
        xd = self.feat.densify([feat_rows[d] for d in docs])
        dense, sparse, _ = model.forward(ag.Tensor(np.concatenate([xq, xd])))
        qd, dd = ag.slice_(dense, rows=slice(0, b)), ag.slice_(dense, rows=slice(b, None))
        q_exp = ag.slice_(sparse, rows=slice(0, b))
        q_sp = ag.add(q_exp, model.cfg.lexical_skip * vocab.lexical(texts))
        dlex = ag.Tensor(vocab.densify([lex_rows[d] for d in docs]))
        loss = ag.cross_entropy(ag.scale(qd @ dd.T, 1 / c.dense_temperature), y, weights) \
            + ag.cross_entropy(ag.scale(q_sp @ dlex.T, c.sparse_scale), y, weights) \
            + ag.scale(ag.sum_(ag.square(ag.mean(q_exp, axis=0))), lam)
        opt.zero_grad(); loss.backward(); opt.step()
        return float(loss.data)

    def _gate_data(self, model, vocab, sparse_index, chunks, n):
        """Outcome-based gate labels: 1 if the cheap spike index already finds the chunk, else 2."""
        texts, pos = PseudoQueries(chunks, seed=int(self.rng.integers(1 << 30))).sample(n)
        out = self._encode_queries(model, vocab, texts)
        labels = np.array([1 if (r := sparse_index.search(out.sparse[i], 1)) and r[0][0] == pos[i] else 2
                           for i in range(n)])
        return list(texts), labels

    def _fit_gate(self, model, steps):
        gt, gl = self.gate_set
        texts = list(gt) + [r[0] for r in self.replay]
        if not texts:
            return None
        labels = np.concatenate([np.asarray(gl, int), np.array([r[2] for r in self.replay], int)])
        weights = np.concatenate([np.ones(len(gt)), np.array([4.0 * r[3] for r in self.replay])])
        return train_gate(model, model.trunk_np(self.feat(texts)), labels.astype(int), steps, weights=weights)

    def _train(self):
        t0 = time.time(); c = self.cfg
        with self.lock:
            chunks = list(self.chunks); old_model, old_vocab = self.model, self.vocab
            replay = [list(r) for r in self.replay]
        if not chunks:
            return
        self._set("training", 0.02, "ساخت واژگان نورون‌های اسپایک… / building spike vocabulary")
        texts = [ch.model_text for ch in chunks]
        vocab = Vocab.build(texts, c.max_vocab)
        if old_model is None:
            model = ReflexNet(c.featurizer.dim, len(vocab), c.model, seed=c.seed)
        else:
            model = copy.deepcopy(old_model); model.resize_vocab(old_vocab.terms, vocab.terms)
        feat_rows = [self.feat.sparse(t) for t in texts]
        lex_rows = [vocab.lexical_sparse(t) for t in texts]
        opt = Adam(model.parameters(), lr=c.lr, weight_decay=c.weight_decay)
        steps = int(np.clip(c.steps_min + c.steps_per_chunk * len(chunks), c.steps_min, c.steps_max))
        if old_model is not None:
            steps = max(c.steps_min, steps // 2)                     # warm start
        pq = PseudoQueries(chunks, seed=c.seed)
        for it in range(1, steps + 1):
            lam = c.flops_lambda * min(1.0, it / (steps * 0.4)) ** 2
            loss = self._step(model, vocab, opt, pq, feat_rows, lex_rows, lam, replay, 0.15 if replay else 0)
            if it % 10 == 0:
                self._set("training", 0.05 + 0.8 * it / steps,
                          f"خودآموزی / self-training {it}/{steps} · loss={loss:.2f}")
        self._set("training", 0.88, "ساخت ایندکس‌ها / indexing")
        sparse_index = InvertedIndex.from_rows(lex_rows)
        dense_vecs = self._encode_chunks(model, feat_rows)
        self._set("training", 0.93, "آموزش گیت هزارد / fitting hazard gate")
        with self.lock:
            if old_vocab is not None:
                self.synapses.remap_terms(old_vocab.terms, vocab.terms)
        self.gate_set = self._gate_data(model, vocab, sparse_index, chunks, c.gate_samples)
        acc = self._fit_gate(model, c.gate_steps)
        with self.lock:
            self.model, self.vocab = model, vocab
            self.feat_rows, self.lex_rows = feat_rows, lex_rows
            self.sparse_index, self.dense_vecs, self.n_indexed = sparse_index, dense_vecs, len(chunks)
            self._index_incremental()                                  # chunks added meanwhile
            self.memory.reencode(lambda t: self._encode_queries(model, vocab, t).dense)
            self.stats.update(gate_acc=acc, train_seconds=time.time() - t0, trained_at=time.time())
        self._set("ready", 1.0, f"آماده ✓ ready · {len(chunks)} chunks · {len(vocab)} spike neurons · {time.time()-t0:.1f}s")
        self.on_event(brain=self.name, actor="system", action="trained", chunks=len(chunks), seconds=round(time.time() - t0, 1))
        self.save()

    # ================================================================ inference
    def _await_first_model(self, timeout: float = 600.0):
        """If the very first training is still running, block (outside the lock) until a model exists."""
        if self.model is None and self.training:
            self.wait(timeout)

    def _require_ready(self):
        if self.model is None:
            raise NotReady("هنوز آموزش ندیده / brain has no trained model yet (ingest data first)")

    def probe(self, text: str) -> dict:
        """Decision only (no retrieval): which spike neurons fire and how deep the gate wants to go."""
        with self.lock:
            if self.model is None or not text.strip():
                return {"spikes": [], "stop": [0.0] * self.cfg.model.levels, "level": None, "reflex": False}
            out = self._encode_queries(self.model, self.vocab, [text]); sp = out.sparse[0]
            typed = set(content_tokens(text)); nz = np.nonzero(sp)[0]; nz = nz[np.argsort(-sp[nz])][:40]
            ep = self.memory.fire(out.dense[0], self.cfg.memory_threshold, self.cfg.memory_min_strength)
            level = 0 if ep else int(out.level[0])
            return {"spikes": [[self.vocab.terms[j], round(float(sp[j]), 3), self.vocab.terms[j] in typed] for j in nz],
                    "stop": [round(float(x), 4) for x in out.stop_probs[0]], "level": level,
                    "level_name": LEVELS[level][0], "confidence": round(float(out.confidence[0]), 4),
                    "reflex": bool(ep), "neurons": len(self.vocab)}

    def _rank(self, level, dense, sparse, k):
        n = self.n_indexed
        s = self.sparse_index.scores(sparse)[:n]
        smax = float(s.max()) if n and s.max() > 0 else 1.0
        d = self.dense_vecs[:n] @ dense
        hebb = self.synapses.scores(sparse, n) * self.cfg.hebb_weight
        if level >= 2:
            dpos = np.maximum(d, 0); dmax = float(dpos.max()) or 1.0; wd = self.cfg.dense_weight
            score = (1 - wd) * s / smax + wd * dpos / dmax + hebb
        else:
            score = s / smax + hebb
        k = min(k, n); idx = np.argpartition(-score, k - 1)[:k]; idx = idx[np.argsort(-score[idx])]
        return [(int(i), float(score[i])) for i in idx], float(s.max()) if n else 0.0, float(d.max()) if n else 0.0

    def search(self, query: str, k: int | None = None) -> dict:
        """Single-forward retrieval. Returns ranked hits + full decision trace."""
        self._await_first_model()
        with self.lock:
            self._require_ready(); c = self.cfg; t0 = time.perf_counter()
            out = self._encode_queries(self.model, self.vocab, [query])          # ← the single forward pass
            t1 = time.perf_counter()
            dense, sparse = out.dense[0], out.sparse[0]
            level, conf = int(out.level[0]), float(out.confidence[0])
            ep = self.memory.fire(dense, c.memory_threshold, c.memory_min_strength)
            path = "gate"
            if ep is not None:
                level, path, conf = 0, "reflex-arc", max(conf, ep[1])
            elif level == 0:
                level, path = 1, "reflex-miss"
            k_eff = k or (c.ks[level] if conf >= c.low_confidence else min(c.ks[level] * 2, 20))
            ranking, raw_sparse, raw_dense = self._rank(max(level, 1), dense, sparse, max(k_eff, 6))
            if ep is not None:
                ranking = [(ep[0], 9.99)] + [r for r in ranking if r[0] != ep[0]]
            t2 = time.perf_counter()
            vocab = self.vocab; typed = set(content_tokens(query))
            nz = np.nonzero(sparse)[0]; nz = nz[np.argsort(-sparse[nz])]
            weights = {vocab.terms[j]: float(sparse[j]) for j in nz}
            top = self.chunks[ranking[0][0]] if ranking else None
            matched = len(set(content_tokens(top.model_text)) & set(weights)) if top else 0
            unsure = (level == 2 and raw_dense < 0.3 and matched == 0) or (raw_sparse == 0 and raw_dense < 0.25)
            mid = self._next_id; self._next_id += 1
            self.msgs[mid] = {"q": query, "dense": dense, "sparse": sparse, "level": level,
                              "ranking": [i for i, _ in ranking], "top": ranking[0][0] if ranking else None}
            while len(self.msgs) > 1000:
                self.msgs.pop(next(iter(self.msgs)))
            self.stats["questions"] += 1
            return {"id": mid, "query": query, "ranking": ranking, "k": k_eff, "weights": weights, "typed": typed,
                    "meta": {"level": level, "level_name": LEVELS[level][0], "level_label": LEVELS[level][1],
                             "path": path, "k": k_eff, "confidence": round(conf, 4),
                             "stop": [round(float(x), 4) for x in out.stop_probs[0]], "unsure": bool(unsure),
                             "spikes": [[vocab.terms[j], round(float(sparse[j]), 3), vocab.terms[j] in typed] for j in nz[:24]],
                             "ms": {"forward": round((t1 - t0) * 1e3, 3), "retrieve": round((t2 - t1) * 1e3, 3)}}}

    def _pack(self, i, s, weights, typed, full=False):
        ch = self.chunks[i]; expanded = {t for t in weights if t not in typed}
        text = best_window(ch, weights, 70 if full else 40)
        d = {"chunk": i, "score": round(s, 4), "title": ch.title, "source": ch.source, "locator": ch.locator,
             "kind": ch.kind, "text": text, "segments": highlight(text, typed, expanded)}
        if ch.kind == "row":
            d["fields"] = [[k, highlight(str(v), typed, expanded)] for k, v in ch.fields.items() if str(v).strip()]
            d["record"] = {k: v for k, v in ch.fields.items() if str(v).strip()}
        if full:
            d["context"] = highlight(ch.text, typed, expanded)
        return d

    def ask(self, question: str, k: int | None = None) -> dict:
        """search() + extractive answer quoted from the data, with citation and alternatives."""
        self._await_first_model()
        with self.lock:
            r = self.search(question, k)
            rk, w, ty = r["ranking"], r["weights"], r["typed"]
            if not rk:
                raise NotFound("no chunks indexed")
            return {"id": r["id"], "answer": self._pack(*rk[0], w, ty, full=True),
                    "hits": [self._pack(i, s, w, ty) for i, s in rk[1:r["k"] + 4]], "meta": r["meta"]}

    # ================================================================ learning
    def feedback(self, message_id: int, verdict: str, chunk: int | None = None, actor: str = "human") -> dict:
        """verdict: 'up' (top answer right) | 'down' (wrong) | 'pick' (this other chunk is right)."""
        with self.lock:
            m = self.msgs.get(int(message_id))
            if m is None:
                raise NotFound("پیام در حافظهٔ کوتاه‌مدت نیست / unknown or expired message id")
            if verdict not in ("up", "down", "pick"):
                raise RakhshError("verdict must be up | down | pick")
            if verdict == "pick" and (chunk is None or not 0 <= int(chunk) < len(self.chunks)):
                raise RakhshError("pick requires a valid chunk id")
            target = m["top"] if verdict == "up" else (int(chunk) if verdict == "pick" else None)
            return self._learn(m["q"], m["dense"], m["sparse"], m["top"], verdict, target, actor)

    def teach(self, question: str, answer: str | None = None, chunk: int | None = None,
              actor: str = "human") -> dict:
        """Directly teach 'question -> answer'. The answer is matched to the best chunk; if no chunk
        contains it, it is added as new knowledge first (source 'taught')."""
        with self.lock:
            self._require_ready()
            if chunk is None:
                if not answer:
                    raise RakhshError("teach needs `answer` text or a `chunk` id")
                chunk = self._locate(answer)
                if chunk is None:
                    self.ingest_text(f"taught-by-{actor.replace(':', '-')}", answer, train=False, actor=actor)
                    chunk = len(self.chunks) - 1
                    self.train_async()          # new words need spike neurons → background retrain
            out = self._encode_queries(self.model, self.vocab, [question])
            r = self._learn(question, out.dense[0], out.sparse[0], None, "pick", int(chunk), actor)
            self.stats["taught"] += 1
            r["chunk"] = int(chunk); r["locator"] = self.chunks[int(chunk)].locator
            return r

    def _locate(self, text: str) -> int | None:
        out = self._encode_queries(self.model, self.vocab, [text])
        rk, raw, _ = self._rank(2, out.dense[0], out.sparse[0], 1)
        if not rk:
            return None
        i = rk[0][0]; want = set(content_tokens(text))
        overlap = len(want & set(content_tokens(self.chunks[i].model_text))) / max(1, len(want))
        return i if overlap >= 0.6 else None

    def _learn(self, q, dense, sparse, top, verdict, target, actor) -> dict:
        c = self.cfg; trust = self.trust.of(actor); learned = []
        if verdict in ("up", "pick"):
            self.synapses.update(target, sparse, +1, c.hebb_lr * trust)
            learned.append("hebbian: query spikes wired to the correct chunk")
            if top is not None and target != top:
                self.synapses.update(top, sparse, -1, c.hebb_lr * trust)
                self.memory.reinforce(q, None, top, -trust)
                learned.append("anti-hebbian: path to the wrong chunk inhibited")
            self.memory.reinforce(q, dense, target, trust)
            j = self.memory.texts.index(q) if q in self.memory.texts else -1
            fires = j >= 0 and self.memory.strength[j] >= c.memory_min_strength - 1e-9
            learned.append("episodic neuron " + ("active: next time → reflex (level 0)" if fires
                                                 else f"strengthened ({self.memory.strength[j]:.2f}/{c.memory_min_strength})"))
            self.replay.append([q, int(target), 0, trust])
            key = "up" if verdict == "up" else "picks"; self.stats[key] += 1
            self.stats["recent"].append(1 if verdict == "up" else 0)
            slow = trust >= c.slow_learning_min_trust
        else:
            self.synapses.update(top, sparse, -1, c.hebb_lr * trust)
            self.memory.reinforce(q, None, top, -trust)
            self.replay = [r for r in self.replay if not (r[0] == q and r[1] == top and r[2] == 0)]
            self.replay.append([q, int(top), 2, trust])        # gate: this needs a deeper search
            learned.append("inhibited + gate learns this kind of question needs a deeper search")
            self.stats["down"] += 1; self.stats["recent"].append(0)
            slow = False
        if not self.training:
            if slow:
                self._online([r for r in self.replay if r[2] == 0][-8:])
                learned.append(f"{c.online_steps} slow gradient steps with replay")
            acc = self._fit_gate(self.model, 80)
            if acc is not None:
                self.stats["gate_acc"] = acc
            learned.append("hazard gate updated")
        self.on_event(brain=self.name, actor=actor, action="feedback", verdict=verdict, trust=trust,
                      query=q[:200], chunk=target if target is not None else top)
        self._save_soon()
        return {"ok": True, "trust": trust, "learned": learned}

    def _online(self, examples, steps=None):
        if not examples:
            return
        c = self.cfg; model = self.model
        if getattr(self, "_opt_model", None) is not model:
            self._opt = Adam(model.parameters(), lr=c.lr / 3, weight_decay=c.weight_decay); self._opt_model = model
        pq = PseudoQueries(self.chunks[:self.n_indexed], seed=int(self.rng.integers(1 << 30)))
        for _ in range(steps or c.online_steps):
            self._step(model, self.vocab, self._opt, pq, self.feat_rows, self.lex_rows, c.flops_lambda, examples, 0.4)
        self.dense_vecs = self._encode_chunks(model, self.feat_rows)
        self.memory.reencode(lambda t: self._encode_queries(model, self.vocab, t).dense)

    def sleep(self, actor: str = "human") -> dict:
        """Consolidation: replay confirmed episodes into slow weights, refit gate, decay fast synapses."""
        with self.lock:
            self._require_ready()
            pos = [r for r in self.replay if r[2] == 0 and r[3] >= self.cfg.slow_learning_min_trust]
            before = len(self.synapses)
            if pos:
                self._online(pos, self.cfg.sleep_steps)
            acc = self._fit_gate(self.model, self.cfg.gate_steps)
            self.synapses.decay(self.cfg.sleep_decay)
            self.stats["sleeps"] += 1; self.stats["gate_acc"] = acc
        self.on_event(brain=self.name, actor=actor, action="sleep", replayed=len(pos))
        self.save()
        return {"ok": True, "replayed": len(pos), "synapses_before": before,
                "synapses_after": len(self.synapses), "gate_acc": acc}

    # ================================================================ state & persistence
    def state(self) -> dict:
        with self.lock:
            rec = list(self.stats["recent"])
            return {"name": self.name, "status": dict(self.status), "sources": self.sources,
                    "chunks": len(self.chunks), "indexed": self.n_indexed,
                    "neurons": len(self.vocab) if self.vocab else 0,
                    "params": self.model.n_params() if self.model else 0,
                    "synapses": len(self.synapses), "episodes": len(self.memory), "replay": len(self.replay),
                    "stats": {k: v for k, v in self.stats.items() if k != "recent"},
                    "satisfaction": (sum(rec) / len(rec)) if rec else None}

    workdir = None      # set by Registry; when None the brain is in-memory only

    def save(self) -> None:
        if self.workdir is None:
            return
        from .storage import save_brain
        with self.lock:
            save_brain(self, self.workdir)

    def _save_soon(self):
        if self.workdir is not None:
            threading.Thread(target=self.save, daemon=True).start()

    def reset(self, actor: str = "human") -> None:
        with self.lock:
            self.wait(); self._clear(); self.msgs.clear()
        self.on_event(brain=self.name, actor=actor, action="reset")
        self.save()
