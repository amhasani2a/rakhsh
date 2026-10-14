"""A tiny reverse-mode autograd engine on numpy (no torch dependency).

Only the ops this project needs, each with an analytic backward. Fused ops
(cross_entropy, hazard_nll, l2_normalize) are numerically stable.
"""
from __future__ import annotations
import numpy as np


class Tensor:
    __slots__ = ("data", "grad", "requires_grad", "_parents", "_backward")

    def __init__(self, data, requires_grad: bool = False, _parents=(), _backward=None):
        self.data = np.asarray(data)
        self.grad = None
        self.requires_grad = requires_grad
        self._parents = _parents
        self._backward = _backward

    shape = property(lambda self: self.data.shape)

    def _acc(self, g):
        if self.requires_grad:
            self.grad = g if self.grad is None else self.grad + g

    def zero_grad(self):
        self.grad = None

    def backward(self, grad=None):
        topo, seen, stack = [], set(), [(self, False)]
        while stack:                         # iterative topological sort
            node, done = stack.pop()
            if done:
                topo.append(node); continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            stack.extend((p, False) for p in node._parents)
        self.grad = np.ones_like(self.data) if grad is None else grad
        for node in reversed(topo):
            if node._backward is not None and node.grad is not None:
                node._backward(node.grad)

    # sugar
    def __add__(self, o): return add(self, o)
    def __sub__(self, o): return add(self, scale(as_tensor(o), -1.0))
    def __mul__(self, o): return mul(self, o)
    def __matmul__(self, o): return matmul(self, o)
    @property
    def T(self): return transpose(self)


def as_tensor(x) -> Tensor:
    return x if isinstance(x, Tensor) else Tensor(x)


def _node(data, parents, backward) -> Tensor:
    rg = any(p.requires_grad for p in parents)
    return Tensor(data, rg, parents if rg else (), backward if rg else None)


def _unbroadcast(g, shape):
    while g.ndim > len(shape):
        g = g.sum(0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(i, keepdims=True)
    return g


def add(a, b):
    a, b = as_tensor(a), as_tensor(b)
    def bw(g):
        a._acc(_unbroadcast(g, a.shape)); b._acc(_unbroadcast(g, b.shape))
    return _node(a.data + b.data, (a, b), bw)


def mul(a, b):
    a, b = as_tensor(a), as_tensor(b)
    def bw(g):
        a._acc(_unbroadcast(g * b.data, a.shape)); b._acc(_unbroadcast(g * a.data, b.shape))
    return _node(a.data * b.data, (a, b), bw)


def scale(a: Tensor, c: float):
    return _node(a.data * c, (a,), lambda g: a._acc(g * c))


def matmul(a, b):
    a, b = as_tensor(a), as_tensor(b)
    def bw(g):
        a._acc(g @ b.data.T); b._acc(a.data.T @ g)
    return _node(a.data @ b.data, (a, b), bw)


def transpose(a: Tensor):
    return _node(a.data.T, (a,), lambda g: a._acc(g.T))


def tanh(a: Tensor):
    y = np.tanh(a.data)
    return _node(y, (a,), lambda g: a._acc(g * (1 - y * y)))


def relu(a: Tensor):
    m = a.data > 0
    return _node(a.data * m, (a,), lambda g: a._acc(g * m))


def log1p(a: Tensor):
    return _node(np.log1p(a.data), (a,), lambda g: a._acc(g / (1 + a.data)))


def square(a: Tensor):
    return _node(a.data ** 2, (a,), lambda g: a._acc(2 * g * a.data))


def sum_(a: Tensor, axis=None):
    def bw(g):
        if axis is not None:
            g = np.expand_dims(g, axis)
        a._acc(np.broadcast_to(g, a.shape).copy())
    return _node(a.data.sum(axis=axis), (a,), bw)


def mean(a: Tensor, axis=None):
    n = a.data.size if axis is None else a.shape[axis]
    return scale(sum_(a, axis), 1.0 / n)


def concat_rows(ts: list[Tensor]):
    sizes = np.cumsum([0] + [t.shape[0] for t in ts])
    def bw(g):
        for t, s, e in zip(ts, sizes[:-1], sizes[1:]):
            t._acc(g[s:e])
    return _node(np.concatenate([t.data for t in ts], 0), tuple(ts), bw)


def slice_(a: Tensor, rows=slice(None), cols=slice(None)):
    def bw(g):
        full = np.zeros_like(a.data); full[rows, cols] = g; a._acc(full)
    return _node(a.data[rows, cols], (a,), bw)


def l2_normalize(a: Tensor, eps: float = 1e-8):
    n = np.linalg.norm(a.data, axis=1, keepdims=True) + eps
    y = a.data / n
    return _node(y, (a,), lambda g: a._acc((g - y * (g * y).sum(1, keepdims=True)) / n))


def cross_entropy(logits: Tensor, targets: np.ndarray, weights: np.ndarray | None = None):
    """Mean (optionally sample-weighted) softmax cross-entropy."""
    z = logits.data - logits.data.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    b = len(targets); idx = np.arange(b)
    w = np.full(b, 1.0 / b) if weights is None else np.asarray(weights, float) / np.sum(weights)
    loss = -(w * np.log(p[idx, targets] + 1e-12)).sum()
    def bw(g):
        d = p.copy(); d[idx, targets] -= 1; logits._acc(g * d * w[:, None])
    return _node(np.array(loss), (logits,), bw)


def _softplus(x):
    return np.logaddexp(0, x)


def hazard_nll(logits: Tensor, levels: np.ndarray, weights: np.ndarray | None = None):
    """Discrete-time hazard likelihood.

    logits: (B, L-1) -> hazard h_l = sigmoid(z_l) = P(stop at level l | reached l).
    A query with label y 'survived' levels < y and 'fired' at y (if y < L-1).
    """
    z = logits.data; b, m = z.shape
    ar = np.arange(m)[None, :]
    before = (ar < levels[:, None]).astype(z.dtype)
    at = (ar == levels[:, None]).astype(z.dtype)
    w = np.full(b, 1.0 / b) if weights is None else np.asarray(weights, float) / np.sum(weights)
    loss = (w[:, None] * (before * _softplus(z) + at * _softplus(-z))).sum()
    h = 1 / (1 + np.exp(-z))
    return _node(np.array(loss), (logits,), lambda g: logits._acc(g * (before * h + at * (h - 1)) * w[:, None]))
