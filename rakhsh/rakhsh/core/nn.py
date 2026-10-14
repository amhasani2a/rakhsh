"""Minimal module system + Adam optimiser on top of the autograd engine."""
from __future__ import annotations
import numpy as np

from .autograd import Tensor, matmul, add


class Module:
    def parameters(self) -> dict[str, Tensor]:
        out = {}
        for name, v in vars(self).items():
            if isinstance(v, Tensor) and v.requires_grad:
                out[name] = v
            elif isinstance(v, Module):
                out.update({f"{name}.{k}": p for k, p in v.parameters().items()})
        return out

    def state_dict(self) -> dict[str, np.ndarray]:
        return {k: p.data for k, p in self.parameters().items()}

    def load_state_dict(self, sd: dict[str, np.ndarray]):
        for k, p in self.parameters().items():
            assert p.data.shape == sd[k].shape, k
            p.data = sd[k].astype(p.data.dtype)


class Linear(Module):
    def __init__(self, n_in: int, n_out: int, rng: np.random.Generator, bias_init: float = 0.0):
        lim = np.sqrt(6 / (n_in + n_out))
        self.W = Tensor(rng.uniform(-lim, lim, (n_in, n_out)).astype(np.float32), requires_grad=True)
        self.b = Tensor(np.full((n_out,), bias_init, np.float32), requires_grad=True)

    def __call__(self, x: Tensor) -> Tensor:
        return add(matmul(x, self.W), self.b)

    def np(self, x: np.ndarray) -> np.ndarray:      # fast inference path, no graph
        return x @ self.W.data + self.b.data


class Adam:
    def __init__(self, params: dict[str, Tensor], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.p = list(params.values()); self.lr, self.b1, self.b2, self.eps, self.wd = lr, *betas, eps, weight_decay
        self.m = [np.zeros_like(p.data) for p in self.p]; self.v = [np.zeros_like(p.data) for p in self.p]; self.t = 0

    def zero_grad(self):
        for p in self.p:
            p.zero_grad()

    def step(self):
        self.t += 1
        for p, m, v in zip(self.p, self.m, self.v):
            if p.grad is None:
                continue
            g = p.grad + self.wd * p.data
            m *= self.b1; m += (1 - self.b1) * g
            v *= self.b2; v += (1 - self.b2) * g * g
            mh = m / (1 - self.b1 ** self.t); vh = v / (1 - self.b2 ** self.t)
            p.data -= (self.lr * mh / (np.sqrt(vh) + self.eps)).astype(p.data.dtype)
