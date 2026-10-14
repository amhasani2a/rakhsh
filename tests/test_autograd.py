import unittest
import numpy as np
from rakhsh.core import autograd as ag


def numgrad(f, x, eps=1e-6):
    g = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        old = x[i]; x[i] = old + eps; a = f(); x[i] = old - eps; b = f(); x[i] = old; g[i] = (a - b) / (2 * eps)
    return g


class TestGradients(unittest.TestCase):
    def check(self, build, *shapes):
        rng = np.random.default_rng(0)
        ts = [ag.Tensor(rng.normal(size=s), requires_grad=True) for s in shapes]
        build(*ts).backward()
        for t in ts:
            n = numgrad(lambda: float(build(*[ag.Tensor(u.data) for u in ts]).data), t.data)
            np.testing.assert_allclose(t.grad, n, rtol=1e-4, atol=1e-6)

    def test_matmul_tanh(self):
        self.check(lambda a, b: ag.sum_(ag.tanh(a @ b)), (3, 4), (4, 2))

    def test_add_broadcast(self):
        self.check(lambda a, b: ag.sum_(ag.square(a + b)), (3, 4), (4,))

    def test_l2norm_ce(self):
        y = np.array([0, 2, 1])
        self.check(lambda a, b: ag.cross_entropy(ag.l2_normalize(a) @ ag.l2_normalize(b).T, y), (3, 5), (4, 5))

    def test_sparse_activation_flops(self):
        self.check(lambda a: ag.sum_(ag.square(ag.mean(ag.log1p(ag.relu(a)), 0))), (4, 6))

    def test_hazard_nll(self):
        lv = np.array([0, 1, 2, 2])
        self.check(lambda a: ag.hazard_nll(a, lv), (4, 2))

    def test_slice_concat(self):
        self.check(lambda a, b: ag.sum_(ag.square(ag.slice_(ag.concat_rows([a, b]), rows=slice(1, 4), cols=slice(0, 2)))), (2, 3), (3, 3))


if __name__ == "__main__":
    unittest.main()
