"""Shared fixtures: a small, fast config and one trained brain reused across test modules."""
import pathlib, tempfile
from rakhsh import BrainConfig, Registry, TrustPolicy

ROOT = pathlib.Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"


def fast_cfg() -> BrainConfig:
    c = BrainConfig()
    c.steps_min, c.steps_max, c.gate_steps = 120, 160, 80
    return c


_CACHE = {}


def trained_registry():
    """One registry + trained 'lib' brain per test run (training is the slow part)."""
    if "reg" not in _CACHE:
        home = tempfile.mkdtemp(prefix="rakhsh-test-")
        reg = Registry(home, trust=TrustPolicy(agents={"claude": 0.8, "spam": 0.1}), brain_cfg=fast_cfg())
        b = reg.brain("lib")
        for f in ("cinema_book.txt", "library.sqlite", "notes.json"):
            b.ingest(f, (SAMPLES / f).read_bytes(), train=False)
        b.train()
        _CACHE["reg"] = reg
    return _CACHE["reg"]
