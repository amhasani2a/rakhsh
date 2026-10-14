"""Typed configuration for every Rakhsh component (plain dataclasses, JSON-serialisable)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any


@dataclass
class FeaturizerConfig:
    """Open-vocabulary hashed input layer (words + char n-grams)."""
    dim: int = 4096
    char_ngrams: tuple = (2, 3, 4)
    use_words: bool = True


@dataclass
class ModelConfig:
    """ReflexNet: shared trunk + one fused head matmul (dense | sparse spikes | hazard gate)."""
    hidden: int = 192
    dense_dim: int = 64
    levels: int = 3                       # 0 = reflex/memory, 1 = spike index, 2 = deep hybrid
    spike_rel_threshold: float = 0.15     # 'action potential': silence spikes < ratio * max
    max_spikes: int = 32
    hazard_threshold: float = 0.5         # first level whose hazard crosses this fires ("jump")
    lexical_skip: float = 1.0             # typed words always fire; the net learns the expansion


@dataclass
class TrustPolicy:
    """How much a feedback/teaching signal is believed, by actor.

    Actors are strings: ``"human"`` or ``"agent:<name>"``. Signals are scaled by trust;
    slow (gradient) learning and instant episodic memory require enough accumulated trust,
    so one misbehaving agent cannot poison a brain on its own.
    """
    human: float = 1.0
    default_agent: float = 0.5
    agents: dict = field(default_factory=dict)       # {"claude": 0.8, "cursor": 0.6}

    def of(self, actor: str) -> float:
        if actor in ("human", "", None):
            return self.human
        name = actor.split(":", 1)[-1]
        return float(self.agents.get(name, self.default_agent))


@dataclass
class BrainConfig:
    featurizer: FeaturizerConfig = field(default_factory=FeaturizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    # ingestion
    max_vocab: int = 20000
    chunk_words: int = 90
    # self-supervised training
    steps_min: int = 300
    steps_max: int = 900
    steps_per_chunk: float = 0.5
    batch: int = 48
    negatives: int = 48
    lr: float = 3e-3
    weight_decay: float = 1e-5
    dense_temperature: float = 0.05
    sparse_scale: float = 4.0
    flops_lambda: float = 3e-3
    input_dropout: float = 0.1
    gate_samples: int = 500
    gate_steps: int = 300
    # plasticity
    online_steps: int = 6
    sleep_steps: int = 120
    hebb_lr: float = 0.6
    hebb_weight: float = 1.0
    sleep_decay: float = 0.5
    memory_threshold: float = 0.93
    memory_min_strength: float = 1.0      # trust that must accumulate before a reflex neuron fires
    slow_learning_min_trust: float = 0.75
    # retrieval policy per level
    ks: tuple = (1, 5, 10)
    dense_weight: float = 0.6
    low_confidence: float = 0.6
    seed: int = 7


@dataclass
class DeciderConfig:
    """General-purpose few-shot decision layer (router / gate) with user-defined labels."""
    featurizer: FeaturizerConfig = field(default_factory=FeaturizerConfig)
    hidden: int = 64
    lr: float = 0.02
    fit_steps: int = 200
    online_steps: int = 40
    abstain_below: float = 0.55
    proto_weight: float = 0.5             # episodic (nearest-example) vote blended with the net
    proto_temperature: float = 0.08
    seed: int = 11


# ------------------------------------------------------------------ (de)serialisation
def to_dict(obj) -> dict:
    return asdict(obj)


def from_dict(cls, d: dict | None):
    """Rebuild a (nested) dataclass from a dict, ignoring unknown keys (forward compatible)."""
    if d is None:
        return cls()
    kw: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in d:
            continue
        v = d[f.name]
        typ = f.type if not isinstance(f.type, str) else globals().get(f.type)
        default = f.default_factory() if callable(f.default_factory) else f.default  # type: ignore[misc]
        if is_dataclass(default):
            v = from_dict(type(default), v)
        elif isinstance(default, tuple):
            v = tuple(v)
        kw[f.name] = v
    return cls(**kw)
