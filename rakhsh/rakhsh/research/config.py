"""Configuration of the original research benchmark (movie dataset)."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict

from ..config import FeaturizerConfig, ModelConfig


@dataclass
class TrainConfig:
    steps: int = 800
    batch_size: int = 48
    lr: float = 3e-3
    weight_decay: float = 1e-5
    dense_temperature: float = 0.05
    sparse_scale: float = 4.0    # logit scale for the sparse contrastive loss
    flops_lambda: float = 3e-3   # sparsity pressure (SPLADE-style FLOPS regulariser)
    flops_warmup: int = 200
    gate_lambda: float = 1.0
    input_dropout: float = 0.1
    seed: int = 7
    log_every: int = 100


@dataclass
class LevelPolicy:
    name: str
    k: int
    use_cache: bool
    use_sparse: bool
    use_dense: bool


@dataclass
class AgentConfig:
    policies: list = field(default_factory=lambda: [
        LevelPolicy("reflex", k=1, use_cache=True, use_sparse=True, use_dense=False),
        LevelPolicy("sparse", k=5, use_cache=True, use_sparse=True, use_dense=False),
        LevelPolicy("deep", k=10, use_cache=False, use_sparse=True, use_dense=True),
    ])
    low_confidence: float = 0.6   # if gate confidence < this, widen K (x2)
    max_k: int = 20
    cache_capacity: int = 256
    cache_similarity: float = 0.97
    dense_weight: float = 0.6     # score fusion weight at the deep level (no re-ranker)


@dataclass
class Config:
    featurizer: FeaturizerConfig = field(default_factory=FeaturizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        f = FeaturizerConfig(**{**d["featurizer"], "char_ngrams": tuple(d["featurizer"]["char_ngrams"])})
        a = dict(d["agent"]); a["policies"] = [LevelPolicy(**p) for p in a["policies"]]
        return cls(f, ModelConfig(**d["model"]), TrainConfig(**d["train"]), AgentConfig(**a))
