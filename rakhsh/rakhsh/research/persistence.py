"""Save / load a trained agent bundle (weights .npz + config/vocab .json)."""
from __future__ import annotations
import json
import pathlib

import numpy as np

from .config import Config
from ..core.model import ReflexNet
from ..core.text import HashedFeaturizer, Vocab


def save(path, model: ReflexNet, vocab: Vocab, cfg: Config):
    p = pathlib.Path(path); p.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p / "weights.npz", **model.state_dict())
    (p / "meta.json").write_text(json.dumps({"config": cfg.to_dict(), "vocab": vocab.to_dict()}, ensure_ascii=False), "utf-8")


def load(path) -> tuple[ReflexNet, HashedFeaturizer, Vocab, Config]:
    p = pathlib.Path(path)
    meta = json.loads((p / "meta.json").read_text("utf-8"))
    cfg, vocab = Config.from_dict(meta["config"]), Vocab.from_dict(meta["vocab"])
    model = ReflexNet(cfg.featurizer.dim, len(vocab), cfg.model)
    model.load_state_dict(dict(np.load(p / "weights.npz")))
    return model, HashedFeaturizer(cfg.featurizer), vocab, cfg
