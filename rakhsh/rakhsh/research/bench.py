"""Reproducible benchmark on the bundled Persian movie dataset (BM25 vs Rakhsh variants)."""
from __future__ import annotations

import pathlib

from .agent import ReflexSearchAgent
from .config import Config
from .data import load_corpus, load_queries
from .evaluate import evaluate, format_report
from .trainer import Trainer


def run(data_dir: str | pathlib.Path, steps: int | None = None, verbose: bool = True) -> dict:
    data = pathlib.Path(data_dir); cfg = Config()
    if steps:
        cfg.train.steps = steps
    docs = load_corpus(data / "corpus.jsonl")
    tr = Trainer(cfg, docs, load_queries(data / "train_queries.jsonl")); tr.fit(verbose=verbose)
    agent = ReflexSearchAgent(tr.model, tr.feat, tr.vocab, docs, cfg)
    rep = evaluate(agent, load_queries(data / "test_queries.jsonl"))
    rep["markdown"] = format_report(rep)
    return rep
