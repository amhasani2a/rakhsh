"""Benchmark: Reflex agent vs BM25 / dense-only / sparse-only / always-deep hybrid."""
from __future__ import annotations
import time
from collections import defaultdict

import numpy as np

from .agent import ReflexSearchAgent
from .data import Query
from ..core.index import BM25, score_fusion


def _metrics(rank_lists: list[list[int]], gold: list[int]) -> dict:
    r1 = r5 = mrr = 0.0
    for r, g in zip(rank_lists, gold):
        if g in r[:1]: r1 += 1
        if g in r[:5]: r5 += 1
        if g in r[:10]: mrr += 1 / (r.index(g) + 1)
    n = max(1, len(gold))
    return {"R@1": r1 / n, "R@5": r5 / n, "MRR@10": mrr / n}


def evaluate(agent: ReflexSearchAgent, queries: list[Query]) -> dict:
    idx = {d.id: i for i, d in enumerate(agent.docs)}
    gold = [idx[q.doc_id] for q in queries]
    bm25 = BM25([d.full_text for d in agent.docs])
    systems: dict[str, list[list[int]]] = defaultdict(list)
    lat: dict[str, list[float]] = defaultdict(list)
    gate_pred, ks, spikes, expanded = [], [], [], []
    agent.cache.clear()

    for q in queries:
        t = time.perf_counter(); r = bm25.search(q.query, 10); lat["bm25"].append(time.perf_counter() - t)
        systems["bm25"].append([i for i, _ in r])

        out = agent.encode([q.query])
        s = agent.sparse_index.search(out.sparse[0], 10); d = agent.dense_index.search(out.dense[0], 10)
        systems["reflex-sparse-only"].append([i for i, _ in s])
        systems["reflex-dense-only"].append([i for i, _ in d])
        systems["always-deep (hybrid)"].append([i for i, _ in score_fusion([s, d], [1 - agent.cfg.agent.dense_weight, agent.cfg.agent.dense_weight], 10)])

        t = time.perf_counter(); res = agent.search(q.query, use_cache=False); lat["reflex-agent"].append(time.perf_counter() - t)
        systems["reflex-agent (adaptive)"].append([idx[h.doc_id] for h in res.hits])
        gate_pred.append(res.level); ks.append(res.k); spikes.append(len(res.spikes))
        expanded.append(sum(1 for _, _, inq in res.spikes if not inq))

    levels = np.array([q.level for q in queries]); pred = np.array(gate_pred)
    L = agent.cfg.model.levels
    report = {"overall": {n: _metrics(r, gold) for n, r in systems.items()}, "by_level": {}}
    for lv in range(L):
        m = levels == lv
        report["by_level"][lv] = {n: _metrics([r[i] for i in np.nonzero(m)[0]], [gold[i] for i in np.nonzero(m)[0]])
                                  for n, r in systems.items()}
    report["gate"] = {"accuracy": float((pred == levels).mean()),
                      "confusion": [[int(((levels == a) & (pred == b)).sum()) for b in range(L)] for a in range(L)]}
    report["efficiency"] = {"avg_K": float(np.mean(ks)), "avg_spikes": float(np.mean(spikes)),
                            "avg_expanded_terms": float(np.mean(expanded)),
                            "vocab_size": len(agent.vocab),
                            "latency_ms_p50": {k: float(np.median(v) * 1e3) for k, v in lat.items()}}
    return report


def format_report(rep: dict) -> str:
    out = ["## Overall", "| system | R@1 | R@5 | MRR@10 |", "|---|---|---|---|"]
    out += [f"| {n} | {m['R@1']:.3f} | {m['R@5']:.3f} | {m['MRR@10']:.3f} |" for n, m in rep["overall"].items()]
    names = {0: "level 0 – navigational", 1: "level 1 – keyword", 2: "level 2 – abstract"}
    for lv, sysm in rep["by_level"].items():
        out += ["", f"## {names.get(int(lv), lv)} (MRR@10)", "| system | R@1 | MRR@10 |", "|---|---|---|"]
        out += [f"| {n} | {m['R@1']:.3f} | {m['MRR@10']:.3f} |" for n, m in sysm.items()]
    g, e = rep["gate"], rep["efficiency"]
    out += ["", f"## Hazard gate", f"accuracy = {g['accuracy']:.3f}", "confusion (rows=true, cols=pred):"]
    out += ["  " + str(row) for row in g["confusion"]]
    out += ["", "## Efficiency", f"avg K = {e['avg_K']:.2f} | avg active spikes = {e['avg_spikes']:.1f} / {e['vocab_size']} "
            f"| avg expanded (not-typed) terms = {e['avg_expanded_terms']:.1f}",
            "p50 latency (ms): " + ", ".join(f"{k}={v:.3f}" for k, v in e["latency_ms_p50"].items())]
    return "\n".join(out)
