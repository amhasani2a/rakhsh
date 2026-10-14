"""Safe, versioned persistence (JSON + NPZ, no pickle → loading a brain never executes code).

Layout of a brain directory::

    meta.json        format version, config, sources, stats, status
    chunks.jsonl     the knowledge
    vocab.json       spike-neuron vocabulary + idf
    model.npz        ReflexNet weights
    plastic.json     fast synapses, episodic memory, replay buffer, gate set
Derived structures (hashed features, indexes, dense vectors) are rebuilt on load.
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import time

import numpy as np

from .config import BrainConfig, from_dict, to_dict
from .core.model import ReflexNet
from .core.plasticity import FastSynapses, ReflexMemory
from .core.text import Vocab
from .ingest import Chunk

FORMAT = 1


def _atomic_write(path: pathlib.Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data); os.replace(tmp, path)


def save_brain(brain, directory) -> None:
    d = pathlib.Path(directory); d.mkdir(parents=True, exist_ok=True)
    stats = {k: (list(v) if k == "recent" else v) for k, v in brain.stats.items()}
    meta = {"format": FORMAT, "name": brain.name, "saved_at": time.time(), "config": to_dict(brain.cfg),
            "sources": brain.sources, "stats": stats, "status": brain.status, "next_id": brain._next_id}
    _atomic_write(d / "meta.json", json.dumps(meta, ensure_ascii=False, indent=1).encode())
    _atomic_write(d / "chunks.jsonl", "\n".join(json.dumps(c.to_json(), ensure_ascii=False) for c in brain.chunks).encode())
    if brain.model is not None:
        _atomic_write(d / "vocab.json", json.dumps(brain.vocab.to_dict(), ensure_ascii=False).encode())
        tmp = d / "model.tmp.npz"; np.savez_compressed(tmp, **brain.model.state_dict()); os.replace(tmp, d / "model.npz")
    else:
        for f in ("vocab.json", "model.npz"):
            (d / f).unlink(missing_ok=True)
    plastic = {"synapses": brain.synapses.to_json(), "memory": brain.memory.to_json(), "replay": brain.replay,
               "gate_set": {"texts": list(brain.gate_set[0]), "labels": np.asarray(brain.gate_set[1]).tolist()}}
    _atomic_write(d / "plastic.json", json.dumps(plastic, ensure_ascii=False).encode())


def load_brain(directory, brain_cls, **kw):
    d = pathlib.Path(directory)
    meta = json.loads((d / "meta.json").read_text("utf-8"))
    if meta.get("format", 0) > FORMAT:
        raise RuntimeError(f"brain format {meta['format']} is newer than this rakhsh version")
    b = brain_cls(meta["name"], from_dict(BrainConfig, meta["config"]), **kw)
    b.sources = meta["sources"]; b._next_id = meta.get("next_id", 1)
    for k, v in meta["stats"].items():
        if k == "recent":
            b.stats["recent"].extend(v)
        else:
            b.stats[k] = v
    text = (d / "chunks.jsonl").read_text("utf-8")
    b.chunks = [Chunk.from_json(json.loads(l)) for l in text.splitlines() if l.strip()]
    pl = json.loads((d / "plastic.json").read_text("utf-8")) if (d / "plastic.json").exists() else {}
    if pl:
        b.synapses = FastSynapses.from_json(pl["synapses"]); b.memory = ReflexMemory.from_json(pl["memory"])
        b.replay = [list(r) for r in pl["replay"]]
        b.gate_set = (pl["gate_set"]["texts"], np.array(pl["gate_set"]["labels"], int))
    if (d / "model.npz").exists():
        b.vocab = Vocab.from_dict(json.loads((d / "vocab.json").read_text("utf-8")))
        b.model = ReflexNet(b.cfg.featurizer.dim, len(b.vocab), b.cfg.model)
        with np.load(d / "model.npz") as z:
            b.model.load_state_dict({k: z[k] for k in z.files})
        texts = [c.model_text for c in b.chunks]
        b.feat_rows = [b.feat.sparse(t) for t in texts]; b.lex_rows = [b.vocab.lexical_sparse(t) for t in texts]
        from .core.index import InvertedIndex
        b.sparse_index = InvertedIndex.from_rows(b.lex_rows)
        b.dense_vecs = b._encode_chunks(b.model, b.feat_rows); b.n_indexed = len(b.chunks)
        b.memory.reencode(lambda t: b._encode_queries(b.model, b.vocab, t).dense)
        b.status = {"state": "ready", "progress": 1.0, "message": f"بازیابی شد ✓ restored · {len(b.chunks)} chunks"}
    elif b.chunks:
        b.status = {"state": "idle", "progress": 0.0, "message": "داده هست ولی آموزش ندیده / data present, not trained"}
    return b


def snapshot(directory, label: str | None = None) -> str:
    """Copy a saved brain directory into ./snapshots/<timestamp>[-label]. Returns the snapshot id."""
    d = pathlib.Path(directory)
    sid = time.strftime("%Y%m%d-%H%M%S") + (f"-{label}" if label else "")
    dst = d / "snapshots" / sid; dst.mkdir(parents=True, exist_ok=False)
    for f in d.iterdir():
        if f.is_file():
            shutil.copy2(f, dst / f.name)
    return sid


def list_snapshots(directory) -> list[str]:
    p = pathlib.Path(directory) / "snapshots"
    return sorted(x.name for x in p.iterdir()) if p.exists() else []


def restore_files(directory, sid: str) -> None:
    d = pathlib.Path(directory); src = d / "snapshots" / sid
    if not src.exists():
        raise FileNotFoundError(sid)
    for f in src.iterdir():
        shutil.copy2(f, d / f.name)
    for f in ("vocab.json", "model.npz"):
        if not (src / f).exists():
            (d / f).unlink(missing_ok=True)
