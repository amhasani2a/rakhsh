"""Registry – a home directory holding many named brains and deciders, plus an audit log.

    home/
      brains/<name>/...          (see storage.py)
      deciders/<name>/...
      audit.jsonl                every ingest / feedback / teach / sleep, with actor + trust
"""
from __future__ import annotations

import json
import pathlib
import re
import shutil
import threading
import time

from .brain import Brain
from .config import BrainConfig, DeciderConfig, TrustPolicy
from .decide import Decider
from .errors import NotFound, RakhshError
from . import storage

_NAME = re.compile(r"^[\w\-\.]{1,64}$", re.UNICODE)


def _check(name: str) -> str:
    if not _NAME.match(name) or name.startswith("."):
        raise RakhshError(f"invalid name {name!r} (letters, digits, - _ . ; max 64)")
    return name


class AuditLog:
    def __init__(self, path: pathlib.Path):
        self.path = path; self._lock = threading.Lock()

    def __call__(self, **event):
        event = {"ts": round(time.time(), 3), **event}
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    def tail(self, n: int = 50) -> list[dict]:
        if not self.path.exists():
            return []
        lines = self.path.read_text("utf-8").splitlines()[-n:]
        return [json.loads(l) for l in lines if l.strip()]


class Registry:
    def __init__(self, home: str | pathlib.Path = "rakhsh_home", trust: TrustPolicy | None = None,
                 brain_cfg: BrainConfig | None = None, decider_cfg: DeciderConfig | None = None):
        self.home = pathlib.Path(home).expanduser().resolve()
        (self.home / "brains").mkdir(parents=True, exist_ok=True)
        (self.home / "deciders").mkdir(parents=True, exist_ok=True)
        self.trust = trust or TrustPolicy()
        self.brain_cfg, self.decider_cfg = brain_cfg, decider_cfg
        self.audit = AuditLog(self.home / "audit.jsonl")
        self._brains: dict[str, Brain] = {}; self._deciders: dict[str, Decider] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ brains
    def brain_names(self) -> list[str]:
        on_disk = {p.name for p in (self.home / "brains").iterdir() if (p / "meta.json").exists()}
        return sorted(on_disk | set(self._brains))

    def brain(self, name: str = "default", create: bool = True) -> Brain:
        _check(name)
        with self._lock:
            if name in self._brains:
                return self._brains[name]
            d = self.home / "brains" / name
            if (d / "meta.json").exists():
                b = storage.load_brain(d, Brain, trust=self.trust, on_event=self.audit)
            elif create:
                b = Brain(name, self.brain_cfg, trust=self.trust, on_event=self.audit)
                self.audit(brain=name, actor="system", action="create")
            else:
                raise NotFound(f"brain not found: {name}")
            b.workdir = d; self._brains[name] = b
            return b

    def delete_brain(self, name: str) -> None:
        with self._lock:
            b = self._brains.pop(name, None)
            if b:
                b.wait()
            shutil.rmtree(self.home / "brains" / _check(name), ignore_errors=True)
            self.audit(brain=name, actor="system", action="delete")

    def snapshot(self, name: str, label: str | None = None) -> str:
        b = self.brain(name, create=False); b.wait(); b.save()
        sid = storage.snapshot(b.workdir, label); self.audit(brain=name, actor="system", action="snapshot", id=sid)
        return sid

    def snapshots(self, name: str) -> list[str]:
        return storage.list_snapshots(self.home / "brains" / _check(name))

    def restore(self, name: str, sid: str) -> Brain:
        with self._lock:
            b = self._brains.pop(name, None)
            if b:
                b.wait()
            storage.restore_files(self.home / "brains" / _check(name), sid)
            self.audit(brain=name, actor="system", action="restore", id=sid)
            return self.brain(name, create=False)

    # ------------------------------------------------------------------ deciders
    def decider_names(self) -> list[str]:
        on_disk = {p.name for p in (self.home / "deciders").iterdir() if (p / "decider.json").exists()}
        return sorted(on_disk | set(self._deciders))

    def create_decider(self, name: str, labels: list[str], mode: str = "categorical", description: str = "",
                       exist_ok: bool = True) -> Decider:
        _check(name)
        with self._lock:
            if name in self.decider_names():
                if not exist_ok:
                    raise RakhshError(f"decider exists: {name}")
                d = self.decider(name)
                if d.labels != list(labels) or d.mode != mode:
                    raise RakhshError(f"decider {name} exists with labels {d.labels} / mode {d.mode}")
                return d
            d = Decider(name, labels, mode, description, self.decider_cfg, trust=self.trust, on_event=self.audit)
            d.workdir = self.home / "deciders" / name; d.save(); self._deciders[name] = d
            self.audit(decider=name, actor="system", action="create", labels=labels, mode=mode)
            return d

    def decider(self, name: str) -> Decider:
        _check(name)
        with self._lock:
            if name not in self._deciders:
                p = self.home / "deciders" / name
                if not (p / "decider.json").exists():
                    raise NotFound(f"decider not found: {name}")
                self._deciders[name] = Decider.load(p, trust=self.trust, on_event=self.audit)
            return self._deciders[name]

    def delete_decider(self, name: str) -> None:
        with self._lock:
            self._deciders.pop(name, None); shutil.rmtree(self.home / "deciders" / _check(name), ignore_errors=True)

    # ------------------------------------------------------------------ misc
    def overview(self) -> dict:
        brains = []
        for n in self.brain_names():
            try:
                s = self.brain(n, create=False).state()
                brains.append({k: s[k] for k in ("name", "status", "chunks", "neurons", "episodes")} | {"sources": len(s["sources"])})
            except Exception as e:      # a corrupt brain must not break the overview
                brains.append({"name": n, "error": str(e)})
        return {"home": str(self.home), "brains": brains, "deciders": [self.decider(n).info() for n in self.decider_names()]}

    def save_all(self):
        for b in self._brains.values():
            b.wait(); b.save()
        for d in self._deciders.values():
            d.save()
