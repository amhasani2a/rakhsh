"""Universal ingestion with a pluggable loader registry.

>>> from rakhsh.ingest import ingest, register_loader
>>> register_loader(["log"], lambda data, name, max_words: my_parser(data, name))
"""
from __future__ import annotations

from typing import Callable

from .chunk import Chunk
from . import loaders as _l

Loader = Callable[[bytes, str, int], "list[Chunk]"]
_REGISTRY: dict[str, Loader] = {}


def register_loader(extensions: list[str], fn: Loader) -> None:
    """Register (or override) a loader for file extensions (without the dot)."""
    for e in extensions:
        _REGISTRY[e.lower().lstrip(".")] = fn


def supported_extensions() -> list[str]:
    return sorted(_REGISTRY)


register_loader(["csv", "tsv"], lambda d, n, w: _l.load_csv(d, n))
register_loader(["json", "jsonl", "ndjson"], lambda d, n, w: _l.load_json(d, n))
register_loader(["sqlite", "sqlite3", "db"], lambda d, n, w: _l.load_sqlite(d, n))
register_loader(["xlsx", "xlsm"], lambda d, n, w: _l.load_xlsx(d, n))
register_loader(["pdf"], lambda d, n, w: _l.load_pdf(d, n, w))
register_loader(["docx"], lambda d, n, w: _l.load_docx(d, n, w))
register_loader(["html", "htm"], lambda d, n, w: _l.load_html(d, n, w))
register_loader(["txt", "md", "markdown", "text", "rst", "log"], lambda d, n, w: _l.chunk_text(_l.decode(d), n, w))


def ingest(name: str, data: bytes, max_words: int = 90) -> list[Chunk]:
    """Convert any supported file (by extension / magic bytes) into chunks. Unknown -> plain text."""
    ext = name.lower().rsplit(".", 1)[-1] if "." in name else "txt"
    if data[:16].startswith(b"SQLite format 3"):
        ext = "sqlite"
    elif data[:5] == b"%PDF-":
        ext = "pdf"
    fn = _REGISTRY.get(ext, _REGISTRY["txt"])
    return fn(data, name, max_words)


__all__ = ["Chunk", "ingest", "register_loader", "supported_extensions"]
