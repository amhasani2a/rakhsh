"""The unit of knowledge every source is converted into."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Chunk:
    source: str                  # file / source name
    locator: str                 # human-readable position: "فصل ۳ · بند ۱۲", "table users · row 41"
    title: str
    text: str
    kind: str = "text"           # "text" | "row"
    fields: dict | None = None   # column -> value for rows
    sentences: list[str] = field(default_factory=list)

    @property
    def model_text(self) -> str:
        return f"{self.title} {self.text}" if self.title else self.text

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "Chunk":
        return cls(**d)
