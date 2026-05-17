"""
Configuration for the watermarking system, loaded from a JSON file.
Unspecified fields fall back to their defaults.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List


DEFAULT_ATTACKS: List[str] = [
    "original",
    "jpeg",
    "jpeg-q80",
    "jpeg-q70",
    "jpeg-q60",
    "jpeg-q50",
    "webp",
    "resize",
    "resize-down",
    "blur",
    "noise",
    "brightness-plus",
    "brightness-minus",
    "semantic-transformation-soft",
    "semantic-transformation-hard",
    "change",
    "social-bluesky",
]


@dataclass
class Settings:
    embed_method: str = "DWT"
    embed_og_hash: bool = True
    hash_size: int = 8
    hash_type: str = "whash"
    delta_dwt: float = 40.0
    key_size: int = 2048
    attacks: List[str] = field(default_factory=lambda: list(DEFAULT_ATTACKS))

    @classmethod
    def from_json(cls, path: str) -> "Settings":
        with open(path) as f:
            data = json.load(f)
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)
