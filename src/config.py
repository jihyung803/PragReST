from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import yaml


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class RunConfig:
    raw: dict[str, Any]

    @property
    def seed(self) -> int:
        return int(self.raw.get("seed", 42))

    def get(self, *keys, default=None):
        node: Any = self.raw
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node
