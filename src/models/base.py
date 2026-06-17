from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Generation:
    text: str
    content: str | None = None
    thinking: str | None = None


class BaseModel:
    def __init__(self, name: str):
        self.name = name

    def generate(self, prompt: str, **kwargs) -> Generation:
        raise NotImplementedError
