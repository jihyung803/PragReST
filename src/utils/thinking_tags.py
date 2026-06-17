from __future__ import annotations

from typing import Iterable

ANGLE_OPEN = "<think>"
ANGLE_CLOSE = "</think>"
BRACKET_OPEN = "[THINK]"
BRACKET_OPEN_LOWER = "[think]"
BRACKET_CLOSE = "[/THINK]"
BRACKET_CLOSE_LOWER = "[/think]"
GEMMA4_OPEN = "<|channel>thought"
GEMMA4_CLOSE = "<channel|>"

ALL_OPEN_TAGS = (ANGLE_OPEN, BRACKET_OPEN, BRACKET_OPEN_LOWER, GEMMA4_OPEN)
ALL_CLOSE_TAGS = (ANGLE_CLOSE, BRACKET_CLOSE, BRACKET_CLOSE_LOWER, GEMMA4_CLOSE)


def is_ministral_family(model_name_or_path: str | None) -> bool:
    name = str(model_name_or_path or "").strip().lower()
    return any(token in name for token in ("ministral", "mistral-3", "mistral3"))


def is_gemma4_family(model_name_or_path: str | None) -> bool:
    name = str(model_name_or_path or "").strip().lower()
    return "gemma-4" in name or "gemma4" in name


def preferred_thinking_close_tag(model_name_or_path: str | None) -> str:
    if is_ministral_family(model_name_or_path):
        return BRACKET_CLOSE_LOWER
    if is_gemma4_family(model_name_or_path):
        return GEMMA4_CLOSE
    return ANGLE_CLOSE


def normalize_early_stopping_text(text: str | None, model_name_or_path: str | None) -> str:
    raw = str(text or "")
    close_tag = preferred_thinking_close_tag(model_name_or_path)
    if not raw.strip():
        return (
            "\n\nConsidering the limited time by the user, I have to give the solution "
            f"based on the thinking directly now.\n{close_tag}\n\n"
        )
    normalized = raw
    for tag in ALL_CLOSE_TAGS:
        normalized = normalized.replace(tag, close_tag)
    return normalized


def contains_thinking_close(text: str | None, *, close_tags: Iterable[str] | None = None) -> bool:
    raw = str(text or "")
    tags = tuple(close_tags or ALL_CLOSE_TAGS)
    lowered = raw.lower()
    return any(tag.lower() in lowered for tag in tags)


def strip_prefixed_thinking(text: str | None) -> str:
    raw = str(text or "")
    stripped = raw.lstrip()
    lowered = stripped.lower()
    for open_tag in ALL_OPEN_TAGS:
        if not lowered.startswith(open_tag.lower()):
            continue
        for close_tag in ALL_CLOSE_TAGS:
            idx = lowered.find(close_tag.lower())
            if idx != -1:
                return stripped[idx + len(close_tag) :].strip()
    return raw.strip()


def split_after_last_thinking_close(text: str | None) -> str:
    raw = str(text or "")
    lowered = raw.lower()
    best = -1
    best_tag = ""
    for tag in ALL_CLOSE_TAGS:
        idx = lowered.rfind(tag.lower())
        if idx > best:
            best = idx
            best_tag = tag
    if best != -1:
        return raw[best + len(best_tag) :].strip()
    return raw.strip()
