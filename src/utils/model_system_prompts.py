from __future__ import annotations

from functools import lru_cache
from typing import Any

from huggingface_hub import hf_hub_download


@lru_cache(maxsize=16)
def _read_hf_text(repo_id: str, filename: str) -> str:
    file_path = hf_hub_download(repo_id=repo_id, filename=filename)
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def build_ministral_reasoning_system_prompt(
    custom_system_prompt: str | None,
    *,
    repo_id: str = "mistralai/Ministral-3-14B-Reasoning-2512",
    filename: str = "SYSTEM_PROMPT.txt",
) -> list[dict[str, Any]]:
    raw = _read_hf_text(repo_id, filename)

    begin_tag = "[THINK]"
    end_tag = "[/THINK]"
    begin_idx = raw.find(begin_tag)
    end_idx = raw.find(end_tag)
    if begin_idx < 0 or end_idx < 0 or end_idx < begin_idx:
        raise RuntimeError(
            f"Could not parse Ministral system prompt thinking tags from {repo_id}/{filename}"
        )

    before = raw[:begin_idx]
    thinking = raw[begin_idx + len(begin_tag) : end_idx]
    after = raw[end_idx + len(end_tag) :]

    custom = str(custom_system_prompt or "").strip()
    if custom:
        suffix = after.rstrip()
        if suffix:
            suffix = f"{suffix}\n\n{custom}"
        else:
            suffix = custom
    else:
        suffix = after

    return [
        {"type": "text", "text": before},
        {"type": "thinking", "thinking": thinking, "closed": True},
        {"type": "text", "text": suffix},
    ]


def adapt_system_prompt(
    system_prompt: str | None,
    *,
    style: str = "plain",
    repo_id: str = "mistralai/Ministral-3-14B-Reasoning-2512",
    filename: str = "SYSTEM_PROMPT.txt",
) -> str | list[dict[str, Any]] | None:
    if not system_prompt:
        return system_prompt
    normalized = str(style or "plain").strip().lower()
    if normalized == "plain":
        return system_prompt
    if normalized == "ministral_reasoning":
        return build_ministral_reasoning_system_prompt(
            system_prompt,
            repo_id=repo_id,
            filename=filename,
        )
    raise ValueError(f"Unsupported system prompt style: {style}")
