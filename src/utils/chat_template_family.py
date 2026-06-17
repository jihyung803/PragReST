from __future__ import annotations

from typing import Any


def is_phi_reasoning_family(model_name_or_path: str | None) -> bool:
    name = str(model_name_or_path or "").strip().lower()
    return "phi" in name and "reasoning" in name


def is_gemma4_family(model_name_or_path: str | None) -> bool:
    name = str(model_name_or_path or "").strip().lower()
    return "gemma-4" in name or "gemma4" in name


def _stringify_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                item_type = str(item.get("type", "")).strip().lower()
                if item_type == "text":
                    text = str(item.get("text", "")).strip()
                    if text:
                        parts.append(text)
                elif item_type == "thinking":
                    thinking = str(item.get("thinking", "")).strip()
                    if thinking:
                        parts.append(f"<think>{thinking}</think>")
                else:
                    text = str(item.get("text", item)).strip()
                    if text:
                        parts.append(text)
            else:
                text = str(item).strip()
                if text:
                    parts.append(text)
        return "\n\n".join(parts).strip()
    return str(content).strip()


def _gemma4_content(content: Any) -> list[dict[str, str]]:
    if isinstance(content, list):
        items: list[dict[str, str]] = []
        for item in content:
            if isinstance(item, dict):
                item_type = str(item.get("type", "text") or "text").strip().lower()
                if item_type == "text":
                    text = str(item.get("text", "") or "")
                    items.append({"type": "text", "text": text})
                else:
                    copied = {str(k): str(v) for k, v in item.items() if v is not None}
                    if copied:
                        items.append(copied)
            else:
                items.append({"type": "text", "text": str(item)})
        return items
    return [{"type": "text", "text": str(content or "")}]


def _adapt_gemma4_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **message,
            "content": _gemma4_content(message.get("content")),
        }
        for message in messages
    ]


def adapt_messages_for_chat_template(
    tokenizer,
    model_name_or_path: str | None,
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if is_gemma4_family(model_name_or_path):
        return _adapt_gemma4_messages(messages), {}

    if not is_phi_reasoning_family(model_name_or_path):
        return messages, {}

    custom_system_parts: list[str] = []
    non_system_messages: list[dict[str, Any]] = []
    for message in messages:
        if str(message.get("role", "")).strip().lower() == "system":
            text = _stringify_content(message.get("content"))
            if text:
                custom_system_parts.append(text)
            continue
        non_system_messages.append(message)

    custom_system = "\n\n".join(part for part in custom_system_parts if part).strip()
    if not custom_system:
        return messages, {}

    adapted_messages = list(non_system_messages)
    for idx, message in enumerate(adapted_messages):
        if str(message.get("role", "")).strip().lower() != "user":
            continue
        original = _stringify_content(message.get("content"))
        merged = f"{custom_system}\n\n{original}".strip() if original else custom_system
        adapted_messages[idx] = {**message, "content": merged}
        return adapted_messages, {}

    adapted_messages.insert(0, {"role": "user", "content": custom_system})
    return adapted_messages, {}
