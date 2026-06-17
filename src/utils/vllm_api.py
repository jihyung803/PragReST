from __future__ import annotations

import time
from typing import Any

import requests

try:
    from src.models.base import BaseModel, Generation
except ModuleNotFoundError:  # pragma: no cover - supports slim runtime checkouts
    from dataclasses import dataclass

    @dataclass
    class Generation:
        text: str
        content: str | None = None
        thinking: str | None = None

    class BaseModel:
        def __init__(self, name: str):
            self.name = name


def _build_chat_completions_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        return normalized + "/chat/completions"
    return normalized + "/v1/chat/completions"


def _coerce_content(message_content: Any) -> str:
    if isinstance(message_content, str):
        return message_content
    if isinstance(message_content, list):
        parts: list[str] = []
        for item in message_content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(message_content or "")


def _is_positive_int(value: Any) -> bool:
    if value is None:
        return False
    try:
        return int(value) > 0
    except Exception:
        return False


class VLLMApiModel(BaseModel):
    def __init__(
        self,
        name: str,
        *,
        base_url: str = "http://localhost:8000",
        api_key: str = "EMPTY",
        timeout: float = 180.0,
        max_retries: int = 3,
    ):
        super().__init__(name)
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.url = _build_chat_completions_url(base_url)
        self.session = requests.Session()

    def _call_chat(self, payload: dict[str, Any]) -> tuple[str, str | None]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.post(
                    self.url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                choice = data["choices"][0]
                msg = choice["message"]
                text = _coerce_content(msg.get("content"))
                finish_reason = choice.get("finish_reason")
                return text, str(finish_reason) if finish_reason is not None else None
            except Exception as exc:  # pragma: no cover
                last_err = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(min(2**attempt, 4))

        raise RuntimeError(f"vLLM API generation failed: {last_err}") from last_err

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float | None = None,
        top_k: int | None = None,
        system_prompt: str | list[dict[str, Any]] | None = None,
        enable_thinking: bool | None = None,
        thinking_budget_tokens: int | None = None,
        answer_budget_tokens: int | None = None,
        early_stopping_text: str | None = None,
        **_,
    ) -> Generation:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        do_sample = float(temperature) > 0.0

        def _build_payload(local_messages: list[dict[str, Any]], max_tokens: int) -> dict[str, Any]:
            payload: dict[str, Any] = {
                "model": self.name,
                "messages": local_messages,
                "max_tokens": max(1, int(max_tokens)),
                "temperature": float(temperature) if do_sample else 0.0,
                "top_p": float(top_p) if (do_sample and top_p is not None) else 1.0,
                "stream": False,
            }
            if do_sample and top_k is not None and int(top_k) > 0:
                payload["top_k"] = int(top_k)
            if enable_thinking is not None:
                payload["extra_body"] = {
                    "chat_template_kwargs": {
                        "enable_thinking": bool(enable_thinking),
                    }
                }
            return payload

        use_budget = _is_positive_int(thinking_budget_tokens) and (enable_thinking is None or bool(enable_thinking))
        if not use_budget:
            text, _ = self._call_chat(_build_payload(messages, int(max_new_tokens)))
            return Generation(text=text, content=text, thinking=None)

        think_budget = int(thinking_budget_tokens or 0)
        answer_budget = int(answer_budget_tokens) if _is_positive_int(answer_budget_tokens) else int(max_new_tokens)
        stop_text = early_stopping_text or (
            "\n\nConsidering the limited time by the user, I have to give the solution "
            "based on the thinking directly now.\n</think>\n\n"
        )

        first_text, finish_reason = self._call_chat(_build_payload(messages, think_budget))
        if finish_reason != "length":
            return Generation(text=first_text, content=first_text, thinking=None)

        extra = "" if "</think>" in first_text else stop_text
        continued_messages = list(messages)
        continued_messages.append({"role": "assistant", "content": first_text + extra})

        second_text, _ = self._call_chat(_build_payload(continued_messages, answer_budget))
        final_text = (first_text + extra + second_text).strip()
        return Generation(text=final_text, content=final_text, thinking=None)
