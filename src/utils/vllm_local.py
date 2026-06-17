from __future__ import annotations

import math
import os
from typing import Any

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

from src.utils.chat_template_family import adapt_messages_for_chat_template
from src.utils.thinking_tags import contains_thinking_close, normalize_early_stopping_text
from src.eval.vllm_boxed_mcqa import is_gemma4_model_spec, is_ministral3_model_spec
from transformers import AutoTokenizer


def _load_tokenizer(name: str):
    try:
        return AutoTokenizer.from_pretrained(
            name,
            trust_remote_code=True,
            fix_mistral_regex=True,
        )
    except TypeError:
        return AutoTokenizer.from_pretrained(
            name,
            trust_remote_code=True,
        )


def _apply_chat_template_text(
    tokenizer,
    model_name: str,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
    enable_thinking: bool,
) -> str:
    messages, template_kwargs = adapt_messages_for_chat_template(tokenizer, model_name, messages)
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
        **template_kwargs,
    }
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def _is_positive_int(value: Any) -> bool:
    if value is None:
        return False
    try:
        return int(value) > 0
    except Exception:
        return False


class VLLMLocalModel(BaseModel):
    def __init__(
        self,
        name: str,
        *,
        dtype: str | None = None,
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 1,
        max_model_len: int | None = None,
        seed: int | None = None,
    ):
        super().__init__(name)
        # vLLM engine workers must use spawn if the parent process has imported
        # torch and may have touched CUDA-related paths.
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        try:
            from vllm import LLM, SamplingParams
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("vllm not available; install dependencies to use VLLMLocalModel") from exc

        llm_kwargs: dict[str, Any] = {
            "model": name,
            "trust_remote_code": True,
            "gpu_memory_utilization": float(gpu_memory_utilization),
            "tensor_parallel_size": int(tensor_parallel_size),
        }
        if is_ministral3_model_spec(name):
            llm_kwargs["tokenizer_mode"] = "mistral"
            llm_kwargs["config_format"] = "mistral"
            llm_kwargs["load_format"] = "mistral"
        if is_gemma4_model_spec(name):
            llm_kwargs["limit_mm_per_prompt"] = {"image": 0, "audio": 0}
        if dtype:
            llm_kwargs["dtype"] = str(dtype)
        if _is_positive_int(max_model_len):
            llm_kwargs["max_model_len"] = int(max_model_len)
        if seed is not None:
            llm_kwargs["seed"] = int(seed)

        self._SamplingParams = SamplingParams
        self._tokenizer = _load_tokenizer(name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._llm = LLM(**llm_kwargs)

    def _sampling_params(
        self,
        *,
        max_tokens: int,
        temperature: float,
        top_p: float | None,
        top_k: int | None,
    ):
        do_sample = float(temperature) > 0.0
        return self._SamplingParams(
            max_tokens=max(1, int(max_tokens)),
            temperature=float(temperature) if do_sample else 0.0,
            top_p=float(top_p) if (do_sample and top_p is not None) else 1.0,
            top_k=int(top_k) if (do_sample and top_k is not None and int(top_k) > 0) else -1,
        )

    def _chat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float | None,
        top_k: int | None,
        enable_thinking: bool | None,
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
    ) -> tuple[str, str | None]:
        messages, template_kwargs = adapt_messages_for_chat_template(self._tokenizer, self.name, messages)
        kwargs: dict[str, Any] = {
            "use_tqdm": False,
            "add_generation_prompt": bool(add_generation_prompt),
            "continue_final_message": bool(continue_final_message),
        }
        chat_template_kwargs: dict[str, Any] = dict(template_kwargs)
        if enable_thinking is not None:
            chat_template_kwargs["enable_thinking"] = bool(enable_thinking)
        if chat_template_kwargs:
            kwargs["chat_template_kwargs"] = chat_template_kwargs

        outputs = self._llm.chat(
            messages,
            self._sampling_params(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            ),
            **kwargs,
        )
        output = outputs[0].outputs[0]
        text = str(output.text)
        finish_reason = getattr(output, "finish_reason", None)
        return text, str(finish_reason) if finish_reason is not None else None

    def _messages(
        self,
        prompt: str,
        *,
        system_prompt: str | list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _build_prompt_text(
        self,
        prompt: str,
        *,
        system_prompt: str | list[dict[str, Any]] | None = None,
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
    ) -> str:
        return _apply_chat_template_text(
            self._tokenizer,
            self.name,
            self._messages(prompt, system_prompt=system_prompt),
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )

    @staticmethod
    def _position_logprob(
        prompt_logprobs: Any,
        position: int,
        token_id: int,
    ) -> float:
        if prompt_logprobs is None:
            return float("-inf")
        try:
            entry = prompt_logprobs[position]
        except Exception:
            return float("-inf")
        if entry is None:
            return float("-inf")
        if isinstance(entry, dict):
            candidate = entry.get(int(token_id))
            if candidate is None:
                return float("-inf")
            value = getattr(candidate, "logprob", candidate)
            try:
                return float(value)
            except Exception:
                return float("-inf")
        return float("-inf")

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
        messages = self._messages(prompt, system_prompt=system_prompt)

        use_budget = _is_positive_int(thinking_budget_tokens) and (enable_thinking is None or bool(enable_thinking))
        if not use_budget:
            text, _ = self._chat(
                messages,
                max_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_p=top_p,
                top_k=top_k,
                enable_thinking=enable_thinking,
            )
            text = text.strip()
            return Generation(text=text, content=text, thinking=None)

        think_budget = int(thinking_budget_tokens or 0)
        answer_budget = int(answer_budget_tokens) if _is_positive_int(answer_budget_tokens) else int(max_new_tokens)
        stop_text = normalize_early_stopping_text(early_stopping_text, self.name)

        first_text, finish_reason = self._chat(
            messages,
            max_tokens=think_budget,
            temperature=float(temperature),
            top_p=top_p,
            top_k=top_k,
            enable_thinking=enable_thinking,
        )
        if finish_reason != "length":
            text = first_text.strip()
            return Generation(text=text, content=text, thinking=None)

        extra = "" if contains_thinking_close(first_text) else stop_text
        continued_messages = list(messages)
        continued_messages.append({"role": "assistant", "content": first_text + extra})

        second_text, _ = self._chat(
            continued_messages,
            max_tokens=answer_budget,
            temperature=float(temperature),
            top_p=top_p,
            top_k=top_k,
            enable_thinking=enable_thinking,
            add_generation_prompt=False,
            continue_final_message=True,
        )
        final_text = (first_text + extra + second_text).strip()
        return Generation(text=final_text, content=final_text, thinking=None)

    def response_logprob(
        self,
        prompt: str,
        response: str,
        system_prompt: str | list[dict[str, Any]] | None = None,
        enable_thinking: bool | None = None,
    ):
        import torch

        prompt_text = self._build_prompt_text(
            prompt,
            system_prompt=system_prompt,
            add_generation_prompt=True,
            enable_thinking=bool(enable_thinking),
        )
        full_text = prompt_text + str(response)

        prompt_token_ids = self._tokenizer(
            prompt_text,
            add_special_tokens=False,
        )["input_ids"]
        full_token_ids = self._tokenizer(
            full_text,
            add_special_tokens=False,
        )["input_ids"]

        prompt_len = int(len(prompt_token_ids))
        full_len = int(len(full_token_ids))
        if prompt_len <= 0 or prompt_len >= full_len:
            return torch.tensor(float("-inf"))

        outputs = self._llm.generate(
            [full_text],
            self._SamplingParams(
                max_tokens=1,
                temperature=0.0,
                top_p=1.0,
                prompt_logprobs=1,
            ),
            use_tqdm=False,
        )
        request_output = outputs[0]
        output_prompt_token_ids = list(getattr(request_output, "prompt_token_ids", []) or [])
        prompt_logprobs = getattr(request_output, "prompt_logprobs", None)
        if not output_prompt_token_ids or prompt_logprobs is None:
            return torch.tensor(float("-inf"))

        # Use the vLLM-normalized prompt token ids when available; they should
        # align with prompt_logprobs more reliably than a separate tokenizer pass.
        token_ids = output_prompt_token_ids
        if prompt_len >= len(token_ids):
            return torch.tensor(float("-inf"))

        total = 0.0
        for pos in range(prompt_len, len(token_ids)):
            lp = self._position_logprob(prompt_logprobs, pos, int(token_ids[pos]))
            if not math.isfinite(lp):
                return torch.tensor(float("-inf"))
            total += lp
        return torch.tensor(total)

    def close(self) -> None:
        try:
            del self._llm
        except Exception:
            pass
