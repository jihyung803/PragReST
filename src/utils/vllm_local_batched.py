from __future__ import annotations

import gc
import os
from types import SimpleNamespace
from typing import Any

from transformers import AutoTokenizer

from src.utils.chat_template_family import adapt_messages_for_chat_template
from src.utils.thinking_tags import contains_thinking_close, normalize_early_stopping_text
from src.eval.vllm_boxed_mcqa import is_gemma4_model_spec, is_ministral3_model_spec


def apply_chat_template_text(
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


def truncate_prompt_text_from_left(tokenizer, prompt_text: str, max_input_length: int) -> str:
    encoded = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]
    if len(encoded) <= max_input_length:
        return prompt_text
    trimmed = encoded[-max_input_length:]
    return tokenizer.decode(trimmed, skip_special_tokens=False)


def token_count(tokenizer, text: str) -> int:
    return len(
        tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
    )


class VLLMLocalBatchedGenerator:
    def __init__(
        self,
        model_name: str,
        *,
        tokenizer_name: str | None = None,
        dtype: str | None = None,
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 1,
        max_model_len: int | None = None,
        seed: int | None = None,
    ) -> None:
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        self._model_name = str(model_name)

        try:
            from vllm import LLM, SamplingParams
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("vllm not available; install dependencies to use VLLMLocalBatchedGenerator") from exc

        self._SamplingParams = SamplingParams
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name or model_name,
                trust_remote_code=True,
                fix_mistral_regex=True,
            )
        except TypeError:
            self._tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name or model_name,
                trust_remote_code=True,
            )

        llm_kwargs: dict[str, Any] = {
            "model": model_name,
            "trust_remote_code": True,
            "gpu_memory_utilization": float(gpu_memory_utilization),
            "tensor_parallel_size": int(tensor_parallel_size),
        }
        if is_ministral3_model_spec(model_name):
            llm_kwargs["tokenizer_mode"] = "mistral"
            llm_kwargs["config_format"] = "mistral"
            llm_kwargs["load_format"] = "mistral"
        if is_gemma4_model_spec(model_name):
            llm_kwargs["limit_mm_per_prompt"] = {"image": 0, "audio": 0}
        if dtype:
            llm_kwargs["dtype"] = str(dtype)
        if max_model_len is not None and int(max_model_len) > 0:
            llm_kwargs["max_model_len"] = int(max_model_len)
            self._max_model_len = int(max_model_len)
        else:
            self._max_model_len = None
        if seed is not None:
            llm_kwargs["seed"] = int(seed)

        self._llm = LLM(**llm_kwargs)

    def _build_prompt_texts(
        self,
        message_batches: list[list[dict[str, Any]]],
        *,
        enable_thinking: bool,
    ) -> list[str]:
        prompt_texts: list[str] = []
        for messages in message_batches:
            prompt_texts.append(
                apply_chat_template_text(
                    self._tokenizer,
                    self._model_name,
                    messages,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
            )
        return prompt_texts

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

    def generate_chat_batch(
        self,
        message_batches: list[list[dict[str, Any]]],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float | None,
        top_k: int | None,
        enable_thinking: bool = False,
        use_tqdm: bool = True,
    ) -> list[str]:
        prompt_texts = self._build_prompt_texts(message_batches, enable_thinking=enable_thinking)
        outputs = self._llm.generate(
            prompt_texts,
            self._sampling_params(
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            ),
            use_tqdm=use_tqdm,
        )
        return [str(output.outputs[0].text).strip() for output in outputs]

    def generate_with_thinking_budget_batch(
        self,
        message_batches: list[list[dict[str, Any]]],
        *,
        thinking_budget_tokens: int,
        answer_budget_tokens: int,
        early_stopping_text: str,
        temperature: float,
        top_p: float | None,
        top_k: int | None,
        enable_thinking: bool = True,
        use_tqdm: bool = True,
    ) -> list[str]:
        prompt_texts = self._build_prompt_texts(message_batches, enable_thinking=enable_thinking)
        first_outputs = self._llm.generate(
            prompt_texts,
            self._sampling_params(
                max_tokens=thinking_budget_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            ),
            use_tqdm=use_tqdm,
        )

        results = [""] * len(prompt_texts)
        needs_second: list[tuple[int, str, str, str]] = []
        stop_text = normalize_early_stopping_text(early_stopping_text, self._model_name)
        second_pass_limit = max(
            1,
            int((self._max_model_len or 0) - 32) if self._max_model_len is not None else 32768,
        )
        for idx, output in enumerate(first_outputs):
            first_text = str(output.outputs[0].text)
            finish_reason = str(output.outputs[0].finish_reason or "")
            if finish_reason != "length":
                results[idx] = first_text.strip()
                continue
            extra = "" if contains_thinking_close(first_text) else stop_text
            continued_prompt = prompt_texts[idx] + first_text + extra
            if self._max_model_len is not None and self._max_model_len > 0:
                continued_prompt = truncate_prompt_text_from_left(
                    self._tokenizer,
                    continued_prompt,
                    second_pass_limit,
                )
            needs_second.append((idx, continued_prompt, first_text, extra))

        if needs_second:
            lengths = [token_count(self._tokenizer, item[1]) for item in needs_second]
            print(
                f"[thinking-budget] second_pass count={len(needs_second)} "
                f"token_limit={second_pass_limit} max_tokens={max(lengths)} "
                f"min_tokens={min(lengths)}"
            )
            second_outputs = self._llm.generate(
                [item[1] for item in needs_second],
                self._sampling_params(
                    max_tokens=answer_budget_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                ),
                use_tqdm=use_tqdm,
            )
            for j, (idx, _, first_text, extra) in enumerate(needs_second):
                second_text = str(second_outputs[j].outputs[0].text)
                results[idx] = (first_text + extra + second_text).strip()

        return results

    def generate_batch(
        self,
        message_batches: list[list[dict[str, Any]]],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float | None,
        top_k: int | None,
        enable_thinking: bool | None = None,
        thinking_budget_tokens: int | None = None,
        answer_budget_tokens: int | None = None,
        early_stopping_text: str | None = None,
        use_tqdm: bool = True,
    ) -> list[str]:
        use_budget = bool(enable_thinking) and int(thinking_budget_tokens or 0) > 0
        if use_budget:
            answer_tokens = int(answer_budget_tokens or 0)
            if answer_tokens <= 0:
                answer_tokens = int(max_new_tokens)
            return self.generate_with_thinking_budget_batch(
                message_batches,
                thinking_budget_tokens=int(thinking_budget_tokens or 0),
                answer_budget_tokens=answer_tokens,
                early_stopping_text=str(early_stopping_text or ""),
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                enable_thinking=bool(enable_thinking),
                use_tqdm=use_tqdm,
            )
        return self.generate_chat_batch(
            message_batches,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            enable_thinking=bool(enable_thinking),
            use_tqdm=use_tqdm,
        )

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
    ):
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        text = self.generate_batch(
            [messages],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            enable_thinking=enable_thinking,
            thinking_budget_tokens=thinking_budget_tokens,
            answer_budget_tokens=answer_budget_tokens,
            early_stopping_text=early_stopping_text,
            use_tqdm=False,
        )[0]
        return SimpleNamespace(text=text, content=text, thinking=None)

    def close(self) -> None:
        try:
            del self._llm
        except Exception:
            pass
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
