from __future__ import annotations

from typing import Any

from .base import BaseModel, Generation
from src.utils.chat_template_family import adapt_messages_for_chat_template
from src.utils.thinking_tags import (
    ALL_CLOSE_TAGS,
    contains_thinking_close,
    normalize_early_stopping_text,
)


_SHARED_CACHE: dict[tuple, dict[str, Any]] = {}


class HFPolicyModel(BaseModel):
    def __init__(
        self,
        name: str,
        lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: list[str] | None = None,
        adapter_name: str | None = None,
        share_base: bool = False,
        device: str | None = None,
        dtype: str | None = None,
        device_map: str | dict | None = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
    ):
        super().__init__(name)
        try:
            import torch
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "transformers/torch not available; install dependencies to use HFPolicyModel"
            ) from exc

        self.torch = torch
        self.adapter_name = adapter_name
        self.lora = lora
        self.lora_target_modules = lora_target_modules
        self.share_base = share_base
        self.device_map = device_map
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = _resolve_dtype(dtype, torch)
        self._shared_key = None

        cache_key = (
            name,
            self.device_map if self.device_map is not None else self.device,
            str(self.dtype),
            load_in_4bit,
            load_in_8bit,
        )
        if share_base:
            self._shared_key = cache_key

        if share_base and cache_key in _SHARED_CACHE:
            cached = _SHARED_CACHE[cache_key]
            self.model = cached["model"]
            self.tokenizer = cached["tokenizer"]
            self.device = cached["device"]
        else:
            config = AutoConfig.from_pretrained(name, trust_remote_code=True)
            self.tokenizer = _load_tokenizer(name, AutoTokenizer, config=config)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            model_kwargs: dict[str, Any] = {"dtype": self.dtype}
            if device_map is not None:
                model_kwargs["device_map"] = device_map
            if load_in_4bit:
                model_kwargs["load_in_4bit"] = True
            if load_in_8bit:
                model_kwargs["load_in_8bit"] = True

            self.model = _load_model(name, AutoModelForCausalLM, config=config, **model_kwargs)
            if device_map is None:
                self.model.to(self.device)
            self.model.eval()
            if share_base:
                _SHARED_CACHE[cache_key] = {
                    "model": self.model,
                    "tokenizer": self.tokenizer,
                    "device": self.device,
                }

        if lora:
            self._ensure_lora_adapter(
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )

        self.optimizer = None
        self.training_enabled = False

    def enable_training(self, lr: float = 1e-4, weight_decay: float = 0.0) -> None:
        torch = self.torch
        params = self._select_trainable_params()
        if not params:
            raise RuntimeError("No trainable parameters found for HFPolicyModel.")
        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        self.training_enabled = True
        self.model.train()

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
        torch = self.torch
        was_training = self.training_enabled
        if was_training:
            self.model.eval()

        self._set_active_adapter()
        inputs = self._build_prompt_inputs(
            system_prompt,
            prompt,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        ).to(self.device)
        prompt_len = inputs["input_ids"].shape[1]

        do_sample = temperature > 0.0
        gen_kwargs: dict[str, Any] = {
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            if top_p is not None:
                gen_kwargs["top_p"] = top_p
            if top_k is not None and int(top_k) > 0:
                gen_kwargs["top_k"] = int(top_k)

        use_budget = _is_positive_int(thinking_budget_tokens) and (enable_thinking is None or bool(enable_thinking))
        think_budget = int(thinking_budget_tokens or 0)
        answer_budget = int(answer_budget_tokens) if _is_positive_int(answer_budget_tokens) else int(max_new_tokens)
        stop_text = normalize_early_stopping_text(early_stopping_text, self.name)

        with torch.no_grad():
            if use_budget:
                first_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max(1, think_budget),
                    **gen_kwargs,
                )
                completion_ids = first_ids[0][prompt_len:].tolist()

                eos_id = self.tokenizer.eos_token_id
                im_end_id = _maybe_special_id(self.tokenizer, "<|im_end|>")
                finished = False
                if eos_id is not None and eos_id in completion_ids:
                    finished = True
                if im_end_id is not None and im_end_id in completion_ids:
                    finished = True

                output_ids = first_ids
                if not finished:
                    continued_ids = first_ids
                    think_close_ids_list = []
                    for close_tag in ALL_CLOSE_TAGS:
                        try:
                            think_close_ids = self.tokenizer.encode(close_tag, add_special_tokens=False)
                        except Exception:
                            think_close_ids = []
                        if think_close_ids:
                            think_close_ids_list.append(think_close_ids)
                    has_think_close = any(
                        _contains_subsequence(completion_ids, think_ids)
                        for think_ids in think_close_ids_list
                    )
                    if not has_think_close:
                        extra_ids = self.tokenizer(
                            [stop_text],
                            return_tensors="pt",
                            add_special_tokens=False,
                        )["input_ids"].to(self.device)
                        continued_ids = torch.cat([continued_ids, extra_ids], dim=-1)

                    attention_mask = torch.ones_like(continued_ids, dtype=torch.long, device=self.device)
                    output_ids = self.model.generate(
                        input_ids=continued_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max(1, answer_budget),
                        **gen_kwargs,
                    )
            else:
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max(1, int(max_new_tokens)),
                    **gen_kwargs,
                )

        generated = output_ids[0][prompt_len:].tolist()
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        thinking, content = _split_thinking(self.tokenizer, generated, text)

        if was_training:
            self.model.train()
        return Generation(text=text, content=content, thinking=thinking)

    def response_logprob(
        self,
        prompt: str,
        response: str,
        system_prompt: str | list[dict[str, Any]] | None = None,
        enable_thinking: bool | None = None,
    ) -> "torch.Tensor":
        torch = self.torch
        self._set_active_adapter()
        prompt_ids = self._build_prompt_inputs(
            system_prompt,
            prompt,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        ).to(self.device)
        response_ids = self.tokenizer(
            response,
            return_tensors="pt",
            truncation=True,
            add_special_tokens=False,
        ).to(self.device)

        input_ids = torch.cat([prompt_ids["input_ids"], response_ids["input_ids"]], dim=-1)
        full_attention_mask = None
        if "attention_mask" in prompt_ids and "attention_mask" in response_ids:
            full_attention_mask = torch.cat(
                [prompt_ids["attention_mask"], response_ids["attention_mask"]], dim=-1
            )

        model_kwargs: dict[str, Any] = {"input_ids": input_ids}
        if full_attention_mask is not None:
            model_kwargs["attention_mask"] = full_attention_mask
        with torch.inference_mode():
            outputs = self.model(**model_kwargs)
            logits = outputs.logits  # [1, seq, vocab]
            log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
            target_ids = input_ids[:, 1:]

            prompt_len = prompt_ids["input_ids"].shape[1]
            if prompt_len <= 0:
                prompt_len = 1

            resp_log_probs = log_probs[:, prompt_len - 1 :, :].gather(
                -1, target_ids[:, prompt_len - 1 :].unsqueeze(-1)
            )
            score = resp_log_probs.squeeze(-1).sum()
        return score

    def step(self, loss: "torch.Tensor", max_grad_norm: float | None = None) -> None:
        if not self.optimizer:
            raise RuntimeError("Optimizer not initialized; call enable_training() first.")
        loss.backward()
        if max_grad_norm is not None:
            self.torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def save_adapter(self, path: str) -> None:
        from pathlib import Path

        self._set_active_adapter()
        if not hasattr(self.model, "save_pretrained"):
            raise RuntimeError("Model does not support save_pretrained.")
        out_path = Path(path)
        out_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(out_path))

    def _build_prompt_inputs(
        self,
        system_prompt: str | list[dict[str, Any]] | None,
        user_prompt: str,
        add_generation_prompt: bool,
        enable_thinking: bool | None = None,
    ):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        messages, template_kwargs = adapt_messages_for_chat_template(self.tokenizer, self.name, messages)

        if _is_mistral_common_backend_tokenizer(self.tokenizer):
            if enable_thinking is not None:
                try:
                    return self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=add_generation_prompt,
                        return_tensors="pt",
                        enable_thinking=enable_thinking,
                        **template_kwargs,
                    )
                except (TypeError, ValueError):
                    pass
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                return_tensors="pt",
                **template_kwargs,
            )

        full_prompt = self._build_prompt(
            system_prompt, user_prompt, add_generation_prompt=add_generation_prompt, enable_thinking=enable_thinking
        )
        return self.tokenizer(full_prompt, return_tensors="pt", truncation=True)

    def _build_prompt(
        self,
        system_prompt: str | list[dict[str, Any]] | None,
        user_prompt: str,
        add_generation_prompt: bool,
        enable_thinking: bool | None = None,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        messages, template_kwargs = adapt_messages_for_chat_template(self.tokenizer, self.name, messages)

        if hasattr(self.tokenizer, "apply_chat_template"):
            if enable_thinking is not None:
                try:
                    return self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=add_generation_prompt,
                        enable_thinking=enable_thinking,
                        **template_kwargs,
                    )
                except (TypeError, ValueError):
                    pass
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                **template_kwargs,
            )

        if system_prompt:
            return system_prompt + "\n\n" + user_prompt
        return user_prompt

    def _ensure_lora_adapter(self, lora_r: int, lora_alpha: int, lora_dropout: float) -> None:
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("peft not available; install to enable LoRA.") from exc

        if getattr(self.model, "is_loaded_in_4bit", False) or getattr(
            self.model, "is_loaded_in_8bit", False
        ):
            self.model = prepare_model_for_kbit_training(self.model)

        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=self.lora_target_modules,
        )

        if not hasattr(self.model, "peft_config"):
            self.model = get_peft_model(self.model, config)
        if self.share_base and self._shared_key:
            _SHARED_CACHE[self._shared_key]["model"] = self.model

        if self.adapter_name and hasattr(self.model, "add_adapter"):
            if self.adapter_name not in getattr(self.model, "peft_config", {}):
                self.model.add_adapter(self.adapter_name, config)
            self.model.set_adapter(self.adapter_name)

    def _set_active_adapter(self) -> None:
        if not hasattr(self.model, "set_adapter"):
            return
        if self.adapter_name:
            peft_config = getattr(self.model, "peft_config", None)
            if peft_config and self.adapter_name in peft_config:
                self.model.set_adapter(self.adapter_name)
                return
            if hasattr(self.model, "disable_adapter"):
                self.model.disable_adapter()
            return
        if hasattr(self.model, "disable_adapter"):
            self.model.disable_adapter()

    def _select_trainable_params(self):
        if self.lora and self.adapter_name and hasattr(self.model, "named_parameters"):
            params = []
            token = f".{self.adapter_name}."
            for name, param in self.model.named_parameters():
                if token in name or name.endswith(f".{self.adapter_name}"):
                    if param.requires_grad:
                        params.append(param)
            if params:
                return params
        return [p for p in self.model.parameters() if p.requires_grad]


def _resolve_dtype(value: str | None, torch_module):
    if value is None:
        return torch_module.bfloat16 if torch_module.cuda.is_available() else torch_module.float32
    value = value.lower()
    if value in {"bf16", "bfloat16"}:
        return torch_module.bfloat16
    if value in {"fp16", "float16"}:
        return torch_module.float16
    if value in {"fp32", "float32"}:
        return torch_module.float32
    return torch_module.float32


def _is_mistral3_family_config(config: Any) -> bool:
    model_type = str(getattr(config, "model_type", "") or "").strip().lower()
    if model_type == "mistral3":
        return True
    text_config = getattr(config, "text_config", None)
    text_model_type = str(getattr(text_config, "model_type", "") or "").strip().lower()
    return text_model_type in {"ministral3", "mistral3"}


def _load_model(name: str, auto_model_cls, *, config: Any, **model_kwargs):
    if _is_mistral3_family_config(config):
        from transformers import Mistral3ForConditionalGeneration

        return Mistral3ForConditionalGeneration.from_pretrained(name, **model_kwargs)
    return auto_model_cls.from_pretrained(name, **model_kwargs)


def _load_tokenizer(name: str, auto_tokenizer_cls, *, config: Any | None = None):
    if config is not None and _is_mistral3_family_config(config):
        from transformers import MistralCommonBackend

        return MistralCommonBackend.from_pretrained(name, trust_remote_code=True)
    common_kwargs = {"trust_remote_code": True}
    try:
        return auto_tokenizer_cls.from_pretrained(name, fix_mistral_regex=True, **common_kwargs)
    except TypeError:
        return auto_tokenizer_cls.from_pretrained(name, **common_kwargs)


def _split_thinking(tokenizer, output_ids: list[int], text: str) -> tuple[str | None, str | None]:
    if not output_ids:
        return None, None
    for close_tag in ALL_CLOSE_TAGS:
        try:
            think_ids = tokenizer.encode(close_tag, add_special_tokens=False)
        except Exception:
            think_ids = []
        if len(think_ids) == 1:
            think_id = think_ids[0]
            try:
                index = len(output_ids) - list(reversed(output_ids)).index(think_id)
            except ValueError:
                index = 0
            if index > 0:
                thinking = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip()
                content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()
                return thinking or None, content or None
    lowered = text.lower()
    for close_tag in ALL_CLOSE_TAGS:
        if close_tag.lower() in lowered:
            split_idx = lowered.find(close_tag.lower())
            tail = text[split_idx + len(close_tag) :]
            return None, tail.strip() or None
    return None, None


def _is_positive_int(value: Any) -> bool:
    if value is None:
        return False
    try:
        return int(value) > 0
    except Exception:
        return False


def _contains_subsequence(sequence: list[int], pattern: list[int]) -> bool:
    if not pattern or len(pattern) > len(sequence):
        return False
    limit = len(sequence) - len(pattern) + 1
    for i in range(limit):
        if sequence[i : i + len(pattern)] == pattern:
            return True
    return False


def _maybe_special_id(tokenizer, token: str) -> int | None:
    try:
        token_id = tokenizer.convert_tokens_to_ids(token)
    except Exception:
        return None
    if isinstance(token_id, int) and token_id >= 0:
        return token_id
    return None


def _is_mistral_common_backend_tokenizer(tokenizer: Any) -> bool:
    return tokenizer.__class__.__name__ == "MistralCommonBackend"
