#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
ROOT_PATH = Path(ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LoRA SFT model on pragmatic thinking dataset.")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument(
        "--train_data_path",
        type=Path,
        default=ROOT_PATH / "data/sft/pragmatic_sft_thinking_train.jsonl",
    )
    parser.add_argument(
        "--val_data_path",
        type=Path,
        default=ROOT_PATH / "data/sft/pragmatic_sft_thinking_val.jsonl",
    )
    parser.add_argument("--output_dir", type=Path, default=ROOT_PATH / "results/sft_pragmatic_lora")
    parser.add_argument("--max_length", type=int, default=16384)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,v_proj,o_proj",
        help="comma-separated target modules",
    )

    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument(
        "--device_map",
        type=str,
        default="",
        help="Hugging Face device_map. Empty string disables auto CPU offload.",
    )
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")

    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--run_name", type=str, default="pragmatic-sft-lora")
    parser.add_argument("--resume_from_checkpoint", type=str, default="")


    return parser.parse_args()


def resolve_model_spec(model_spec: str) -> dict[str, str | None]:
    raw = str(model_spec or "").strip()
    if "::" in raw:
        base_model, lora_adapter = [part.strip() for part in raw.split("::", 1)]
    else:
        base_model = raw
        lora_adapter = ""
    if not base_model:
        raise ValueError(f"Invalid model spec without base model: {model_spec!r}")
    return {
        "base_model": base_model,
        "lora_adapter": lora_adapter or None,
    }


def _is_gemma4_model(model_name_or_path: str) -> bool:
    raw = str(model_name_or_path or "").strip().lower()
    return "gemma-4" in raw or "gemma4" in raw


def _load_causal_or_conditional_model(model_name_or_path: str, model_kwargs: dict[str, Any]):
    if _is_gemma4_model(model_name_or_path):
        from transformers import Gemma4ForConditionalGeneration

        return Gemma4ForConditionalGeneration.from_pretrained(model_name_or_path, **model_kwargs)
    return AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)


def _resolve_lora_target_modules(model: torch.nn.Module, model_name_or_path: str, raw_targets: str) -> list[str] | str:
    targets = [x.strip() for x in raw_targets.split(",") if x.strip()]
    if targets == ["all-linear"]:
        return "all-linear"
    if not _is_gemma4_model(model_name_or_path):
        return targets

    wanted_suffixes = tuple(f".{target}" for target in targets)
    language_targets = [
        name
        for name, _module in model.named_modules()
        if name.startswith("model.language_model.layers.") and name.endswith(wanted_suffixes)
    ]
    if not language_targets:
        raise RuntimeError(
            f"Could not resolve Gemma4 LoRA targets from {targets!r}; "
            "expected modules under model.language_model.layers.*"
        )
    print(f"[lora] resolved Gemma4 language-only target modules: {len(language_targets)}", flush=True)
    return language_targets


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def resolve_dtype(dtype_str: str):
    if dtype_str == "auto":
        return "auto"
    if dtype_str == "bfloat16":
        return torch.bfloat16
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "float32":
        return torch.float32
    return "auto"


def apply_chat_template_text(
    tokenizer,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
    enable_thinking: bool,
) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


class PragmaticSFTDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer, max_length: int):
        self.items: list[dict[str, list[int]]] = []

        skipped = 0
        for row in rows:
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                skipped += 1
                continue
            if messages[-1].get("role") != "assistant":
                skipped += 1
                continue

            prompt_messages = messages[:-1]
            full_messages = messages

            prompt_text = apply_chat_template_text(
                tokenizer,
                prompt_messages,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            full_text = apply_chat_template_text(
                tokenizer,
                full_messages,
                add_generation_prompt=False,
                enable_thinking=True,
            )

            prompt_ids = tokenizer(
                prompt_text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )["input_ids"]

            full_ids = tokenizer(
                full_text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )["input_ids"]

            if len(full_ids) <= len(prompt_ids):
                skipped += 1
                continue

            labels = full_ids.copy()
            prompt_cut = min(len(prompt_ids), len(labels) - 1)
            for i in range(prompt_cut):
                labels[i] = -100

            self.items.append(
                {
                    "input_ids": full_ids,
                    "labels": labels,
                    "attention_mask": [1] * len(full_ids),
                }
            )

        print(f"[dataset] loaded={len(self.items)} skipped={skipped}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return self.items[idx]


@dataclass
class DataCollatorForCausalSFT:
    pad_token_id: int

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)

        input_ids = []
        attention_mask = []
        labels = []

        for f in features:
            seq_len = len(f["input_ids"])
            pad_len = max_len - seq_len

            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def print_trainable_ratio(model) -> None:
    trainable = 0
    total = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    ratio = (trainable / total) * 100 if total else 0.0
    print(f"[params] trainable={trainable:,} total={total:,} ({ratio:.4f}%)")


def build_training_arguments(args, has_eval: bool) -> TrainingArguments:
    kwargs: dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "overwrite_output_dir": True,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "optim": str(args.optim),
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "warmup_steps": args.warmup_steps,
        "warmup_ratio": args.warmup_ratio,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "logging_steps": args.logging_steps,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "save_strategy": "steps",
        "bf16": args.dtype == "bfloat16",
        "fp16": args.dtype == "float16",
        "report_to": [] if args.report_to.lower() == "none" else [args.report_to],
        "run_name": args.run_name,
        "dataloader_pin_memory": True,
        "remove_unused_columns": False,
        "seed": args.seed,
    }

    eval_value = "steps" if has_eval else "no"
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in ta_params:
        kwargs["eval_strategy"] = eval_value
    else:
        kwargs["evaluation_strategy"] = eval_value

    kwargs = {k: v for k, v in kwargs.items() if k in ta_params}
    return TrainingArguments(**kwargs)


def main() -> None:
    args = parse_args()
    model_spec = resolve_model_spec(args.model_name_or_path)
    base_model_name = str(model_spec["base_model"])
    lora_adapter = model_spec["lora_adapter"]

    train_rows = read_jsonl(args.train_data_path)
    val_rows = read_jsonl(args.val_data_path)

    if not train_rows:
        raise RuntimeError(f"No train rows found at {args.train_data_path}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = resolve_dtype(args.dtype)
    model_kwargs: dict[str, Any] = {}
    if dtype != "auto":
        model_kwargs["dtype"] = dtype
    if args.device_map:
        model_kwargs["device_map"] = args.device_map
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if args.dtype == "bfloat16" else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    try:
        model = _load_causal_or_conditional_model(base_model_name, model_kwargs)
    except TypeError:
        if "dtype" in model_kwargs:
            model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
        model = _load_causal_or_conditional_model(base_model_name, model_kwargs)

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    if lora_adapter:
        model = PeftModel.from_pretrained(model, str(lora_adapter), is_trainable=True)
        print(f"[lora] continuing adapter={lora_adapter} base={base_model_name}")
    else:
        target_modules = _resolve_lora_target_modules(model, base_model_name, args.lora_target_modules)
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_config)

    if not args.device_map:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"[device] loaded model on {device} (device_map disabled)")
    else:
        print(f"[device] using device_map={args.device_map}")

    if args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            input_embeddings = model.get_input_embeddings()

            def make_inputs_require_grad(_module, _inputs, output):
                output.requires_grad_(True)

            input_embeddings.register_forward_hook(make_inputs_require_grad)
        model.gradient_checkpointing_enable()

    print_trainable_ratio(model)

    train_dataset = PragmaticSFTDataset(train_rows, tokenizer, max_length=args.max_length)
    eval_dataset = PragmaticSFTDataset(val_rows, tokenizer, max_length=args.max_length) if val_rows else None

    collator = DataCollatorForCausalSFT(pad_token_id=tokenizer.pad_token_id)
    training_args = build_training_arguments(args, has_eval=eval_dataset is not None)

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": collator,
    }
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)

    resume_ckpt = args.resume_from_checkpoint if args.resume_from_checkpoint else None
    trainer.train(resume_from_checkpoint=resume_ckpt)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    print(f"[done] saved model/tokenizer to {args.output_dir}")


if __name__ == "__main__":
    main()
