#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_pragmatic_sft_lora import (  # noqa: E402
    DataCollatorForCausalSFT,
    PragmaticSFTDataset,
    print_trainable_ratio,
    read_jsonl,
    resolve_dtype,
)


def _is_ministral3_model(model_name_or_path: str) -> bool:
    raw = str(model_name_or_path or "").strip().lower()
    return "ministral-3" in raw or "ministral3" in raw


def _is_gemma4_model(model_name_or_path: str) -> bool:
    raw = str(model_name_or_path or "").strip().lower()
    return "gemma-4" in raw or "gemma4" in raw


def _load_tokenizer(model_name_or_path: str, *, trust_remote_code: bool):
    kwargs: dict[str, Any] = {"trust_remote_code": bool(trust_remote_code)}
    if _is_ministral3_model(model_name_or_path):
        kwargs["fix_mistral_regex"] = True
    try:
        return AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)
    except TypeError:
        kwargs.pop("fix_mistral_regex", None)
        return AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)


def _load_full_ft_model(model_name_or_path: str, model_kwargs: dict[str, Any]):
    if _is_ministral3_model(model_name_or_path):
        from transformers import Mistral3ForConditionalGeneration

        return Mistral3ForConditionalGeneration.from_pretrained(model_name_or_path, **model_kwargs)
    if _is_gemma4_model(model_name_or_path):
        from transformers import Gemma4ForConditionalGeneration

        return Gemma4ForConditionalGeneration.from_pretrained(model_name_or_path, **model_kwargs)
    return AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)


def _resolve_fsdp_layer_cls_to_wrap(model, requested: str) -> str:
    requested = str(requested or "").strip()
    if not requested:
        return requested

    available = {module.__class__.__name__ for _, module in model.named_modules()}
    if requested in available:
        return requested

    alias_map = {
        "MistralDecoderLayer": "Ministral3DecoderLayer",
    }
    alias = alias_map.get(requested)
    if alias and alias in available:
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"[train-args] remapping fsdp layer {requested!r} -> {alias!r}")
        return alias

    decoder_candidates = sorted(name for name in available if name.endswith("DecoderLayer"))
    if len(decoder_candidates) == 1:
        chosen = decoder_candidates[0]
        if int(os.environ.get("RANK", "0")) == 0:
            print(
                f"[train-args] requested fsdp layer {requested!r} not found; "
                f"using only available decoder layer {chosen!r}"
            )
        return chosen

    return requested


def _save_dtype_name(dtype_arg: str, save_dtype_arg: str) -> str:
    requested = str(save_dtype_arg or "").strip().lower()
    if requested in {"", "training"}:
        requested = str(dtype_arg or "").strip().lower()
    aliases = {
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp16": "float16",
        "float16": "float16",
        "fp32": "float32",
        "float32": "float32",
        "auto": "",
        "none": "",
        "off": "",
        "disabled": "",
    }
    if requested not in aliases:
        raise ValueError(f"Unsupported save dtype: {save_dtype_arg!r}")
    return aliases[requested]


def _torch_dtype_from_name(dtype_name: str) -> torch.dtype:
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch dtype name: {dtype_name!r}")


def _cast_safetensors_file(path: Path, target_dtype: torch.dtype) -> tuple[int, int]:
    tensors: dict[str, torch.Tensor] = {}
    converted = 0
    preserved = 0
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            if tensor.is_floating_point() and tensor.dtype != target_dtype:
                tensor = tensor.to(target_dtype)
                converted += 1
            else:
                preserved += 1
            tensors[key] = tensor
    save_file(tensors, path, metadata=metadata)
    return converted, preserved


def _cast_saved_model_dtype(output_dir: Path, dtype_name: str) -> None:
    if not dtype_name:
        return
    target_dtype = _torch_dtype_from_name(dtype_name)
    safetensor_files = sorted(output_dir.glob("*.safetensors"))
    if not safetensor_files:
        print(f"[save-dtype] no safetensors files found in {output_dir}; skipping dtype cast")
        return

    total_converted = 0
    total_preserved = 0
    print(f"[save-dtype] casting saved floating weights to {dtype_name}")
    for path in safetensor_files:
        converted, preserved = _cast_safetensors_file(path, target_dtype)
        total_converted += converted
        total_preserved += preserved
        print(f"[save-dtype] {path.name}: converted={converted} preserved={preserved}")

    config_path = output_dir / "config.json"
    if config_path.exists():
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        cfg["torch_dtype"] = dtype_name
        config_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[save-dtype] done converted={total_converted} preserved={total_preserved}")


class SaveDTypeCastCallback(TrainerCallback):
    def __init__(self, dtype_name: str) -> None:
        self.dtype_name = str(dtype_name or "")

    def on_save(self, args, state, control, **kwargs):
        if not self.dtype_name:
            return control
        if not bool(getattr(state, "is_world_process_zero", False)):
            return control
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if checkpoint_dir.exists():
            _cast_saved_model_dtype(checkpoint_dir, self.dtype_name)
        return control


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full fine-tune Qwen-style causal LM on pragmatic SFT data.")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--train_data_path", type=Path, required=True)
    parser.add_argument("--val_data_path", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)

    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="Absolute warmup optimizer steps. If >0, this overrides --warmup_ratio.",
    )
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_total_limit", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--run_name", type=str, default="pragmatic-sft-full")
    parser.add_argument("--resume_from_checkpoint", type=str, default="")
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument(
        "--fsdp",
        type=str,
        default="full_shard auto_wrap",
        help="TrainingArguments fsdp value. Empty disables FSDP.",
    )
    parser.add_argument("--fsdp_transformer_layer_cls_to_wrap", type=str, default="Qwen3DecoderLayer")
    parser.add_argument("--fsdp_state_dict_type", type=str, default="FULL_STATE_DICT")
    parser.add_argument("--fsdp_use_orig_params", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fsdp_cpu_ram_efficient_loading", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deepspeed", type=str, default="")
    parser.add_argument("--save_safetensors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--save_dtype",
        type=str,
        default="training",
        choices=["training", "auto", "none", "bfloat16", "bf16", "float16", "fp16", "float32", "fp32"],
        help=(
            "Dtype for final saved floating-point safetensors. 'training' uses --dtype; "
            "'auto'/'none' leaves Trainer output unchanged."
        ),
    )
    parser.add_argument(
        "--save_only_model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save only model weights at checkpoints. This avoids very large FSDP optimizer-state files.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_training_arguments(args: argparse.Namespace, has_eval: bool) -> TrainingArguments:
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    kwargs: dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "num_train_epochs": float(args.num_train_epochs),
        "max_steps": int(args.max_steps),
        "warmup_ratio": float(args.warmup_ratio),
        "per_device_train_batch_size": int(args.per_device_train_batch_size),
        "per_device_eval_batch_size": int(args.per_device_eval_batch_size),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "logging_steps": int(args.logging_steps),
        "eval_steps": int(args.eval_steps),
        "save_steps": int(args.save_steps),
        "save_strategy": "steps",
        "save_total_limit": int(args.save_total_limit),
        "bf16": args.dtype == "bfloat16",
        "fp16": args.dtype == "float16",
        "report_to": [] if str(args.report_to).lower() == "none" else [args.report_to],
        "run_name": str(args.run_name),
        "dataloader_pin_memory": True,
        "remove_unused_columns": False,
        "seed": int(args.seed),
        "optim": str(args.optim),
        "lr_scheduler_type": str(args.lr_scheduler_type),
        "max_grad_norm": float(args.max_grad_norm),
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "gradient_checkpointing_kwargs": {"use_reentrant": False} if bool(args.gradient_checkpointing) else None,
        "save_safetensors": bool(args.save_safetensors),
    }

    eval_value = "steps" if has_eval else "no"
    kwargs["eval_strategy" if "eval_strategy" in ta_params else "evaluation_strategy"] = eval_value
    if "overwrite_output_dir" in ta_params:
        kwargs["overwrite_output_dir"] = False
    if "save_only_model" in ta_params:
        kwargs["save_only_model"] = bool(args.save_only_model)
    if int(args.warmup_steps) > 0:
        kwargs["warmup_steps"] = int(args.warmup_steps)
        kwargs["warmup_ratio"] = 0.0

    if str(args.fsdp).strip():
        kwargs["fsdp"] = str(args.fsdp).strip()
        kwargs["fsdp_config"] = {
            "transformer_layer_cls_to_wrap": str(args.fsdp_transformer_layer_cls_to_wrap),
            "state_dict_type": str(args.fsdp_state_dict_type),
            "use_orig_params": bool(args.fsdp_use_orig_params),
            "cpu_ram_efficient_loading": bool(args.fsdp_cpu_ram_efficient_loading),
        }
    if str(args.deepspeed).strip():
        kwargs["deepspeed"] = str(args.deepspeed).strip()

    filtered_kwargs = {key: value for key, value in kwargs.items() if key in ta_params and value is not None}
    dropped_kwargs = sorted(key for key in kwargs if key not in ta_params and kwargs[key] is not None)
    if dropped_kwargs and int(os.environ.get("RANK", "0")) == 0:
        print(f"[train-args] dropping unsupported TrainingArguments kwargs: {', '.join(dropped_kwargs)}")

    return TrainingArguments(**filtered_kwargs)


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if torch.cuda.is_available() and local_rank >= 0:
        torch.cuda.set_device(local_rank)
        print(f"[distributed] local_rank={local_rank} cuda_device={torch.cuda.current_device()}")
    if int(os.environ.get("RANK", "0")) == 0:
        print(
            "[train-args] "
            f"fsdp={args.fsdp!r} "
            f"fsdp_state_dict_type={args.fsdp_state_dict_type!r} "
            f"save_only_model={bool(args.save_only_model)!r} "
            f"gradient_checkpointing={bool(args.gradient_checkpointing)!r}"
        )

    train_rows = read_jsonl(args.train_data_path)
    val_rows = read_jsonl(args.val_data_path) if args.val_data_path is not None else []
    if not train_rows:
        raise RuntimeError(f"No train rows found at {args.train_data_path}")

    tokenizer = _load_tokenizer(args.model_name_or_path, trust_remote_code=bool(args.trust_remote_code))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dtype = resolve_dtype(args.dtype)
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": bool(args.trust_remote_code),
        "low_cpu_mem_usage": True,
    }
    if dtype != "auto":
        model_kwargs["torch_dtype"] = dtype

    model = _load_full_ft_model(args.model_name_or_path, model_kwargs)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if bool(args.gradient_checkpointing) and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    args.fsdp_transformer_layer_cls_to_wrap = _resolve_fsdp_layer_cls_to_wrap(
        model, args.fsdp_transformer_layer_cls_to_wrap
    )

    print_trainable_ratio(model)

    train_dataset = PragmaticSFTDataset(train_rows, tokenizer, max_length=int(args.max_length))
    eval_dataset = PragmaticSFTDataset(val_rows, tokenizer, max_length=int(args.max_length)) if val_rows else None
    collator = DataCollatorForCausalSFT(pad_token_id=int(tokenizer.pad_token_id))
    training_args = build_training_arguments(args, has_eval=eval_dataset is not None)
    save_dtype = _save_dtype_name(args.dtype, args.save_dtype)
    callbacks = [SaveDTypeCastCallback(save_dtype)] if bool(args.save_safetensors) and save_dtype else None

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=collator,
        callbacks=callbacks,
    )

    resume_ckpt = str(args.resume_from_checkpoint).strip() or None
    trainer.train(resume_from_checkpoint=resume_ckpt)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output_dir))
    if trainer.is_world_process_zero():
        if bool(args.save_safetensors):
            _cast_saved_model_dtype(args.output_dir, save_dtype)
        tokenizer.save_pretrained(str(args.output_dir))
        metadata = {
            "model_name_or_path": args.model_name_or_path,
            "train_data_path": str(args.train_data_path),
            "val_data_path": str(args.val_data_path) if args.val_data_path is not None else "",
            "max_length": int(args.max_length),
            "learning_rate": float(args.learning_rate),
            "num_train_epochs": float(args.num_train_epochs),
            "warmup_steps": int(args.warmup_steps),
            "warmup_ratio": float(args.warmup_ratio),
            "optim": str(args.optim),
            "fsdp": str(args.fsdp),
            "fsdp_transformer_layer_cls_to_wrap": str(args.fsdp_transformer_layer_cls_to_wrap),
            "dtype": str(args.dtype),
            "save_dtype": str(save_dtype),
        }
        (args.output_dir / "full_finetune_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[done] saved full fine-tuned model/tokenizer to {args.output_dir}")
    if hasattr(trainer, "accelerator"):
        trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
