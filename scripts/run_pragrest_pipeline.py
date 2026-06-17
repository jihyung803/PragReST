#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import yaml


ROOT = Path(__file__).resolve().parents[1]
STAGE_ORDER = ("data_generation", "audit", "sft_data_generation", "train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the PragReST data, filtering, SFT-data, and training stages from one YAML config."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "pragmatic_pipeline.example.yaml",
        help="Pipeline YAML config.",
    )
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated stage subset to run. Valid stages: data_generation,audit,sft_data_generation,train.",
    )
    parser.add_argument(
        "--skip",
        type=str,
        default="",
        help="Comma-separated stage subset to skip after applying config enabled flags and --only.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return parser.parse_args()


def _config_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def _repo_path(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    path = Path(raw).expanduser()
    if path.is_absolute():
        return str(path)
    return str(ROOT / path)


def _model_spec(value: Any) -> str:
    raw = str(value or "").strip()
    if "::" not in raw:
        return raw
    base_model, adapter = [part.strip() for part in raw.split("::", 1)]
    return f"{base_model}::{_repo_path(adapter)}"


def _section(mapping: dict[str, Any], *keys: str) -> dict[str, Any]:
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, dict):
            return {}
        cur = cur.get(key, {})
    return cur if isinstance(cur, dict) else {}


def _enabled(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _split_stage_list(value: str) -> set[str]:
    stages = {part.strip() for part in value.split(",") if part.strip()}
    unknown = stages - set(STAGE_ORDER)
    if unknown:
        raise SystemExit(f"Unknown stage(s): {', '.join(sorted(unknown))}")
    return stages


def _add(cmd: list[str], flag: str, value: Any, *, path: bool = False, model: bool = False) -> None:
    if value is None:
        return
    if isinstance(value, str) and not value.strip():
        return
    cmd.append(flag)
    if path:
        cmd.append(_repo_path(value))
    elif model:
        cmd.append(_model_spec(value))
    else:
        cmd.append(str(value))


def _add_true(cmd: list[str], flag: str, value: Any) -> None:
    if _enabled(value, False):
        cmd.append(flag)


def _add_bool_optional(cmd: list[str], name: str, value: Any) -> None:
    if value is None:
        return
    cmd.append(f"--{name}" if _enabled(value, False) else f"--no-{name}")


def _python(config: dict[str, Any]) -> str:
    runtime_python = _section(config, "runtime").get("python")
    return _repo_path(runtime_python) if runtime_python else sys.executable


def _vllm_args(cmd: list[str], config: dict[str, Any]) -> None:
    vllm = _section(config, "runtime", "vllm")
    _add(cmd, "--vllm_base_url", vllm.get("base_url"))
    _add(cmd, "--vllm_api_key", vllm.get("api_key"))
    _add(cmd, "--vllm_timeout", vllm.get("timeout"))
    _add(cmd, "--vllm_max_retries", vllm.get("max_retries"))
    _add(cmd, "--vllm_gpu_memory_utilization", vllm.get("gpu_memory_utilization"))
    _add(cmd, "--vllm_tensor_parallel_size", vllm.get("tensor_parallel_size"))
    _add(cmd, "--vllm_max_model_len", vllm.get("max_model_len"))


def _common_env(config: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    runtime = _section(config, "runtime")
    if runtime.get("cuda_visible_devices") is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(runtime["cuda_visible_devices"])
    if _enabled(runtime.get("clear_ld_library_path"), False):
        env["LD_LIBRARY_PATH"] = ""

    vllm = _section(config, "runtime", "vllm")
    if vllm.get("enable_v1_multiprocessing") is not None:
        env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1" if _enabled(vllm["enable_v1_multiprocessing"]) else "0"
    if vllm.get("worker_multiproc_method"):
        env["VLLM_WORKER_MULTIPROC_METHOD"] = str(vllm["worker_multiproc_method"])
    return env


def _generation_cmd(py: str, config_path: Path, config: dict[str, Any]) -> list[str]:
    stage = _section(config, "stages", "data_generation")
    models = _section(config, "models")
    domain = _section(stage, "domain")
    qa = _section(stage, "qa")
    runtime = _section(config, "runtime")

    cmd = [py, str(ROOT / "scripts" / "build_pragmatic_qa_domain_sessions.py")]
    _add(cmd, "--config_path", config_path)
    _add(cmd, "--model_name_or_path", models.get("generation") or models.get("base"), model=True)
    _add(cmd, "--gen_backend", stage.get("backend"))
    _add(cmd, "--output_path", stage.get("output"), path=True)
    _add(cmd, "--domains_output_path", stage.get("domains_output"), path=True)
    _add(cmd, "--domains_input_path", stage.get("domains_input_path"), path=True)
    _add(cmd, "--domain_count", stage.get("domain_count"))
    _add(cmd, "--items_per_domain", stage.get("items_per_domain"))
    _add(cmd, "--domain_generation_attempts", stage.get("domain_generation_attempts"))
    _add(cmd, "--qa_attempts_per_item", stage.get("qa_attempts_per_item"))
    _add(cmd, "--qa_examples_path", stage.get("qa_fewshot_examples_path"), path=True)
    _add(cmd, "--qa_few_shot_max", stage.get("qa_fewshot_max"))
    _add(cmd, "--qa_few_shot_subset_k", stage.get("qa_fewshot_subset_k"))
    _add_true(cmd, "--qa_few_shot_shuffle", stage.get("qa_fewshot_shuffle"))

    _add(cmd, "--domain_temperature", domain.get("temperature"))
    _add(cmd, "--domain_top_p", domain.get("top_p"))
    _add(cmd, "--domain_top_k", domain.get("top_k"))
    _add(cmd, "--domain_max_new_tokens", domain.get("max_new_tokens"))
    _add(cmd, "--domain_thinking_budget_tokens", domain.get("thinking_budget_tokens"))
    _add(cmd, "--domain_answer_budget_tokens", domain.get("answer_budget_tokens"))
    _add_bool_optional(cmd, "domain_enable_thinking", domain.get("enable_thinking"))

    _add(cmd, "--qa_temperature", qa.get("temperature"))
    _add(cmd, "--qa_top_p", qa.get("top_p"))
    _add(cmd, "--qa_top_k", qa.get("top_k"))
    _add(cmd, "--qa_max_new_tokens", qa.get("max_new_tokens"))
    _add(cmd, "--qa_thinking_budget_tokens", qa.get("thinking_budget_tokens"))
    _add(cmd, "--qa_answer_budget_tokens", qa.get("answer_budget_tokens"))
    _add_bool_optional(cmd, "qa_enable_thinking", qa.get("enable_thinking"))

    _add(cmd, "--seed", config.get("seed"))
    _add_true(cmd, "--overwrite", runtime.get("overwrite"))
    _add_true(cmd, "--include_domain_field", stage.get("include_domain_field", True))
    _vllm_args(cmd, config)
    return cmd


def _audit_cmd(py: str, config_path: Path, config: dict[str, Any]) -> list[str]:
    stage = _section(config, "stages", "audit")
    data_stage = _section(config, "stages", "data_generation")
    models = _section(config, "models")
    runtime = _section(config, "runtime")

    cmd = [py, str(ROOT / "scripts" / "audit_pragmatic_generated_data.py")]
    _add(cmd, "--config_path", config_path)
    _add(cmd, "--model_name_or_path", models.get("audit") or models.get("base"), model=True)
    _add(cmd, "--gen_backend", stage.get("backend"))
    _add(cmd, "--qa_input_path", stage.get("qa_input_path") or data_stage.get("output"), path=True)
    _add(cmd, "--qa_output_path", stage.get("audited_output"), path=True)
    _add(cmd, "--summary_output_path", stage.get("summary_output"), path=True)
    _add(cmd, "--filter_strategy", stage.get("filter_strategy"))
    _add(cmd, "--score_field", stage.get("score_field"))
    _add(cmd, "--quality_threshold", stage.get("quality_threshold"))
    _add(cmd, "--drop_bottom_percent", stage.get("drop_bottom_percent"))
    _add(cmd, "--bottom_percent_scope", stage.get("bottom_percent_scope"))
    _add_true(cmd, "--attach_audit_to_kept", stage.get("attach_to_kept"))
    _add(cmd, "--max_items_per_dataset", stage.get("max_items_per_dataset"))
    _add_bool_optional(cmd, "enable_thinking", stage.get("enable_thinking"))
    _add(cmd, "--temperature", stage.get("temperature"))
    _add(cmd, "--top_p", stage.get("top_p"))
    _add(cmd, "--max_new_tokens", stage.get("max_new_tokens"))
    _add(cmd, "--thinking_budget_tokens", stage.get("thinking_budget_tokens"))
    _add(cmd, "--answer_budget_tokens", stage.get("answer_budget_tokens"))
    _add(cmd, "--seed", config.get("seed"))
    _add_true(cmd, "--overwrite", runtime.get("overwrite"))
    _vllm_args(cmd, config)
    return cmd


def _sft_data_cmd(py: str, config: dict[str, Any]) -> list[str]:
    stage = _section(config, "stages", "sft_data_generation")
    audit_stage = _section(config, "stages", "audit")
    models = _section(config, "models")
    runtime = _section(config, "runtime")
    judge = _section(stage, "judge")
    star = _section(stage, "star")

    cmd = [py, str(ROOT / "scripts" / "build_pragmatic_sft_dataset_star.py")]
    _add(cmd, "--model_name_or_path", stage.get("model") or models.get("generation") or models.get("base"), model=True)
    _add(cmd, "--qa_data_path", stage.get("qa_audited_input") or audit_stage.get("audited_output"), path=True)
    _add(cmd, "--output_path", stage.get("output_all"), path=True)
    _add(cmd, "--train_output_path", stage.get("output_train"), path=True)
    _add(cmd, "--val_output_path", stage.get("output_val"), path=True)
    _add(cmd, "--val_ratio", stage.get("val_ratio"))
    _add(cmd, "--seed", config.get("seed"))
    _add(cmd, "--max_samples", stage.get("max_samples"))
    _add(cmd, "--max_qa_samples", stage.get("max_qa_samples"))
    _add(cmd, "--qa_mix_weight", stage.get("qa_mix_weight"))
    _add(cmd, "--teacher_prompt_mode", stage.get("teacher_prompt_mode"))
    _add(cmd, "--thinking_budget_tokens", stage.get("thinking_budget_tokens"))
    _add(cmd, "--answer_budget_tokens", stage.get("answer_budget_tokens"))
    _add(cmd, "--temperature", stage.get("temperature"))
    _add(cmd, "--top_p", stage.get("top_p"))
    _add(cmd, "--top_k", stage.get("top_k"))
    _add(cmd, "--max_input_length", stage.get("max_input_length"))
    _add(cmd, "--gen_backend", stage.get("backend"))
    _add_true(cmd, "--keep_pragmatic_instruction_in_sft", stage.get("keep_pragmatic_instruction_in_sft"))

    _add_true(cmd, "--judge_drop_without_refine", judge.get("drop_without_refine"))
    _add(cmd, "--judge_refine_rounds", judge.get("refine_rounds"))
    _add(cmd, "--judge_method", judge.get("method"))
    _add(cmd, "--judge_margin_threshold", judge.get("margin_threshold"))
    _add(cmd, "--judge_score_batch_size", judge.get("score_batch_size"))
    _add(cmd, "--judge_confidence_threshold", judge.get("confidence_threshold"))
    _add_bool_optional(cmd, "judge_enable_thinking", judge.get("enable_thinking"))
    _add(cmd, "--judge_max_new_tokens", judge.get("max_new_tokens"))
    _add(cmd, "--judge_temperature", judge.get("temperature"))
    _add(cmd, "--judge_top_p", judge.get("top_p"))
    _add(cmd, "--judge_top_k", judge.get("top_k"))
    _add_true(cmd, "--star_rationalize_failed", star.get("rationalize_failed"))
    _add(cmd, "--star_rationalization_rounds", star.get("rationalization_rounds"))
    _add_true(cmd, "--overwrite", runtime.get("overwrite"))
    _vllm_args(cmd, config)
    return cmd


def _train_cmd(py: str, config: dict[str, Any]) -> list[str]:
    stage = _section(config, "stages", "train")
    sft_stage = _section(config, "stages", "sft_data_generation")
    models = _section(config, "models")
    mode = str(stage.get("mode", "lora")).strip().lower()
    full = _section(stage, "full_ft")

    if mode in {"full", "full_ft", "full-finetune", "full_finetune"}:
        script = ROOT / "scripts" / "train_pragmatic_sft_full.py"
        launcher = str(full.get("launcher", "") or "").strip().lower()
        nproc = int(full.get("nproc_per_node") or 1)
        nnodes = int(full.get("nnodes") or 1)
        if launcher == "torchrun" or nproc > 1 or nnodes > 1:
            cmd = [
                "torchrun",
                "--nproc_per_node",
                str(nproc),
                "--nnodes",
                str(nnodes),
                "--node_rank",
                str(full.get("node_rank", 0)),
            ]
            if full.get("master_addr"):
                cmd.extend(["--master_addr", str(full["master_addr"])])
            if full.get("master_port"):
                cmd.extend(["--master_port", str(full["master_port"])])
            cmd.append(str(script))
        else:
            cmd = [py, str(script)]
    elif mode == "lora":
        cmd = [py, str(ROOT / "scripts" / "train_pragmatic_sft_lora.py")]
    else:
        raise SystemExit(f"Unsupported train mode: {mode}")

    _add(cmd, "--model_name_or_path", models.get("train") or models.get("base"), model=(mode == "lora"))
    _add(cmd, "--train_data_path", stage.get("train_data_path") or sft_stage.get("output_train"), path=True)
    _add(cmd, "--val_data_path", stage.get("val_data_path") or sft_stage.get("output_val"), path=True)
    _add(cmd, "--output_dir", stage.get("output_dir"), path=True)
    _add(cmd, "--max_length", stage.get("max_length"))
    _add(cmd, "--learning_rate", stage.get("learning_rate"))
    _add(cmd, "--num_train_epochs", stage.get("epochs"))
    _add(cmd, "--max_steps", full.get("max_steps") if mode != "lora" else stage.get("max_steps"))
    _add(cmd, "--warmup_steps", full.get("warmup_steps") if mode != "lora" else stage.get("warmup_steps"))
    _add(cmd, "--warmup_ratio", full.get("warmup_ratio") if mode != "lora" else stage.get("warmup_ratio"))
    _add(cmd, "--per_device_train_batch_size", stage.get("batch_size"))
    _add(cmd, "--per_device_eval_batch_size", full.get("eval_batch_size") or stage.get("eval_batch_size"))
    _add(cmd, "--gradient_accumulation_steps", stage.get("grad_accum"))
    _add(cmd, "--eval_steps", stage.get("eval_steps"))
    _add(cmd, "--save_steps", stage.get("save_steps"))
    _add(cmd, "--logging_steps", full.get("logging_steps") or stage.get("logging_steps"))
    _add(cmd, "--seed", config.get("seed"))
    _add(cmd, "--dtype", full.get("dtype") or stage.get("dtype"))
    _add(cmd, "--optim", full.get("optim") or stage.get("optim"))
    _add(cmd, "--report_to", stage.get("report_to"))
    _add(cmd, "--run_name", stage.get("run_name"))
    _add(cmd, "--resume_from_checkpoint", stage.get("resume_from_checkpoint"))

    if mode == "lora":
        lora = _section(stage, "lora")
        _add(cmd, "--lora_r", lora.get("r"))
        _add(cmd, "--lora_alpha", lora.get("alpha"))
        _add(cmd, "--lora_dropout", lora.get("dropout"))
        _add(cmd, "--lora_target_modules", lora.get("target_modules"))
        _add_true(cmd, "--gradient_checkpointing", lora.get("gradient_checkpointing"))
        _add_true(cmd, "--load_in_4bit", lora.get("load_in_4bit"))
    else:
        _add(cmd, "--save_total_limit", full.get("save_total_limit"))
        _add(cmd, "--fsdp", full.get("fsdp"))
        _add(cmd, "--fsdp_transformer_layer_cls_to_wrap", full.get("fsdp_layer"))
        _add(cmd, "--fsdp_state_dict_type", full.get("fsdp_state_dict_type"))
        _add_bool_optional(cmd, "fsdp_use_orig_params", full.get("fsdp_use_orig_params"))
        _add_bool_optional(cmd, "fsdp_cpu_ram_efficient_loading", full.get("fsdp_cpu_ram_efficient_loading"))
        _add(cmd, "--deepspeed", full.get("deepspeed"))
        _add_bool_optional(cmd, "gradient_checkpointing", full.get("gradient_checkpointing"))
        _add_bool_optional(cmd, "save_only_model", full.get("save_only_model"))

    return cmd


def _stage_commands(config_path: Path, config: dict[str, Any]) -> dict[str, list[str]]:
    py = _python(config)
    return {
        "data_generation": _generation_cmd(py, config_path, config),
        "audit": _audit_cmd(py, config_path, config),
        "sft_data_generation": _sft_data_cmd(py, config),
        "train": _train_cmd(py, config),
    }


def _active_stages(config: dict[str, Any], only: str, skip: str) -> list[str]:
    stages_config = _section(config, "stages")
    active = [
        stage
        for stage in STAGE_ORDER
        if _enabled(_section(stages_config, stage).get("enabled"), False)
    ]
    only_set = _split_stage_list(only)
    skip_set = _split_stage_list(skip)
    if only_set:
        active = [stage for stage in STAGE_ORDER if stage in only_set]
    return [stage for stage in active if stage not in skip_set]


def _print_cmd(stage: str, cmd: Iterable[str]) -> None:
    print(f"[{stage}]")
    print("  " + " ".join(shlex.quote(part) for part in cmd))


def main() -> None:
    args = parse_args()
    config_path = _config_path(args.config)
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    active = _active_stages(config, args.only, args.skip)
    if not active:
        print("No stages selected. Enable stages in the config or pass --only.")
        return

    commands = _stage_commands(config_path, config)
    env = _common_env(config)

    for stage in active:
        cmd = commands[stage]
        _print_cmd(stage, cmd)
        if args.dry_run:
            continue
        subprocess.run(cmd, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
