#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
ROOT_PATH = Path(ROOT)

from src.config import load_config
from src.utils.vllm_local_batched import VLLMLocalBatchedGenerator
from src.utils.io import set_seed
from src.utils.model_system_prompts import adapt_system_prompt
from src.utils.thinking_tags import normalize_early_stopping_text, strip_prefixed_thinking


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build pragmatic QA dataset using independent domain-seeded sessions. "
            "Step1: generate domain seeds. Step2: generate QA per domain."
        )
    )
    parser.add_argument("--config_path", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--output_path",
        type=Path,
        default=ROOT_PATH / "data/pragmaticQA_dataset_Qwen14b.jsonl",
        help="Final pragmatic QA dataset path.",
    )
    parser.add_argument(
        "--domains_output_path",
        type=Path,
        default=ROOT_PATH / "data/pragmaticQA_domain_keywords_Qwen14b.jsonl",
        help="Where to store generated domain keywords.",
    )
    parser.add_argument(
        "--domains_input_path",
        type=Path,
        default=None,
        help="Optional pre-defined domain keyword file (jsonl/json/txt). If set, skip domain generation.",
    )
    parser.add_argument("--domain_count", type=int, default=100)
    parser.add_argument(
        "--items_per_domain",
        type=int,
        default=1,
        help="Number of QA items to generate for each (domain, section) pair.",
    )
    parser.add_argument("--domain_generation_attempts", type=int, default=20)
    parser.add_argument("--qa_attempts_per_item", type=int, default=6)

    parser.add_argument("--model_role", type=str, default="")
    parser.add_argument("--model_name_or_path", type=str, default="")
    parser.add_argument("--dtype", type=str, default="")
    parser.add_argument("--device_map", type=str, default="")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument(
        "--gen_backend",
        type=str,
        default="hf",
        choices=["hf", "vllm_api", "vllm_local", "vllm_local_batched"],
        help="Generation backend.",
    )
    parser.add_argument(
        "--vllm_base_url",
        type=str,
        default="http://localhost:8000",
        help="vLLM OpenAI-compatible server URL.",
    )
    parser.add_argument(
        "--vllm_model_name",
        type=str,
        default="",
        help="Served model name in vLLM. Defaults to --model_name_or_path or model config name.",
    )
    parser.add_argument(
        "--vllm_api_key",
        type=str,
        default="EMPTY",
        help="Bearer token for vLLM OpenAI API. Use EMPTY for local server without auth.",
    )
    parser.add_argument("--vllm_timeout", type=float, default=180.0)
    parser.add_argument("--vllm_max_retries", type=int, default=3)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--vllm_max_model_len", type=int, default=0)

    parser.add_argument("--domain_temperature", type=float, default=None)
    parser.add_argument("--domain_top_p", type=float, default=None)
    parser.add_argument("--domain_top_k", type=int, default=None)
    parser.add_argument("--domain_max_new_tokens", type=int, default=2048)

    parser.add_argument("--qa_temperature", type=float, default=None)
    parser.add_argument("--qa_top_p", type=float, default=None)
    parser.add_argument("--qa_top_k", type=int, default=None)
    parser.add_argument("--qa_max_new_tokens", type=int, default=2048)

    parser.add_argument("--domain_thinking_budget_tokens", type=int, default=0)
    parser.add_argument("--domain_answer_budget_tokens", type=int, default=0)
    parser.add_argument("--qa_thinking_budget_tokens", type=int, default=0)
    parser.add_argument("--qa_answer_budget_tokens", type=int, default=0)
    parser.add_argument(
        "--thinking_early_stopping_text",
        type=str,
        default=(
            "\n\nConsidering the limited time by the user, I have to give the solution "
            "based on the thinking directly now.\n[/think]\n\n"
        ),
        help="Early-stopping text appended when thinking budget is reached before the model emits a think-closing tag.",
    )

    parser.add_argument(
        "--enable_thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override enable_thinking for both domain and QA generation.",
    )
    parser.add_argument(
        "--domain_enable_thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override enable_thinking for domain generation only.",
    )
    parser.add_argument(
        "--qa_enable_thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override enable_thinking for QA/item generation only.",
    )

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--include_domain_field", action="store_true")
    parser.add_argument(
        "--system_prompt_style",
        type=str,
        default="plain",
        choices=["plain", "ministral_reasoning"],
        help="Optional wrapper style for system prompts used during domain and QA generation.",
    )
    parser.add_argument(
        "--system_prompt_repo_id",
        type=str,
        default="",
        help="HF repo id for external model system prompt assets.",
    )
    parser.add_argument(
        "--system_prompt_filename",
        type=str,
        default="",
        help="Filename to load from --system_prompt_repo_id when system prompt wrapping is enabled.",
    )

    parser.add_argument(
        "--qa_examples_path",
        type=Path,
        default=ROOT_PATH / "data/pragmatic_mcq_examples.jsonl",
        help="Few-shot QA examples path (jsonl with section/content/question/answer).",
    )
    parser.add_argument(
        "--qa_few_shot_max",
        type=int,
        default=7,
        help="Max few-shot examples per section to inject into QA generation prompt.",
    )
    parser.add_argument(
        "--qa_few_shot_shuffle",
        action="store_true",
        help="Shuffle few-shot example order per generated item.",
    )
    parser.add_argument(
        "--qa_few_shot_subset_k",
        type=int,
        default=3,
        help="Per item, randomly sample up to k few-shot examples from the section pool (0 disables subset sampling).",
    )

    return parser.parse_args()


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _build_model(cfg: dict[str, Any], role: str) -> Any:
    from src.models.hf import HFPolicyModel

    model_type = str((cfg or {}).get("type", "hf")).strip().lower()
    if model_type != "hf":
        raise ValueError(f"Unsupported model type for role={role}: {model_type}")
    name = str((cfg or {}).get("name", "")).strip()
    if not name:
        raise ValueError(f"Missing model name for role={role}")

    return HFPolicyModel(
        name,
        lora=bool(cfg.get("lora", False)),
        lora_r=int(cfg.get("lora_r", 8)),
        lora_alpha=int(cfg.get("lora_alpha", 16)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        lora_target_modules=cfg.get("lora_target_modules", None),
        adapter_name=cfg.get("adapter_name", role),
        share_base=bool(cfg.get("share_base", False)),
        device=cfg.get("device", None),
        dtype=cfg.get("dtype", None),
        device_map=cfg.get("device_map", None),
        load_in_4bit=bool(cfg.get("load_in_4bit", False)),
        load_in_8bit=bool(cfg.get("load_in_8bit", False)),
    )


def _strip_thinking(text: str) -> str:
    return strip_prefixed_thinking(text)


def _strip_code_fence(text: str) -> str:
    raw = str(text or "").strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
    if m:
        return m.group(1).strip()
    return raw


def _find_last_json_span(text: str) -> tuple[int, int] | None:
    in_string = False
    escape = False
    depth = 0
    start = None
    last_span = None
    for idx, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    last_span = (start, idx)
    return last_span


def _parse_jsonish(text: str) -> dict[str, Any] | list[Any] | None:
    cleaned = _strip_thinking(text)
    if not cleaned:
        return None

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_\-]*", "", cleaned).strip()
        cleaned = cleaned.rstrip("`").strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, (dict, list)):
            return parsed
    except Exception:
        pass

    span = _find_last_json_span(cleaned)
    if span is not None:
        try:
            s, e = span
            parsed = json.loads(cleaned[s : e + 1])
            if isinstance(parsed, (dict, list)):
                return parsed
        except Exception:
            return None
    return None


def _parse_jsonl_records_min(text: str, required_keys: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cleaned = _strip_thinking(text)
    for line in cleaned.splitlines():
        row = line.strip()
        if not row:
            continue
        row = re.sub(r"^\d+[.)]\s*", "", row)
        try:
            obj = json.loads(row)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        if not all(k in obj for k in required_keys):
            continue
        out.append({k: obj.get(k) for k in required_keys})
    return out


def _parse_cqa_tuple_records(text: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []

    def _push(content: Any, question: Any, answer: Any) -> None:
        c = str(content or "").strip()
        q = str(question or "").strip()
        a = str(answer or "").strip()
        if c and q and a:
            out.append({"content": c, "question": q, "answer": a})

    cleaned = _strip_thinking(text)
    if not cleaned:
        return out

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_\-]*", "", cleaned).strip()
        cleaned = cleaned.rstrip("`").strip()

    try:
        payload = ast.literal_eval(cleaned)
    except Exception:
        payload = None

    if isinstance(payload, dict) and all(k in payload for k in ("content", "question", "answer")):
        _push(payload.get("content"), payload.get("question"), payload.get("answer"))
        return out

    if isinstance(payload, (tuple, list)):
        if len(payload) == 3 and not any(isinstance(x, (tuple, list, dict)) for x in payload):
            _push(payload[0], payload[1], payload[2])
            return out
        for item in payload:
            if isinstance(item, dict) and all(k in item for k in ("content", "question", "answer")):
                _push(item.get("content"), item.get("question"), item.get("answer"))
            elif isinstance(item, (tuple, list)) and len(item) == 3:
                _push(item[0], item[1], item[2])
        if out:
            return out

    for line in cleaned.splitlines():
        row = re.sub(r"^\d+[.)]\s*", "", line.strip())
        if not row:
            continue
        try:
            parsed = ast.literal_eval(row)
        except Exception:
            continue
        if isinstance(parsed, dict) and all(k in parsed for k in ("content", "question", "answer")):
            _push(parsed.get("content"), parsed.get("question"), parsed.get("answer"))
        elif isinstance(parsed, (tuple, list)) and len(parsed) == 3:
            _push(parsed[0], parsed[1], parsed[2])

    return out


def _normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _clean_domain(text: Any) -> str:
    s = str(text or "").strip()
    s = re.sub(r"^[-*]+\s*", "", s)
    s = re.sub(r"^\d+[.)]\s*", "", s)
    s = s.strip().strip('"').strip("'")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _parse_domain_keywords(raw: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    def _add(value: Any) -> None:
        item = _clean_domain(value)
        if not item:
            return
        key = _normalize_text(item)
        if not key or key in seen:
            return
        seen.add(key)
        out.append(item)

    parsed = _parse_jsonish(raw)
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, str):
                _add(item)
            elif isinstance(item, dict):
                _add(item.get("domain") or item.get("keyword") or item.get("seed"))
    elif isinstance(parsed, dict):
        for key in ("domains", "keywords", "domain_keywords", "items"):
            values = parsed.get(key)
            if isinstance(values, list):
                for value in values:
                    if isinstance(value, str):
                        _add(value)
                    elif isinstance(value, dict):
                        _add(value.get("domain") or value.get("keyword") or value.get("seed"))
                break
        if not out:
            _add(parsed.get("domain") or parsed.get("keyword") or parsed.get("seed"))

    for row in _parse_jsonl_records_min(raw, ["domain"]):
        _add(row.get("domain"))

    if out:
        return out

    cleaned = _strip_thinking(raw)
    for line in cleaned.splitlines():
        item = _clean_domain(line)
        if not item:
            continue
        if item.startswith("{") or item.startswith("["):
            continue
        _add(item)

    return out


def _parse_qa_candidates(raw: str) -> list[dict[str, str]]:
    required = ["content", "question", "answer"]
    out: list[dict[str, str]] = []

    # Prefer labeled plain-text parsing first (higher quality than strict JSON outputs).
    cleaned = _strip_code_fence(_strip_thinking(raw))
    m_content = re.search(r"(?is)\bcontent\s*:\s*(.+?)\s*(?=\n\s*question\s*:)", cleaned)
    m_question = re.search(r"(?is)\bquestion\s*:\s*(.+?)\s*(?=\n\s*answer\s*:)", cleaned)
    m_answer = re.search(r"(?is)\banswer\s*:\s*(.+?)\s*$", cleaned)
    if m_content and m_question and m_answer:
        out.append(
            {
                "content": m_content.group(1).strip(),
                "question": m_question.group(1).strip(),
                "answer": m_answer.group(1).strip(),
            }
        )
        return out

    for row in _parse_jsonl_records_min(raw, required):
        out.append({k: str(row.get(k) or "").strip() for k in required})
    if out:
        return out

    parsed = _parse_jsonish(raw)
    if isinstance(parsed, dict):
        if all(k in parsed for k in required):
            out.append({k: str(parsed.get(k) or "").strip() for k in required})
    elif isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict) and all(k in item for k in required):
                out.append({k: str(item.get(k) or "").strip() for k in required})
    if out:
        return out

    return _parse_cqa_tuple_records(raw)


def _validate_qa_candidate(item: dict[str, str]) -> dict[str, str] | None:
    content = str(item.get("content") or "").strip()
    question = str(item.get("question") or "").strip()
    answer = str(item.get("answer") or "").strip()

    if not content or not question or not answer:
        return None
    if len(content) < 20 or len(question) < 8 or len(answer) < 3:
        return None
    if "?" not in question:
        question = question.rstrip(".") + "?"

    return {"content": content, "question": question, "answer": answer}


def _tuple_quote(text: Any) -> str:
    return json.dumps(str(text or ""), ensure_ascii=False)


def _to_cqa_tuple(content: str, question: str, answer: str) -> str:
    return f"({_tuple_quote(content)}, {_tuple_quote(question)}, {_tuple_quote(answer)})"


def _section_norm(name: Any) -> str:
    return re.sub(r"\s+", " ", str(name or "").strip().lower())


def _load_few_shot_by_section(path: Path | None, few_shot_max: int) -> dict[str, list[dict[str, str]]]:
    if path is None:
        return {}

    raw = str(path).strip()
    if not raw or raw == ".":
        return {}

    p = Path(raw)
    if not p.exists() or p.is_dir():
        return {}

    grouped: dict[str, list[dict[str, str]]] = {}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            row = line.strip()
            if not row:
                continue
            try:
                obj = json.loads(row)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue

            section = str(obj.get("section") or "").strip()
            content = str(obj.get("content") or "").strip()
            question = str(obj.get("question") or "").strip()
            answer = str(obj.get("answer") or "").strip()
            if not section or not content or not question or not answer:
                continue

            key = _section_norm(section)
            bucket = grouped.setdefault(key, [])
            if few_shot_max > 0 and len(bucket) >= few_shot_max:
                continue
            bucket.append({"content": content, "question": question, "answer": answer})

    return grouped


def _format_few_shot_jsonl(examples: list[dict[str, str]]) -> str:
    if not examples:
        return ""
    blocks: list[str] = []
    for idx, ex in enumerate(examples, start=1):
        blocks.append(
            "\n".join(
                [
                    f"Example {idx}",
                    f"Content: {ex.get('content', '')}",
                    f"Question: {ex.get('question', '')}",
                    f"Answer: {ex.get('answer', '')}",
                ]
            )
        )
    return "\n\n".join(blocks)


def _load_sections(cfg_raw: dict[str, Any]) -> list[dict[str, str]]:
    dataset_cfg = cfg_raw.get("dataset_gen", {}) if isinstance(cfg_raw, dict) else {}
    raw_sections = dataset_cfg.get("sections", [])

    out: list[dict[str, str]] = []
    if isinstance(raw_sections, list):
        for item in raw_sections:
            if isinstance(item, dict):
                name = str(item.get("name") or "").strip()
                description = str(item.get("description") or "").strip()
            else:
                name = str(item or "").strip()
                description = ""
            if not name:
                continue
            out.append({"name": name, "description": description})

    if out:
        return out

    return [{"name": "Pragmatic QA", "description": "Pragmatic inference over contextual meaning."}]


def _load_domain_file(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"domains_input_path not found: {path}")
    if path.is_dir():
        raise IsADirectoryError(
            f"domains_input_path points to a directory: {path}. "
            "Pass a file path or omit --domains_input_path."
        )

    text = path.read_text(encoding="utf-8")
    parsed = _parse_domain_keywords(text)
    if parsed:
        return parsed

    out: list[str] = []
    for line in text.splitlines():
        item = _clean_domain(line)
        if item:
            out.append(item)
    if not out:
        raise RuntimeError(f"No domain keywords parsed from {path}")
    return out


def _default_domain_system_prompt() -> str:
    return (
        "You generate diverse domain seed keywords.\n"
        "Return keyword outputs only.\n"
        "Domains must be natural and intuitive for humans.\n"
        "Each domain seed must contain 2-4 words."
    )


def _default_domain_user_prompt(request_count: int, existing_domains: str) -> str:
    return (
        f"Generate exactly {request_count} domain keywords.\n"
        "Output plain text only as a numbered list, one domain seed per line.\n"
        f"Use consecutive numbering from 1 to {request_count}.\n"
        "Format:\n"
        "1. domain seed\n"
        "2. domain seed\n"
        "...\n"
        f"{request_count}. domain seed\n"
        "No JSON and no commentary.\n"
        "Keep keywords concise (2-4 words) and maximize diversity.\n"
        f"Stop after exactly {request_count} items.\n"
        f"Avoid these existing keywords:\n{existing_domains}"
    )


def _default_qa_system_prompt(section_name: str, section_description: str) -> str:
    return (
        "You are a pragmatic QA data generator.\n"
        "Generate exactly one item for the target section below.\n\n"
        f"Target section: {section_name}\n"
        f"Section description: {section_description}\n\n"
        "Return plain text only with this exact format:\n"
        "Question must be impossible to answer without pragmatic interpretation."
        " Answer must be short and clear.\n"
        "Content: ...\n"
        "Question: ...\n"
        "Answer: ...\n"
        "No JSON and no extra commentary."
    )


def _default_qa_user_prompt(
    domain: str,
    item_index: int,
    items_per_domain: int,
    few_shot_examples_jsonl: str,
) -> str:
    examples_block = ""
    if few_shot_examples_jsonl:
        examples_block = f"\nSection examples (style reference):\n{few_shot_examples_jsonl}\n"

    return (
        f"Domain keyword: {domain}\n"
        f"Item: {item_index}/{items_per_domain}\n\n"
        "Field definitions:\n"
        "- content: a concrete scenario/context only, with enough detail for pragmatic inference.\n"
        "- question: one explicit question about the scenario.\n"
        "- answer: a minimal, short, concise, single correct answer to the question without any explanation.\n"
        f"{examples_block}"
        "Output plain text only with this exact format:\n"
        "Content: ...\n"
        "Question: ...\n"
        "Answer: ...\n"
        "No JSON, no markdown, no extra commentary."
    )


def _set_all_seeds(seed: int, *, touch_cuda: bool = True) -> None:
    set_seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if touch_cuda and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _adapt_system_prompt(prompt: str, generation_ctx: dict[str, Any]) -> str | list[dict[str, Any]]:
    return adapt_system_prompt(
        prompt,
        style=str(generation_ctx.get("system_prompt_style", "plain")),
        repo_id=str(generation_ctx.get("system_prompt_repo_id", "")),
        filename=str(generation_ctx.get("system_prompt_filename", "")),
    )


def _prepare_model_context(
    cfg_raw: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[Any, dict[str, Any]]:
    dataset_cfg = cfg_raw.get("dataset_gen", {}) if isinstance(cfg_raw, dict) else {}
    llm_cfg = dataset_cfg.get("use_llm", {}) if isinstance(dataset_cfg, dict) else {}
    models_cfg = cfg_raw.get("models", {}) if isinstance(cfg_raw, dict) else {}

    role = str(args.model_role or llm_cfg.get("model_role", "speaker")).strip() or "speaker"
    model_cfg = models_cfg.get(role) or models_cfg.get("speaker")
    if model_cfg is None:
        raise ValueError(f"Model config not found for role={role} and fallback=speaker")

    if args.model_name_or_path:
        model_cfg = {**model_cfg, "name": args.model_name_or_path}
    if args.dtype:
        model_cfg = {**model_cfg, "dtype": args.dtype}
    if args.device_map:
        model_cfg = {**model_cfg, "device_map": args.device_map}
    if args.load_in_4bit:
        model_cfg = {**model_cfg, "load_in_4bit": True}
    if args.load_in_8bit:
        model_cfg = {**model_cfg, "load_in_8bit": True}

    if args.gen_backend == "vllm_api":
        from src.utils.vllm_api import VLLMApiModel

        vllm_model_name = (
            str(args.vllm_model_name).strip()
            or str(args.model_name_or_path).strip()
            or str(model_cfg.get("name", "")).strip()
        )
        if not vllm_model_name:
            raise ValueError("vLLM backend requires model name (--vllm_model_name or --model_name_or_path)")
        model: Any = VLLMApiModel(
            vllm_model_name,
            base_url=args.vllm_base_url,
            api_key=args.vllm_api_key,
            timeout=float(args.vllm_timeout),
            max_retries=int(args.vllm_max_retries),
        )
    elif args.gen_backend == "vllm_local":
        from src.utils.vllm_local import VLLMLocalModel

        vllm_model_name = (
            str(args.vllm_model_name).strip()
            or str(args.model_name_or_path).strip()
            or str(model_cfg.get("name", "")).strip()
        )
        if not vllm_model_name:
            raise ValueError("vLLM backend requires model name (--vllm_model_name or --model_name_or_path)")
        model = VLLMLocalModel(
            vllm_model_name,
            dtype=str(model_cfg.get("dtype", "")).strip() or None,
            gpu_memory_utilization=float(args.vllm_gpu_memory_utilization),
            tensor_parallel_size=int(args.vllm_tensor_parallel_size),
            max_model_len=int(args.vllm_max_model_len) if int(args.vllm_max_model_len) > 0 else None,
            seed=int(args.seed) if args.seed is not None else int(cfg_raw.get("seed", 42)),
        )
    elif args.gen_backend == "vllm_local_batched":
        vllm_model_name = (
            str(args.vllm_model_name).strip()
            or str(args.model_name_or_path).strip()
            or str(model_cfg.get("name", "")).strip()
        )
        if not vllm_model_name:
            raise ValueError("vLLM backend requires model name (--vllm_model_name or --model_name_or_path)")
        model = VLLMLocalBatchedGenerator(
            model_name=vllm_model_name,
            tokenizer_name=vllm_model_name,
            dtype=str(model_cfg.get("dtype", "")).strip() or None,
            gpu_memory_utilization=float(args.vllm_gpu_memory_utilization),
            tensor_parallel_size=int(args.vllm_tensor_parallel_size),
            max_model_len=int(args.vllm_max_model_len) if int(args.vllm_max_model_len) > 0 else None,
            seed=int(args.seed) if args.seed is not None else int(cfg_raw.get("seed", 42)),
        )
    else:
        model = _build_model(model_cfg, role)

    base_temperature = float(llm_cfg.get("temperature", 0.7))
    base_top_p = llm_cfg.get("top_p", 0.9)
    base_top_k = llm_cfg.get("top_k", 50)
    base_max_new_tokens = int(llm_cfg.get("max_new_tokens", 2048))
    base_enable_thinking = _coerce_bool(llm_cfg.get("enable_thinking", False), default=False)
    enable_thinking = base_enable_thinking if args.enable_thinking is None else bool(args.enable_thinking)
    domain_enable_thinking = (
        enable_thinking if args.domain_enable_thinking is None else bool(args.domain_enable_thinking)
    )
    qa_enable_thinking = enable_thinking if args.qa_enable_thinking is None else bool(args.qa_enable_thinking)

    return model, {
        "domain_temperature": base_temperature if args.domain_temperature is None else float(args.domain_temperature),
        "domain_top_p": base_top_p if args.domain_top_p is None else args.domain_top_p,
        "domain_top_k": base_top_k if args.domain_top_k is None else int(args.domain_top_k),
        "domain_max_new_tokens": int(args.domain_max_new_tokens or base_max_new_tokens),
        "qa_temperature": base_temperature if args.qa_temperature is None else float(args.qa_temperature),
        "qa_top_p": base_top_p if args.qa_top_p is None else args.qa_top_p,
        "qa_top_k": base_top_k if args.qa_top_k is None else int(args.qa_top_k),
        "qa_max_new_tokens": int(args.qa_max_new_tokens or base_max_new_tokens),
        "domain_thinking_budget_tokens": int(args.domain_thinking_budget_tokens or 0),
        "domain_answer_budget_tokens": int(args.domain_answer_budget_tokens or 0),
        "qa_thinking_budget_tokens": int(args.qa_thinking_budget_tokens or 0),
        "qa_answer_budget_tokens": int(args.qa_answer_budget_tokens or 0),
        "thinking_early_stopping_text": normalize_early_stopping_text(
            args.thinking_early_stopping_text,
            args.model_name_or_path,
        ),
        "enable_thinking": enable_thinking,
        "domain_enable_thinking": domain_enable_thinking,
        "qa_enable_thinking": qa_enable_thinking,
        "system_prompt_style": str(args.system_prompt_style or "plain"),
        "system_prompt_repo_id": str(args.system_prompt_repo_id or "").strip(),
        "system_prompt_filename": str(args.system_prompt_filename or "").strip(),
    }


def _generate_domain_keywords(
    *,
    model: HFPolicyModel,
    generation_ctx: dict[str, Any],
    domain_count: int,
    attempts: int,
) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()

    pbar = tqdm(total=domain_count, desc="domains", unit="domain") if tqdm is not None else None
    try:
        for attempt_idx in range(1, attempts + 1):
            if len(selected) >= domain_count:
                break
            attempt_started = time.time()
            remaining = domain_count - len(selected)
            request_count = remaining
            existing_domains = "\n".join(f"- {x}" for x in selected[-200:])

            system_prompt = _adapt_system_prompt(_default_domain_system_prompt(), generation_ctx)
            user_prompt = _default_domain_user_prompt(request_count, existing_domains)
            max_new_tokens = max(
                int(generation_ctx["domain_max_new_tokens"]),
                1000,
                request_count * 12,
            )

            gen = model.generate(
                user_prompt,
                system_prompt=system_prompt,
                max_new_tokens=max_new_tokens,
                temperature=float(generation_ctx["domain_temperature"]),
                top_p=generation_ctx.get("domain_top_p", None),
                top_k=generation_ctx.get("domain_top_k", None),
                enable_thinking=bool(generation_ctx["domain_enable_thinking"]),
                thinking_budget_tokens=int(generation_ctx.get("domain_thinking_budget_tokens", 0)),
                answer_budget_tokens=int(generation_ctx.get("domain_answer_budget_tokens", 0)),
                early_stopping_text=str(generation_ctx.get("thinking_early_stopping_text", "")),
            )
            raw = gen.content or gen.text or ""
            parsed = _parse_domain_keywords(raw)

            added = 0
            for dom in parsed:
                key = _normalize_text(dom)
                if not key or key in seen:
                    continue
                seen.add(key)
                selected.append(dom)
                added += 1
                if len(selected) >= domain_count:
                    break

            if pbar is not None and added > 0:
                pbar.update(added)
            cleaned_raw = _strip_thinking(raw).strip()
            preview = re.sub(r"\s+", " ", cleaned_raw[:200])
            elapsed = time.time() - attempt_started
            log_msg = (
                f"[domain] attempt={attempt_idx} requested={request_count} parsed={len(parsed)} "
                f"added={added} total={len(selected)}/{domain_count} empty={int(not bool(cleaned_raw))} "
                f"max_new_tokens={max_new_tokens} elapsed={elapsed:.2f}s"
            )
            if pbar is not None:
                pbar.write(log_msg)
            else:
                print(log_msg, flush=True)
            if not parsed and cleaned_raw:
                debug_msg = f"[domain-debug] preview={preview!r}"
                if pbar is not None:
                    pbar.write(debug_msg)
                else:
                    print(debug_msg, flush=True)
    finally:
        if pbar is not None:
            pbar.close()

    if len(selected) < domain_count:
        raise RuntimeError(f"Failed to generate enough domain keywords: {len(selected)}/{domain_count}")
    return selected[:domain_count]


def _generate_single_qa_item(
    *,
    model: HFPolicyModel,
    generation_ctx: dict[str, Any],
    domain: str,
    item_index: int,
    items_per_domain: int,
    target_section: dict[str, str],
    few_shot_examples_jsonl: str,
    max_attempts: int,
) -> dict[str, str]:
    section_name = str(target_section.get("name") or "Pragmatic QA").strip()
    section_description = str(target_section.get("description") or "").strip()

    for attempt in range(1, max_attempts + 1):
        system_prompt = _adapt_system_prompt(
            _default_qa_system_prompt(section_name, section_description),
            generation_ctx,
        )
        user_prompt = _default_qa_user_prompt(domain, item_index, items_per_domain, few_shot_examples_jsonl)

        gen = model.generate(
            user_prompt,
            system_prompt=system_prompt,
            max_new_tokens=int(generation_ctx["qa_max_new_tokens"]),
            temperature=float(generation_ctx["qa_temperature"]),
            top_p=generation_ctx.get("qa_top_p", None),
            top_k=generation_ctx.get("qa_top_k", None),
            enable_thinking=bool(generation_ctx["qa_enable_thinking"]),
            thinking_budget_tokens=int(generation_ctx.get("qa_thinking_budget_tokens", 0)),
            answer_budget_tokens=int(generation_ctx.get("qa_answer_budget_tokens", 0)),
            early_stopping_text=str(generation_ctx.get("thinking_early_stopping_text", "")),
        )
        raw = gen.content or gen.text or ""

        parsed = _parse_qa_candidates(raw)
        if not parsed:
            print(
                f"[qa] section={section_name!r} domain={domain!r} idx={item_index} attempt={attempt} parse=0"
            )
            continue

        for cand in parsed:
            valid = _validate_qa_candidate(cand)
            if valid is None:
                continue
            return {
                "section": section_name,
                "content": valid["content"],
                "question": valid["question"],
                "answer": valid["answer"],
            }

        print(
            f"[qa] section={section_name!r} domain={domain!r} idx={item_index} attempt={attempt} valid=0"
        )

    raise RuntimeError(
        f"Failed to generate valid QA item for section={section_name!r}, domain={domain!r}, item={item_index}"
    )


def _build_section_schedule(total_items: int, num_sections: int, seed: int) -> list[int]:
    if num_sections <= 0:
        return [0] * max(0, total_items)

    local_rng = random.Random(seed + 20260226)
    schedule: list[int] = []
    while len(schedule) < total_items:
        block = list(range(num_sections))
        local_rng.shuffle(block)
        schedule.extend(block)
    return schedule[:total_items]


def _has_domains_input_path(path: Path | None) -> bool:
    if path is None:
        return False
    raw = str(path).strip()
    if raw in {"", "."}:
        return False
    return True


def _build_qa_messages(
    *,
    generation_ctx: dict[str, Any],
    domain: str,
    item_index: int,
    items_per_domain: int,
    target_section: dict[str, str],
    few_shot_examples_jsonl: str,
) -> list[dict[str, Any]]:
    section_name = str(target_section.get("name") or "Pragmatic QA").strip()
    section_description = str(target_section.get("description") or "").strip()
    return [
        {
            "role": "system",
            "content": _adapt_system_prompt(
                _default_qa_system_prompt(section_name, section_description),
                generation_ctx,
            ),
        },
        {
            "role": "user",
            "content": _default_qa_user_prompt(domain, item_index, items_per_domain, few_shot_examples_jsonl),
        },
    ]


def _generate_qa_records_batched(
    *,
    generator: VLLMLocalBatchedGenerator,
    generation_ctx: dict[str, Any],
    domains: list[str],
    sections: list[dict[str, str]],
    few_shot_by_section: dict[str, list[dict[str, Any]]],
    args: argparse.Namespace,
    seed: int,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for domain_idx, domain in enumerate(domains, start=1):
        for section_idx, target_section in enumerate(sections, start=1):
            section_examples = list(
                few_shot_by_section.get(_section_norm(target_section.get("name", "")), [])
            )
            for local_idx in range(1, args.items_per_domain + 1):
                prompt_examples = list(section_examples)
                shuffle_rng = random.Random(
                    seed + (domain_idx * 1_000_003) + (section_idx * 1009) + (local_idx * 17)
                )
                if args.qa_few_shot_subset_k > 0 and len(prompt_examples) > args.qa_few_shot_subset_k:
                    prompt_examples = shuffle_rng.sample(prompt_examples, args.qa_few_shot_subset_k)
                if (args.qa_few_shot_shuffle or args.qa_few_shot_subset_k > 0) and len(prompt_examples) > 1:
                    shuffle_rng.shuffle(prompt_examples)
                specs.append(
                    {
                        "domain_idx": domain_idx,
                        "domain": domain,
                        "section_idx": section_idx,
                        "target_section": target_section,
                        "local_idx": local_idx,
                        "few_shot_examples_jsonl": _format_few_shot_jsonl(prompt_examples),
                    }
                )

    total_target = len(specs)
    records_by_idx: list[dict[str, Any] | None] = [None] * total_target
    seen_signatures: set[str] = set()
    collision_retry_used = [False] * total_target
    pending = list(range(total_target))
    completed = 0

    pbar = tqdm(total=total_target, desc="qa_generate", unit="item") if tqdm is not None else None
    try:
        for attempt in range(1, args.qa_attempts_per_item + 1):
            if not pending:
                break
            print(f"[qa-batch] attempt={attempt} pending={len(pending)}")
            message_batches = [
                _build_qa_messages(
                    generation_ctx=generation_ctx,
                    domain=specs[idx]["domain"],
                    item_index=int(specs[idx]["local_idx"]),
                    items_per_domain=args.items_per_domain,
                    target_section=specs[idx]["target_section"],
                    few_shot_examples_jsonl=specs[idx]["few_shot_examples_jsonl"],
                )
                for idx in pending
            ]
            raws = generator.generate_batch(
                message_batches,
                max_new_tokens=int(generation_ctx["qa_max_new_tokens"]),
                temperature=float(generation_ctx["qa_temperature"]),
                top_p=generation_ctx.get("qa_top_p", None),
                top_k=generation_ctx.get("qa_top_k", None),
                enable_thinking=bool(generation_ctx["qa_enable_thinking"]),
                thinking_budget_tokens=int(generation_ctx.get("qa_thinking_budget_tokens", 0)),
                answer_budget_tokens=int(generation_ctx.get("qa_answer_budget_tokens", 0)),
                early_stopping_text=str(generation_ctx.get("thinking_early_stopping_text", "")),
                use_tqdm=True,
            )

            next_pending: list[int] = []
            for pos, spec_idx in enumerate(pending):
                spec = specs[spec_idx]
                raw = raws[pos]
                section_name = str(spec["target_section"].get("name") or "Pragmatic QA").strip()
                parsed = _parse_qa_candidates(raw)
                if not parsed:
                    print(
                        f"[qa] section={section_name!r} domain={spec['domain']!r} idx={spec['local_idx']} attempt={attempt} parse=0"
                    )
                    next_pending.append(spec_idx)
                    continue

                item: dict[str, str] | None = None
                for cand in parsed:
                    valid = _validate_qa_candidate(cand)
                    if valid is None:
                        continue
                    item = {
                        "section": section_name,
                        "content": valid["content"],
                        "question": valid["question"],
                        "answer": valid["answer"],
                    }
                    break

                if item is None:
                    print(
                        f"[qa] section={section_name!r} domain={spec['domain']!r} idx={spec['local_idx']} attempt={attempt} valid=0"
                    )
                    next_pending.append(spec_idx)
                    continue

                signature = _normalize_text(
                    " || ".join([item["section"], item["content"], item["question"], item["answer"]])
                )
                if (
                    signature in seen_signatures
                    and not collision_retry_used[spec_idx]
                    and attempt < args.qa_attempts_per_item
                ):
                    collision_retry_used[spec_idx] = True
                    next_pending.append(spec_idx)
                    continue

                seen_signatures.add(signature)
                record: dict[str, Any] = {
                    "section": item["section"],
                    "content": item["content"],
                    "question": item["question"],
                    "answer": item["answer"],
                    "tuple": _to_cqa_tuple(item["content"], item["question"], item["answer"]),
                }
                if args.include_domain_field:
                    record["domain_seed"] = spec["domain"]
                    record["domain_index"] = spec["domain_idx"]
                records_by_idx[spec_idx] = record
                completed += 1
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix_str(f"section={item['section']}", refresh=False)
                print(
                    f"[qa] progress={completed}/{total_target} domain={spec['domain_idx']}/{len(domains)} "
                    f"section={spec['section_idx']}/{len(sections)} item={spec['local_idx']}/{args.items_per_domain} "
                    f"name={item['section']}"
                )
            pending = next_pending
    finally:
        if pbar is not None:
            pbar.close()

    if pending:
        first = specs[pending[0]]
        section_name = str(first["target_section"].get("name") or "Pragmatic QA").strip()
        raise RuntimeError(
            f"Failed to generate valid QA item for section={section_name!r}, "
            f"domain={first['domain']!r}, item={first['local_idx']} (remaining={len(pending)})"
        )

    return [record for record in records_by_idx if record is not None]


def main() -> None:
    args = parse_args()

    if args.domain_count < 1:
        raise ValueError("domain_count must be >= 1")
    if args.items_per_domain < 1:
        raise ValueError("items_per_domain must be >= 1")

    cfg_raw = load_config(str(args.config_path))
    seed = int(cfg_raw.get("seed", 42)) if args.seed is None else int(args.seed)
    _set_all_seeds(seed, touch_cuda=(args.gen_backend not in {
        "vllm_local",
        "vllm_local_batched",
    }))

    if args.output_path.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {args.output_path}. Use --overwrite to replace it.")

    sections = _load_sections(cfg_raw)
    few_shot_by_section = _load_few_shot_by_section(args.qa_examples_path, args.qa_few_shot_max)
    if few_shot_by_section:
        total_examples = sum(len(v) for v in few_shot_by_section.values())
        print(
            f"[few-shot] loaded sections={len(few_shot_by_section)} total={total_examples} "
            f"per_section_max={args.qa_few_shot_max} subset_k={args.qa_few_shot_subset_k} "
            f"path={args.qa_examples_path} shuffle={args.qa_few_shot_shuffle}"
        )
    else:
        print(f"[few-shot] not loaded (path missing/empty): {args.qa_examples_path}")

    model, generation_ctx = _prepare_model_context(cfg_raw, args)
    if hasattr(model, "close"):
        import atexit

        atexit.register(model.close)

    print(f"[backend] {args.gen_backend}")

    print(
        f"[thinking_budget] domain={generation_ctx.get('domain_thinking_budget_tokens', 0)}/{generation_ctx.get('domain_answer_budget_tokens', 0)} "
        f"qa={generation_ctx.get('qa_thinking_budget_tokens', 0)}/{generation_ctx.get('qa_answer_budget_tokens', 0)}"
    )

    if _has_domains_input_path(args.domains_input_path):
        domains = _load_domain_file(Path(args.domains_input_path))
        uniq_domains: list[str] = []
        seen_domains: set[str] = set()
        for domain in domains:
            key = _normalize_text(domain)
            if not key or key in seen_domains:
                continue
            seen_domains.add(key)
            uniq_domains.append(domain)
        domains = uniq_domains[: args.domain_count]
        if len(domains) < args.domain_count:
            raise RuntimeError(
                f"domains_input_path provided only {len(domains)} unique domains; domain_count={args.domain_count}"
            )
        print(f"[domain] loaded from file: {len(domains)}")
    else:
        domains = _generate_domain_keywords(
            model=model,
            generation_ctx=generation_ctx,
            domain_count=args.domain_count,
            attempts=args.domain_generation_attempts,
        )

    args.domains_output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.domains_output_path.open("w", encoding="utf-8") as f:
        for idx, domain in enumerate(domains, start=1):
            f.write(json.dumps({"id": idx, "domain": domain}, ensure_ascii=False) + "\n")
    print(f"[domain] saved seeds: {args.domains_output_path} ({len(domains)})")

    if args.gen_backend == "vllm_local_batched":
        records = _generate_qa_records_batched(
            generator=model,
            generation_ctx=generation_ctx,
            domains=domains,
            sections=sections,
            few_shot_by_section=few_shot_by_section,
            args=args,
            seed=seed,
        )
    else:
        total_target = len(domains) * len(sections) * args.items_per_domain

        records: list[dict[str, Any]] = []
        seen_signatures: set[str] = set()

        qa_pbar = tqdm(total=total_target, desc="qa_generate", unit="item") if tqdm is not None else None
        domain_iter: Any = enumerate(domains, start=1)
        if tqdm is not None:
            domain_iter = enumerate(tqdm(domains, desc="domain_sessions", unit="domain"), start=1)

        global_idx = 0
        try:
            for domain_idx, domain in domain_iter:
                for section_idx, target_section in enumerate(sections, start=1):
                    section_examples = list(
                        few_shot_by_section.get(_section_norm(target_section.get("name", "")), [])
                    )
                    for local_idx in range(1, args.items_per_domain + 1):
                        prompt_examples = list(section_examples)
                        shuffle_rng = random.Random(
                            seed + (domain_idx * 1_000_003) + (section_idx * 1009) + (local_idx * 17)
                        )

                        if args.qa_few_shot_subset_k > 0 and len(prompt_examples) > args.qa_few_shot_subset_k:
                            prompt_examples = shuffle_rng.sample(prompt_examples, args.qa_few_shot_subset_k)

                        if (args.qa_few_shot_shuffle or args.qa_few_shot_subset_k > 0) and len(prompt_examples) > 1:
                            shuffle_rng.shuffle(prompt_examples)

                        section_examples_jsonl = _format_few_shot_jsonl(prompt_examples)
                        item = _generate_single_qa_item(
                            model=model,
                            generation_ctx=generation_ctx,
                            domain=domain,
                            item_index=local_idx,
                            items_per_domain=args.items_per_domain,
                            target_section=target_section,
                            few_shot_examples_jsonl=section_examples_jsonl,
                            max_attempts=args.qa_attempts_per_item,
                        )

                        signature = _normalize_text(
                            " || ".join([item["section"], item["content"], item["question"], item["answer"]])
                        )
                        if signature in seen_signatures:
                            # one extra regenerate attempt for collision
                            item = _generate_single_qa_item(
                                model=model,
                                generation_ctx=generation_ctx,
                                domain=domain,
                                item_index=local_idx,
                                items_per_domain=args.items_per_domain,
                                target_section=target_section,
                                few_shot_examples_jsonl=section_examples_jsonl,
                                max_attempts=args.qa_attempts_per_item,
                            )
                            signature = _normalize_text(
                                " || ".join([item["section"], item["content"], item["question"], item["answer"]])
                            )

                        seen_signatures.add(signature)

                        record = {
                            "section": item["section"],
                            "content": item["content"],
                            "question": item["question"],
                            "answer": item["answer"],
                            "tuple": _to_cqa_tuple(item["content"], item["question"], item["answer"]),
                        }
                        if args.include_domain_field:
                            record["domain_seed"] = domain
                            record["domain_index"] = domain_idx

                        records.append(record)
                        global_idx += 1

                        if qa_pbar is not None:
                            qa_pbar.update(1)
                            qa_pbar.set_postfix_str(f"section={item['section']}", refresh=False)

                        print(
                            f"[qa] progress={global_idx}/{total_target} domain={domain_idx}/{len(domains)} "
                            f"section={section_idx}/{len(sections)} item={local_idx}/{args.items_per_domain} "
                            f"name={item['section']}"
                        )
        finally:
            if qa_pbar is not None:
                qa_pbar.close()

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[done] wrote {len(records)} items to {args.output_path}")


if __name__ == "__main__":
    main()
