#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.thinking_tags import (
    ALL_CLOSE_TAGS,
    contains_thinking_close,
    normalize_early_stopping_text,
    split_after_last_thinking_close,
    strip_prefixed_thinking,
)
from src.utils.chat_template_family import adapt_messages_for_chat_template


TEACHER_SYSTEM_PROMPT_QA = (
    "You are solving a pragmatic question-answering task.\n\n"
    "Your goal is to choose the best answer by inferring the speaker's intended meaning in context.\n\n"
    "Do not rely on the literal meaning of the utterance alone. Instead, interpret the utterance as a communicative "
    "choice made by a roughly rational and informative speaker.\n"
    "Think about why this speaker chose this utterance, in this context, given what the speaker likely knows.\n\n"
    "When answering a question, use the following reasoning principles:\n\n"
    "1. Identify the literal meaning of the utterance.\n"
    "2. Use the context and shared background to determine what the speaker is likely trying to communicate.\n"
    "3. Consider why the speaker chose this utterance instead of other plausible alternatives.\n"
    "4. Assume the speaker is trying to provide relevant information in context, but may not say more than is needed.\n"
    "5. Use the speaker's likely knowledge and the shared context to infer what the listener is expected to understand.\n"
    "6. Choose the answer that best explains the utterance as a rational, informative, and contextually relevant choice.\n\n"
    
    "When possible, justify your interpretation contrastively until you reach to the one clear interpretation:\n"
    "- state one more direct, stronger, or more literal alternative the speaker could have said,\n"
    "- explain what that alternative would have implied,\n"
    "- then explain why the actual utterance suggests a different intended meaning.\n"
    "Reason in the form:\n"
    "\"If the speaker had intended X, they would likely have said Y. Because they said Z instead, the intended meaning is more likely W.\"\n\n"
    
    "Guidelines:\n"
    "- Prefer the answer that matches the speaker's intended meaning, not just the surface wording.\n"
    "- Use context, shared background, speaker knowledge, and plausible alternatives to interpret the utterance.\n"
    "- Prefer interpretations that best explain the speaker's choice of wording.\n"
    "- Do not infer more than the context, shared background, and the speaker's choice support."
)

TEACHER_SYSTEM_PROMPT_QA_LIGHT = (
    "You are solving a pragmatic question-answering task.\n\n"
    "Answer the question by considering the pragmatic meaning of the utterance in its context.\n"
    "Pay attention to what the speaker likely intends to communicate, not only the literal wording.\n\n"
    "Output only the final answer text."
)

SFT_SYSTEM_PROMPT_QA = (
    "You answer the given question from the provided context. "
)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def is_ministral3_model_name(model_name_or_path: str) -> bool:
    raw = str(model_name_or_path or "").strip().lower()
    return "ministral-3" in raw or "ministral3" in raw


def _load_section_description_map(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, str]:
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}

    dataset_gen = raw.get("dataset_gen", {}) if isinstance(raw, dict) else {}
    sections = dataset_gen.get("sections", []) if isinstance(dataset_gen, dict) else []
    out: dict[str, str] = {}
    if not isinstance(sections, list):
        return out
    for item in sections:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        desc = str(item.get("description", "")).strip()
        if name:
            out[name] = desc
    return out


SECTION_DESCRIPTION_MAP = _load_section_description_map()


def _get_section_description(section: str, row: dict[str, Any]) -> str:
    inline = str(row.get("section_description", row.get("description", ""))).strip()
    if inline:
        return inline
    return str(SECTION_DESCRIPTION_MAP.get(section, "")).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build pragmatic SFT dataset with thinking-budget teacher outputs.")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument(
        "--qa_data_path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=ROOT / "data/sft/pragmatic_sft_thinking_all.jsonl",
    )
    parser.add_argument(
        "--train_output_path",
        type=Path,
        default=ROOT / "data/sft/pragmatic_sft_thinking_train.jsonl",
    )
    parser.add_argument(
        "--val_output_path",
        type=Path,
        default=ROOT / "data/sft/pragmatic_sft_thinking_val.jsonl",
    )
    parser.add_argument("--val_ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--max_qa_samples", type=int, default=0)
    parser.add_argument("--qa_mix_weight", type=int, default=1)
    parser.add_argument(
        "--teacher_prompt_mode",
        type=str,
        default="pragmatic",
        choices=["pragmatic", "pragmatic_light", "neutral"],
        help=(
            "Teacher prompt style for generation. 'pragmatic_light' mentions pragmatic "
            "meaning without contrastive/counterfactual reasoning; 'neutral' uses the "
            "simple non-counterfactual QA prompt."
        ),
    )
    parser.add_argument("--thinking_budget_tokens", type=int, default=8192)
    parser.add_argument("--answer_budget_tokens", type=int, default=512)
    parser.add_argument(
        "--early_stopping_text",
        type=str,
        default=(
            "\n\nConsidering the limited time by the user, I have to give the solution "
            "based on the thinking directly now.\n[/think]\n\n"
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--max_input_length", type=int, default=4096)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--load_in_4bit", action="store_true")

    parser.add_argument(
        "--gen_backend",
        type=str,
        default="hf",
        choices=["hf", "vllm_api", "vllm_local_batched"],
        help="Teacher generation backend.",
    )
    parser.add_argument(
        "--vllm_base_url",
        type=str,
        default="http://localhost:8000",
        help="vLLM OpenAI-compatible server URL. Accepts base URL or /v1 URL.",
    )
    parser.add_argument(
        "--vllm_model_name",
        type=str,
        default="",
        help="Model name served by vLLM. Defaults to --model_name_or_path.",
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

    parser.add_argument(
        "--keep_pragmatic_instruction_in_sft",
        action="store_true",
        help="If set, keep pragmatic directives in saved SFT messages. Default removes them.",
    )
    parser.add_argument(
        "--judge_refine_rounds",
        type=int,
        default=0,
        help="Number of judge-guided refinement rounds. 0 disables judge refinement.",
    )
    parser.add_argument("--judge_max_new_tokens", type=int, default=2048)
    parser.add_argument(
        "--judge_enable_thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable thinking for generated confidence judging. The final judge output is parsed "
            "after the closing thinking tag."
        ),
    )
    parser.add_argument("--judge_temperature", type=float, default=0.0)
    parser.add_argument("--judge_top_p", type=float, default=1.0)
    parser.add_argument("--judge_top_k", type=int, default=50)
    parser.add_argument(
        "--judge_method",
        type=str,
        default="margin",
        choices=["margin", "confidence"],
        help="Judge filter method. 'margin' uses P(yes)-P(no); 'confidence' parses generated JSON.",
    )
    parser.add_argument(
        "--judge_confidence_threshold",
        type=int,
        default=8,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--judge_margin_threshold",
        type=float,
        default=0.8,
        help="Minimum P(yes)-P(no) required by --judge_method margin.",
    )
    parser.add_argument(
        "--judge_score_batch_size",
        type=int,
        default=1024,
        help="Maximum expanded continuation prompts per vLLM prompt-logprob scoring call.",
    )
    parser.add_argument(
        "--judge_drop_failed",
        action="store_true",
        help="Drop samples that still fail judge after all refinement rounds.",
    )
    parser.add_argument(
        "--judge_drop_without_refine",
        action="store_true",
        help="Run one judge pass and drop failed samples without refinement re-generation.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


@dataclass
class Task:
    task_id: str
    task_type: str
    source_dataset: str
    teacher_messages: list[dict[str, str]]
    sft_messages: list[dict[str, str]]
    metadata: dict[str, Any]


@dataclass
class JudgeResult:
    passed: bool
    feedback: str
    corrected_answer: str
    raw_output: str
    judgment: str | None = None
    confidence: int | None = None
    yes_logprob: float | None = None
    no_logprob: float | None = None
    yes_prob: float | None = None
    no_prob: float | None = None
    logprob_margin: float | None = None


def load_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _normalize_chat_message_content(content: Any) -> Any:
    if isinstance(content, str):
        return content
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
                        parts.append(thinking)
                else:
                    text = str(item.get("text", item)).strip()
                    if text:
                        parts.append(text)
            else:
                text = str(item).strip()
                if text:
                    parts.append(text)
        return "\n\n".join(parts).strip()
    if content is None:
        return ""
    return str(content)


def _normalize_chat_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for message in messages:
        item = dict(message)
        if "content" in item:
            item["content"] = _normalize_chat_message_content(item.get("content"))
        normalized.append(item)
    return normalized


def _summarize_chat_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for idx, message in enumerate(messages):
        content = message.get("content")
        item: dict[str, Any] = {
            "index": idx,
            "role": str(message.get("role", "")),
            "content_type": type(content).__name__,
        }
        if isinstance(content, str):
            item["content_preview"] = content[:200]
            item["content_length"] = len(content)
        elif isinstance(content, list):
            item["list_length"] = len(content)
            item["list_item_types"] = [type(x).__name__ for x in content[:5]]
            item["content_preview"] = str(content[:2])[:200]
        else:
            item["content_preview"] = repr(content)[:200]
        summary.append(item)
    return summary


def apply_chat_template_text(
    tokenizer,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
    enable_thinking: bool,
    model_name_or_path: str | None = None,
) -> str:
    messages = _normalize_chat_messages(messages)
    messages, template_kwargs = adapt_messages_for_chat_template(
        tokenizer,
        str(model_name_or_path or getattr(tokenizer, "name_or_path", "")),
        messages,
    )
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
        **template_kwargs,
    }
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)
    except TypeError:
        try:
            return tokenizer.apply_chat_template(messages, **kwargs)
        except Exception as exc:
            raise RuntimeError(
                "apply_chat_template failed after normalization; "
                f"model={model_name_or_path or getattr(tokenizer, 'name_or_path', '')} "
                f"messages={json.dumps(_summarize_chat_messages(messages), ensure_ascii=False)}"
            ) from exc
    except Exception as exc:
        raise RuntimeError(
            "apply_chat_template failed after normalization; "
            f"model={model_name_or_path or getattr(tokenizer, 'name_or_path', '')} "
            f"messages={json.dumps(_summarize_chat_messages(messages), ensure_ascii=False)}"
        ) from exc


def contains_subsequence(sequence: list[int], pattern: list[int]) -> bool:
    if not pattern or len(pattern) > len(sequence):
        return False
    for i in range(len(sequence) - len(pattern) + 1):
        if sequence[i : i + len(pattern)] == pattern:
            return True
    return False


def maybe_special_id(tokenizer, token: str) -> int | None:
    try:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0:
            return token_id
    except Exception:
        return None
    return None


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


def truncate_prompt_text(tokenizer, prompt_text: str, max_input_length: int) -> str:
    encoded = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_input_length,
    )["input_ids"]
    return tokenizer.decode(encoded, skip_special_tokens=False)


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


def build_vllm_completions_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        return normalized + "/completions"
    return normalized + "/v1/completions"


def call_vllm_completion(
    *,
    base_url: str,
    api_key: str,
    model_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int | None,
    timeout: float,
    max_retries: int,
) -> tuple[str, str]:
    url = build_vllm_completions_url(base_url)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    do_sample = temperature > 0.0
    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature if do_sample else 0.0,
        "top_p": top_p if do_sample else 1.0,
        "stream": False,
        "n": 1,
    }
    if do_sample and top_k is not None and int(top_k) > 0:
        payload["top_k"] = int(top_k)

    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]
            return str(choice.get("text", "")), str(choice.get("finish_reason", ""))
        except Exception as exc:
            last_err = exc
            if attempt == max_retries:
                break
            time.sleep(min(2**attempt, 4))

    assert last_err is not None
    raise last_err




def strip_thinking(raw: str) -> str:
    return split_after_last_thinking_close(raw)


def trim_thinking_prefix_through_close(
    raw: str,
    *,
    early_stopping_text: str,
    model_name_or_path: str | None = None,
) -> str:
    text = str(raw or "")
    lowered = text.lower()
    best = -1
    best_tag = ""
    for tag in ALL_CLOSE_TAGS:
        idx = lowered.rfind(tag.lower())
        if idx > best:
            best = idx
            best_tag = tag
    if best != -1:
        return text[: best + len(best_tag)].rstrip() + "\n\n"
    stop_text = normalize_early_stopping_text(early_stopping_text, model_name_or_path)
    return text.rstrip() + stop_text


def extract_final_output_text(raw: str) -> str:
    text = strip_thinking(raw)
    m = re.search(r"\\boxed\{(.+)\}", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def has_nonempty_final_output(raw: str) -> bool:
    return bool(extract_final_output_text(raw).strip())


def parse_jsonish(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        pass

    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
        except Exception:
            pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and start < end:
        candidate = cleaned[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass
        try:
            return json.loads(candidate.replace("'", '"'))
        except Exception:
            pass

    raise ValueError("Could not parse judge JSON output")


def _clamp_confidence(value: Any) -> int | None:
    if value is None:
        return None
    try:
        confidence = int(round(float(value)))
    except Exception:
        return None
    return max(1, min(10, confidence))


def parse_confidence_judge_result(raw_text: str, confidence_threshold: int) -> JudgeResult:
    fallback = JudgeResult(
        passed=False,
        feedback="Judge output was not parseable JSON.",
        corrected_answer="",
        raw_output=raw_text,
        judgment=None,
        confidence=None,
    )
    try:
        parsed = parse_jsonish(strip_thinking(raw_text))
    except Exception:
        return fallback

    judgment = str(parsed.get("judgment", parsed.get("verdict", ""))).strip().lower()
    if judgment not in {"yes", "no"}:
        judgment = None
    confidence = _clamp_confidence(parsed.get("confidence", parsed.get("confidence_score")))
    explanation = str(
        parsed.get(
            "explanation",
            parsed.get("feedback", parsed.get("rationale", "")),
        )
    ).strip()
    corrected = str(
        parsed.get(
            "corrected_answer",
            parsed.get("ideal_answer", parsed.get("target_answer", "")),
        )
    ).strip()
    passed = judgment == "yes" and confidence is not None and confidence >= int(confidence_threshold)
    if judgment == "yes" and confidence is not None and confidence < int(confidence_threshold):
        explanation = (
            f"Judge returned yes with confidence {confidence}, below threshold "
            f"{int(confidence_threshold)}. {explanation}"
        ).strip()
    elif judgment is None or confidence is None:
        explanation = explanation or "Judge output lacked a valid yes/no judgment or confidence score."
    return JudgeResult(
        passed=passed,
        feedback=explanation,
        corrected_answer=corrected,
        raw_output=raw_text,
        judgment=judgment,
        confidence=confidence,
    )


def build_logprob_judge_result(
    *,
    yes_logprob: float,
    no_logprob: float,
    margin_threshold: float,
    raw_output: str = "",
    feedback: str = "",
) -> JudgeResult:
    yes_lp = float(yes_logprob)
    no_lp = float(no_logprob)
    p_yes = math.exp(yes_lp) if math.isfinite(yes_lp) else 0.0
    p_no = math.exp(no_lp) if math.isfinite(no_lp) else 0.0

    margin = p_yes - p_no
    passed = math.isfinite(margin) and margin >= float(margin_threshold)
    margin_feedback = (
        f"Judge raw-softmax margin P(yes)-P(no)={margin:.4f}; "
        f"P(yes)={p_yes:.4f}; P(no)={p_no:.4f}; "
        f"logP(yes)={yes_lp:.4f}; logP(no)={no_lp:.4f}; "
        f"threshold={float(margin_threshold):.4f}."
    )
    if feedback:
        feedback = f"{feedback} {margin_feedback}".strip()
    else:
        feedback = margin_feedback
    return JudgeResult(
        passed=passed,
        feedback=feedback,
        corrected_answer="",
        raw_output=raw_output,
        judgment="yes" if passed else "no",
        confidence=None,
        yes_logprob=float(yes_logprob),
        no_logprob=float(no_logprob),
        yes_prob=p_yes,
        no_prob=p_no,
        logprob_margin=margin,
    )


def _continuation_probability_mass(scores: dict[str, float], continuations: list[str]) -> tuple[float, float]:
    total = 0.0
    for continuation in continuations:
        value = float(scores.get(continuation, float("-inf")))
        if math.isfinite(value):
            total += math.exp(value)
    log_total = math.log(total) if total > 0.0 else float("-inf")
    return total, log_total


def parse_judge_result(raw_text: str) -> JudgeResult:
    fallback = JudgeResult(
        passed=False,
        feedback="Judge output was not parseable JSON.",
        corrected_answer="",
        raw_output=raw_text,
    )
    try:
        parsed = parse_jsonish(strip_thinking(raw_text))
    except Exception:
        return fallback

    judgment = str(parsed.get("judgment", parsed.get("verdict", ""))).strip().lower()
    passed = judgment in {"yes", "pass", "correct", "true", "1"}
    feedback = str(parsed.get("feedback", parsed.get("rationale", ""))).strip()
    corrected = str(
        parsed.get(
            "corrected_answer",
            parsed.get("ideal_answer", parsed.get("target_answer", "")),
        )
    ).strip()
    return JudgeResult(
        passed=passed,
        feedback=feedback,
        corrected_answer=corrected,
        raw_output=raw_text,
        judgment=judgment if judgment else None,
        confidence=None,
    )


def _qa_judge_fields(task: Task) -> tuple[str, str, str]:
    row = task.metadata.get("source_row", {})
    if not isinstance(row, dict):
        row = {}
    content = str(row.get("content", task.metadata.get("content", "")) or "").strip()
    question = str(row.get("question", task.metadata.get("question", "")) or "").strip()
    reference = str(
        task.metadata.get(
            "reference_answer",
            row.get("answer", task.metadata.get("reference", "")),
        )
        or ""
    ).strip()
    return content, question, reference


def build_judge_messages(task: Task, candidate_output: str) -> list[dict[str, str]]:
    if task.task_type == "qa":
        content, question, reference = _qa_judge_fields(task)

        user_content = (
            "Task: QA answer grading\n\n"
            f"Context:\n{content}\n\n"
            f"Question:\n{question}\n\n"
            f"Reference answer:\n{reference}\n\n"
            f"Candidate answer:\n{candidate_output}\n\n"
            "Is the candidate answer semantically correct given the context, question, and reference answer? "
            "Is the question itself answerable given the context? If the question is unanswerable, judge the candidate as incorrect."
            "Return JSON only after thinking, with keys: "
            '"feedback" (short reason), and "corrected_answer" (empty if judgment is yes).'
            '"judgment" ("yes" or "no"), "confidence" (integer 1-10), '
        )
        return [
            {
                "role": "system",
                "content": (
                    "You are a strict QA evaluator. "
                    "After any thinking, respond with one JSON object only. "
                    "Do not emit markdown or extra text outside the JSON."
                ),
            },
            {"role": "user", "content": user_content},
        ]

    raise ValueError(f"Unsupported task_type for QA-only SFT builder: {task.task_type}")


def build_margin_judge_messages(task: Task, candidate_output: str) -> list[dict[str, str]]:
    if task.task_type == "qa":
        content, question, reference = _qa_judge_fields(task)
        user_content = (
            "Task: QA answer grading\n\n"
            f"Context:\n{content}\n\n"
            f"Question:\n{question}\n\n"
            f"Reference answer:\n{reference}\n\n"
            f"Candidate answer:\n{candidate_output}\n\n"
            "Is the candidate answer semantically correct given the context, "
            "question, and reference answer? "
            "Answer with a single word: yes or no."
        )
        return [
            {
                "role": "system",
                "content": (
                    "You are a strict QA evaluator. "
                    "Respond with exactly one word: either 'yes' or 'no'. "
                    "Do not emit any other text, punctuation, or explanation."
                ),
            },
            {"role": "user", "content": user_content},
        ]

    return build_judge_messages(task, candidate_output)


def build_refinement_messages(task: Task, previous_output: str, judge_result: JudgeResult) -> list[dict[str, str]]:
    previous_final = extract_final_output_text(previous_output)
    assistant_turn = previous_final if previous_final else "(empty answer)"
    feedback = judge_result.feedback.strip() or "Your previous output did not satisfy the task constraints."
    corrected = judge_result.corrected_answer.strip()

    if task.task_type == "qa":
        row = task.metadata.get("source_row", {})
        if not corrected:
            corrected = str(task.metadata.get("reference_answer", row.get("answer", ""))).strip()

        refine_user = (
            "Your previous answer was judged incorrect.\n\n"
            f"Previous final answer:\n{previous_final}\n\n"
            f"Judge feedback:\n{feedback}\n\n"
            f"Reference-correct answer:\n{corrected}\n\n"
            "Answer the same question again. Output only the final answer text. "
            "\n\nWrite your final response in \\boxed{} on the last line."
        )
    else:
        raise ValueError(f"Unsupported task_type for QA-only SFT builder: {task.task_type}")

    return [
        task.teacher_messages[0],
        task.teacher_messages[1],
        {"role": "assistant", "content": assistant_turn},
        {"role": "user", "content": refine_user},
    ]


def generate_chat_once_hf(
    model,
    tokenizer,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    max_input_length: int,
    top_k: int | None = None,
    enable_thinking: bool = False,
    model_name_or_path: str | None = None,
) -> str:
    prompt_text = apply_chat_template_text(
        tokenizer,
        messages,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        model_name_or_path=model_name_or_path,
    )
    model_inputs = tokenizer(
        [prompt_text],
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
    )
    model_device = next(model.parameters()).device
    model_inputs = {k: v.to(model_device) for k, v in model_inputs.items()}
    prompt_len = model_inputs["input_ids"].shape[-1]

    do_sample = temperature > 0.0
    gen_kwargs = {
        "do_sample": do_sample,
        "temperature": temperature if do_sample else None,
        "top_p": top_p if do_sample else None,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample and top_k is not None and int(top_k) > 0:
        gen_kwargs["top_k"] = int(top_k)
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    with torch.no_grad():
        out_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            **gen_kwargs,
        )

    new_ids = out_ids[0][prompt_len:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def score_continuation_hf(
    model,
    tokenizer,
    messages: list[dict[str, str]],
    continuation: str,
    *,
    max_input_length: int,
    model_name_or_path: str | None = None,
    assistant_prefix: str = "",
    enable_thinking: bool = False,
) -> float:
    prompt_text = apply_chat_template_text(
        tokenizer,
        messages,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        model_name_or_path=model_name_or_path,
    )
    prompt_text = truncate_prompt_text(tokenizer, prompt_text, max_input_length)
    prompt_text = prompt_text + str(assistant_prefix or "")
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt_text + continuation, add_special_tokens=False)["input_ids"]
    prompt_len = len(prompt_ids)
    if prompt_len <= 0 or prompt_len >= len(full_ids):
        return float("-inf")

    model_device = next(model.parameters()).device
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=model_device)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits

    total = 0.0
    for pos in range(prompt_len, len(full_ids)):
        token_id = int(full_ids[pos])
        log_probs = torch.log_softmax(logits[0, pos - 1], dim=-1)
        total += float(log_probs[token_id].item())
    return total


def generate_judge_thinking_prefix_hf(
    model,
    tokenizer,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int,
    early_stopping_text: str,
    temperature: float,
    top_p: float,
    max_input_length: int,
    top_k: int | None = None,
    model_name_or_path: str | None = None,
) -> str:
    raw_prefix = generate_chat_once_hf(
        model,
        tokenizer,
        messages,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_input_length=max_input_length,
        enable_thinking=True,
        model_name_or_path=model_name_or_path,
    )
    return trim_thinking_prefix_through_close(
        raw_prefix,
        early_stopping_text=early_stopping_text,
        model_name_or_path=model_name_or_path,
    )


def generate_chat_once_vllm(
    *,
    tokenizer,
    messages: list[dict[str, str]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    max_input_length: int,
    top_k: int | None = None,
    vllm_base_url: str,
    vllm_model_name: str,
    vllm_api_key: str,
    vllm_timeout: float,
    vllm_max_retries: int,
    enable_thinking: bool = False,
) -> str:
    prompt_text = apply_chat_template_text(
        tokenizer,
        messages,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        model_name_or_path=vllm_model_name,
    )
    prompt_text = truncate_prompt_text(tokenizer, prompt_text, max_input_length)

    text, _ = call_vllm_completion(
        base_url=vllm_base_url,
        api_key=vllm_api_key,
        model_name=vllm_model_name,
        prompt=prompt_text,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        timeout=vllm_timeout,
        max_retries=vllm_max_retries,
    )
    return text.strip()


def generate_with_thinking_budget_hf(
    model,
    tokenizer,
    messages: list[dict[str, str]],
    *,
    thinking_budget_tokens: int,
    answer_budget_tokens: int,
    early_stopping_text: str,
    temperature: float,
    top_p: float,
    max_input_length: int,
    top_k: int | None = None,
    model_name_or_path: str | None = None,
) -> str:
    prompt_text = apply_chat_template_text(
        tokenizer,
        messages,
        add_generation_prompt=True,
        enable_thinking=True,
        model_name_or_path=model_name_or_path,
    )

    model_inputs = tokenizer(
        [prompt_text],
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
    )

    model_device = next(model.parameters()).device
    model_inputs = {k: v.to(model_device) for k, v in model_inputs.items()}
    prompt_len = model_inputs["input_ids"].shape[-1]

    do_sample = temperature > 0.0
    gen_kwargs = {
        "do_sample": do_sample,
        "temperature": temperature if do_sample else None,
        "top_p": top_p if do_sample else None,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample and top_k is not None and int(top_k) > 0:
        gen_kwargs["top_k"] = int(top_k)
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    with torch.no_grad():
        first_ids = model.generate(
            **model_inputs,
            max_new_tokens=thinking_budget_tokens,
            **gen_kwargs,
        )

    completion_ids = first_ids[0][prompt_len:].tolist()

    eos_id = tokenizer.eos_token_id
    im_end_id = maybe_special_id(tokenizer, "<|im_end|>")
    stop_text = normalize_early_stopping_text(early_stopping_text, model_name_or_path)
    close_tag_candidates = [tag for tag in {stop_text.strip().splitlines()[-1].strip()} if tag]
    if not close_tag_candidates:
        close_tag_candidates = []
    think_close_ids_candidates = [
        tokenizer.encode(tag, add_special_tokens=False)
        for tag in close_tag_candidates
        if str(tag).strip()
    ]

    finished = False
    if eos_id is not None and eos_id in completion_ids:
        finished = True
    if im_end_id is not None and im_end_id in completion_ids:
        finished = True

    if not finished:
        continued_ids = first_ids
        has_think_close = any(
            think_ids and contains_subsequence(completion_ids, think_ids)
            for think_ids in think_close_ids_candidates
        )
        if not has_think_close:
            extra_ids = tokenizer(
                [stop_text],
                return_tensors="pt",
                add_special_tokens=False,
            )["input_ids"].to(model_device)
            continued_ids = torch.cat([continued_ids, extra_ids], dim=-1)

        attention_mask = torch.ones_like(continued_ids, dtype=torch.long, device=model_device)

        with torch.no_grad():
            second_ids = model.generate(
                input_ids=continued_ids,
                attention_mask=attention_mask,
                max_new_tokens=answer_budget_tokens,
                **gen_kwargs,
            )
        final_ids = second_ids[0][prompt_len:]
    else:
        final_ids = first_ids[0][prompt_len:]

    return tokenizer.decode(final_ids, skip_special_tokens=True).strip()


def generate_with_thinking_budget_vllm(
    *,
    tokenizer,
    messages: list[dict[str, str]],
    thinking_budget_tokens: int,
    answer_budget_tokens: int,
    early_stopping_text: str,
    temperature: float,
    top_p: float,
    max_input_length: int,
    top_k: int | None = None,
    vllm_base_url: str,
    vllm_model_name: str,
    vllm_api_key: str,
    vllm_timeout: float,
    vllm_max_retries: int,
    model_name_or_path: str | None = None,
) -> str:
    prompt_text = apply_chat_template_text(
        tokenizer,
        messages,
        add_generation_prompt=True,
        enable_thinking=True,
        model_name_or_path=vllm_model_name,
    )
    prompt_text = truncate_prompt_text(tokenizer, prompt_text, max_input_length)

    first_text, finish_reason = call_vllm_completion(
        base_url=vllm_base_url,
        api_key=vllm_api_key,
        model_name=vllm_model_name,
        prompt=prompt_text,
        max_tokens=thinking_budget_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        timeout=vllm_timeout,
        max_retries=vllm_max_retries,
    )

    if finish_reason != "length":
        return first_text.strip()

    stop_text = normalize_early_stopping_text(early_stopping_text, model_name_or_path)
    extra = ""
    if not contains_thinking_close(first_text):
        extra = stop_text

    continued_prompt = prompt_text + first_text + extra

    second_text, _ = call_vllm_completion(
        base_url=vllm_base_url,
        api_key=vllm_api_key,
        model_name=vllm_model_name,
        prompt=continued_prompt,
        max_tokens=answer_budget_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        timeout=vllm_timeout,
        max_retries=vllm_max_retries,
    )

    return (first_text + extra + second_text).strip()


def _normalize_turns(raw_turns: Any) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    if not isinstance(raw_turns, list):
        return out
    for idx, turn in enumerate(raw_turns, start=1):
        if not isinstance(turn, dict):
            continue
        utterance = str(turn.get("utterance", "")).strip()
        if not utterance:
            continue
        speaker = str(turn.get("speaker", "")).strip() or ("A" if idx % 2 == 1 else "B")
        out.append({"speaker": speaker, "utterance": utterance})
    return out


def _coerce_int(value: Any, default: int = -1) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    s = str(value or "").strip()
    if not s:
        return default
    try:
        return int(s)
    except Exception:
        return default


def build_qa_user_text(
    section: str,
    section_desc: str,
    content: str,
    question: str,
    *,
    pragmatic: bool,
) -> str:
    if pragmatic:
        desc_block = f"Category Description: {section_desc}\n" if section_desc else ""
        return (
            "Task: Pragmatic QA.\n"
            f"Pragmatic Category: {section}\n"
            f"{desc_block}\n"
            f"Context:\n{content}\n\n"
            f"Question:\n{question}\n\n"
            "Answer the question by inferring the speaker's intended meaning in context.\n"
            "Do not rely on literal wording alone.\n"
            "Use the pragmatic category above as guidance for what kind of implied meaning to look for.\n"
            "Consider what the speaker is trying to communicate, why this wording was chosen, "
            "and what the listener is expected to infer from the context.\n"
            "Do not infer more than the context and category support.\n"
            "Output only the final answer text.\n"
            "\n\nWrite your final response in \\boxed{} on the last line."
        )
    return (
        "Task: QA.\n"
        f"Context:\n{content}\n\n"
        f"Question:\n{question}\n\n"
        "Answer the question.\n"
        "Output only the final answer text.\n"
        "\n\nWrite your final response in \\boxed{} on the last line."
    )


def build_qa_tasks(
    rows: list[dict[str, Any]],
    *,
    max_samples: int,
    keep_pragmatic_instruction_in_sft: bool,
    teacher_prompt_mode: str,
) -> list[Task]:
    tasks: list[Task] = []
    prompt_mode = str(teacher_prompt_mode or "pragmatic").strip().lower()
    use_pragmatic_teacher = prompt_mode == "pragmatic"
    use_light_pragmatic_teacher = prompt_mode == "pragmatic_light"
    for row_idx, row in enumerate(rows):
        section = str(row.get("section", "unknown"))
        section_desc = _get_section_description(section, row)
        content = str(row.get("content", "")).strip()
        question = str(row.get("question", "")).strip()

        if use_pragmatic_teacher:
            teacher_system_text = TEACHER_SYSTEM_PROMPT_QA
        elif use_light_pragmatic_teacher:
            teacher_system_text = TEACHER_SYSTEM_PROMPT_QA_LIGHT
        else:
            teacher_system_text = SFT_SYSTEM_PROMPT_QA
        teacher_user_text = build_qa_user_text(
            section,
            section_desc,
            content,
            question,
            pragmatic=use_pragmatic_teacher,
        )
        if keep_pragmatic_instruction_in_sft:
            sft_system_text = teacher_system_text
            sft_user_text = teacher_user_text
        else:
            sft_system_text = SFT_SYSTEM_PROMPT_QA
            sft_user_text = build_qa_user_text(section, section_desc, content, question, pragmatic=False)

        sample_id = row.get("id", row_idx)
        tasks.append(
            Task(
                task_id=f"qa-{sample_id}",
                task_type="qa",
                source_dataset="pragmaticQA_dataset_Qwen14b",
                teacher_messages=[
                    {"role": "system", "content": teacher_system_text},
                    {"role": "user", "content": teacher_user_text},
                ],
                sft_messages=[
                    {"role": "system", "content": sft_system_text},
                    {"role": "user", "content": sft_user_text},
                ],
                metadata={
                    "section": section,
                    "section_description": section_desc,
                    "reference_answer": str(row.get("answer", "")).strip(),
                    "teacher_prompt_mode": prompt_mode,
                    "source_row": row,
                },
            )
        )

        if max_samples > 0 and len(tasks) >= max_samples:
            break

    return tasks


def _record_group_key(record: dict[str, Any], fallback_idx: int) -> str:
    metadata = record.get("metadata", {})
    source_row = metadata.get("source_row", {}) if isinstance(metadata, dict) else {}
    candidates = []
    if isinstance(source_row, dict):
        candidates.extend(
            [
                source_row.get("domain_index"),
                source_row.get("domain_seed"),
                source_row.get("domain"),
            ]
        )
    if isinstance(metadata, dict):
        candidates.extend(
            [
                metadata.get("domain_index"),
                metadata.get("domain_seed"),
                metadata.get("domain"),
            ]
        )

    for value in candidates:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return f"domain::{text}"
    return f"record::{fallback_idx}"


def split_train_val(records: list[dict[str, Any]], val_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not records:
        return [], []

    groups: dict[str, list[int]] = {}
    for idx, record in enumerate(records):
        groups.setdefault(_record_group_key(record, idx), []).append(idx)

    rng = random.Random(seed)
    group_keys = list(groups.keys())
    rng.shuffle(group_keys)

    val_size = int(len(records) * val_ratio)
    if val_ratio > 0 and val_size == 0:
        val_size = 1

    val_idx: set[int] = set()
    if val_size > 0:
        for key in group_keys:
            if len(val_idx) >= val_size and val_idx:
                break
            val_idx.update(groups[key])

    train = [records[i] for i in range(len(records)) if i not in val_idx]
    val = [records[i] for i in range(len(records)) if i in val_idx]
    return train, val


def resolve_vllm_model_spec(model_spec: str) -> dict[str, str | None]:
    raw = str(model_spec or "").strip()
    if "::" in raw:
        base_model, lora_adapter = [part.strip() for part in raw.split("::", 1)]
        display_name = Path(lora_adapter).name
    else:
        base_model = raw
        lora_adapter = ""
        display_name = ""
    if not base_model:
        raise ValueError(f"Invalid model spec without base model: {model_spec!r}")
    if not display_name:
        display_name = Path(lora_adapter).name if lora_adapter else base_model.rsplit("/", 1)[-1]
    return {
        "base_model": base_model,
        "lora_adapter": lora_adapter or None,
        "display_name": display_name,
    }


def tokenizer_model_name_for_generation(model_name_or_path: str) -> str:
    return str(resolve_vllm_model_spec(model_name_or_path)["base_model"])



class LocalVLLMBatchedGenerator:
    def __init__(
        self,
        *,
        model_name: str,
        tokenizer,
        dtype: str,
        max_model_len: int,
        gpu_memory_utilization: float,
        tensor_parallel_size: int,
        seed: int,
    ) -> None:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        resolved = resolve_vllm_model_spec(model_name)
        base_model = str(resolved["base_model"])
        lora_adapter = resolved["lora_adapter"]
        self._SamplingParams = SamplingParams
        self._tokenizer = tokenizer
        self._lora_request = None
        llm_kwargs: dict[str, Any] = {
            "model": base_model,
            "dtype": dtype,
            "max_model_len": int(max_model_len),
            "trust_remote_code": True,
            "gpu_memory_utilization": float(gpu_memory_utilization),
            "tensor_parallel_size": int(tensor_parallel_size),
            "seed": int(seed),
        }
        if is_ministral3_model_name(base_model):
            llm_kwargs["tokenizer_mode"] = "mistral"
            llm_kwargs["config_format"] = "mistral"
            llm_kwargs["load_format"] = "mistral"
        if lora_adapter:
            llm_kwargs["enable_lora"] = True
            self._lora_request = LoRARequest(
                lora_name=str(resolved["display_name"]),
                lora_int_id=1,
                lora_path=str(lora_adapter),
                base_model_name=base_model,
            )
        self._llm = LLM(
            **llm_kwargs,
        )

    def _build_prompt_texts(
        self,
        message_batches: list[list[dict[str, str]]],
        *,
        enable_thinking: bool,
        max_input_length: int,
        model_name_or_path: str | None = None,
    ) -> list[str]:
        prompt_texts: list[str] = []
        for messages in message_batches:
            prompt_text = apply_chat_template_text(
                self._tokenizer,
                messages,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
                model_name_or_path=model_name_or_path,
            )
            prompt_texts.append(truncate_prompt_text(self._tokenizer, prompt_text, max_input_length))
        return prompt_texts

    def _sampling_params(
        self,
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int | None,
    ):
        do_sample = temperature > 0.0
        return self._SamplingParams(
            max_tokens=max(1, int(max_tokens)),
            temperature=temperature if do_sample else 0.0,
            top_p=top_p if do_sample else 1.0,
            top_k=int(top_k) if (do_sample and top_k is not None and int(top_k) > 0) else -1,
        )

    def generate_chat_batch(
        self,
        message_batches: list[list[dict[str, str]]],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        max_input_length: int,
        top_k: int | None = None,
        enable_thinking: bool = False,
        model_name_or_path: str | None = None,
    ) -> list[str]:
        prompt_texts = self._build_prompt_texts(
            message_batches,
            enable_thinking=enable_thinking,
            max_input_length=max_input_length,
            model_name_or_path=model_name_or_path,
        )
        outputs = self._llm.generate(
            prompt_texts,
            self._sampling_params(
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            ),
            use_tqdm=True,
            lora_request=self._lora_request,
        )
        return [str(output.outputs[0].text).strip() for output in outputs]

    def generate_with_thinking_budget_batch(
        self,
        message_batches: list[list[dict[str, str]]],
        *,
        thinking_budget_tokens: int,
        answer_budget_tokens: int,
        early_stopping_text: str,
        temperature: float,
        top_p: float,
        max_input_length: int,
        top_k: int | None = None,
        model_name_or_path: str | None = None,
    ) -> list[str]:
        prompt_texts = self._build_prompt_texts(
            message_batches,
            enable_thinking=True,
            max_input_length=max_input_length,
            model_name_or_path=model_name_or_path,
        )
        first_outputs = self._llm.generate(
            prompt_texts,
            self._sampling_params(
                max_tokens=thinking_budget_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            ),
            use_tqdm=True,
            lora_request=self._lora_request,
        )

        results = [""] * len(prompt_texts)
        needs_second: list[tuple[int, str, str, str]] = []
        stop_text = normalize_early_stopping_text(early_stopping_text, model_name_or_path)
        second_pass_limit = max(1, int(max_input_length) - 32)
        for idx, output in enumerate(first_outputs):
            first_text = str(output.outputs[0].text)
            finish_reason = str(output.outputs[0].finish_reason or "")
            if finish_reason != "length":
                results[idx] = first_text.strip()
                continue
            extra = "" if contains_thinking_close(first_text) else stop_text
            continued_prompt = truncate_prompt_text_from_left(
                self._tokenizer,
                prompt_texts[idx] + first_text + extra,
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
                use_tqdm=True,
                lora_request=self._lora_request,
            )
            for j, (idx, _, first_text, extra) in enumerate(needs_second):
                second_text = str(second_outputs[j].outputs[0].text)
                results[idx] = (first_text + extra + second_text).strip()

        return results

    @staticmethod
    def _position_logprob(prompt_logprobs: Any, position: int, token_id: int) -> float:
        if prompt_logprobs is None:
            return float("-inf")
        try:
            entry = prompt_logprobs[position]
        except Exception:
            return float("-inf")
        if entry is None or not isinstance(entry, dict):
            return float("-inf")
        candidate = entry.get(int(token_id))
        if candidate is None:
            return float("-inf")
        value = getattr(candidate, "logprob", candidate)
        try:
            return float(value)
        except Exception:
            return float("-inf")

    def score_continuations_batch(
        self,
        message_batches: list[list[dict[str, str]]],
        continuations: list[str],
        *,
        max_input_length: int,
        model_name_or_path: str | None = None,
        assistant_prefixes: list[str] | None = None,
        enable_thinking: bool = False,
        score_batch_size: int = 1024,
    ) -> list[dict[str, float]]:
        prompt_texts = self._build_prompt_texts(
            message_batches,
            enable_thinking=enable_thinking,
            max_input_length=max_input_length,
            model_name_or_path=model_name_or_path,
        )
        if assistant_prefixes is None:
            assistant_prefixes = [""] * len(prompt_texts)
        if len(assistant_prefixes) != len(prompt_texts):
            raise ValueError("assistant_prefixes length must match message_batches length")
        full_texts: list[str] = []
        prompt_lengths: list[int] = []
        keys: list[tuple[int, str]] = []
        for batch_idx, prompt_text in enumerate(prompt_texts):
            prompt_text = prompt_text + str(assistant_prefixes[batch_idx] or "")
            prompt_len = len(self._tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
            for continuation in continuations:
                full_texts.append(prompt_text + continuation)
                prompt_lengths.append(prompt_len)
                keys.append((batch_idx, continuation))

        scores: list[dict[str, float]] = [dict() for _ in message_batches]
        chunk_size = max(1, int(score_batch_size or 1))
        if len(full_texts) > chunk_size:
            print(f"[judge] scoring expanded_prompts={len(full_texts)} chunk_size={chunk_size}")
        sampling_params = self._SamplingParams(
            max_tokens=1,
            temperature=0.0,
            top_p=1.0,
            prompt_logprobs=1,
        )
        for start in range(0, len(full_texts), chunk_size):
            end = min(start + chunk_size, len(full_texts))
            outputs = self._llm.generate(
                full_texts[start:end],
                sampling_params,
                use_tqdm=True,
                lora_request=self._lora_request,
            )
            for output, prompt_len, (batch_idx, continuation) in zip(
                outputs,
                prompt_lengths[start:end],
                keys[start:end],
            ):
                token_ids = list(getattr(output, "prompt_token_ids", []) or [])
                prompt_logprobs = getattr(output, "prompt_logprobs", None)
                if prompt_len <= 0 or prompt_len >= len(token_ids):
                    scores[batch_idx][continuation] = float("-inf")
                    continue

                total = 0.0
                for pos in range(prompt_len, len(token_ids)):
                    lp = self._position_logprob(prompt_logprobs, pos, int(token_ids[pos]))
                    if not math.isfinite(lp):
                        total = float("-inf")
                        break
                    total += lp
                scores[batch_idx][continuation] = total
        flat_scores = [score for score_map in scores for score in score_map.values()]
        finite_scores = sum(1 for score in flat_scores if math.isfinite(float(score)))
        if flat_scores and finite_scores == 0:
            print(
                "[judge] warning: all continuation logprobs are non-finite; "
                "tokenizer/prompt alignment may be broken for this model/backend"
            )
        return scores

    def generate_judge_thinking_prefixes_batch(
        self,
        message_batches: list[list[dict[str, str]]],
        *,
        max_new_tokens: int,
        early_stopping_text: str,
        temperature: float,
        top_p: float,
        max_input_length: int,
        top_k: int | None = None,
        model_name_or_path: str | None = None,
    ) -> list[str]:
        raw_outputs = self.generate_chat_batch(
            message_batches,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_input_length=max_input_length,
            enable_thinking=True,
            model_name_or_path=model_name_or_path,
        )
        return [
            trim_thinking_prefix_through_close(
                raw,
                early_stopping_text=early_stopping_text,
                model_name_or_path=model_name_or_path,
            )
            for raw in raw_outputs
        ]

    def close(self) -> None:
        del self._llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_records_with_vllm_local_batched(args: argparse.Namespace, tasks: list[Task], tokenizer):
    if not tasks:
        return [], None

    judge_refine_rounds = args.judge_refine_rounds
    judge_drop_failed = args.judge_drop_failed
    judge_mode = "refine"
    if args.judge_drop_without_refine:
        judge_refine_rounds = 0
        judge_drop_failed = True
        judge_mode = "drop_only"

    judge_enabled = judge_refine_rounds > 0 or args.judge_drop_without_refine
    max_model_len = int(args.max_input_length) + max(
        int(args.thinking_budget_tokens) + int(args.answer_budget_tokens),
        int(args.judge_max_new_tokens),
    ) + 512
    if int(getattr(args, "vllm_max_model_len", 0) or 0) > 0:
        max_model_len = int(args.vllm_max_model_len)

    generator = LocalVLLMBatchedGenerator(
        model_name=args.model_name_or_path,
        tokenizer=tokenizer,
        dtype=str(args.dtype),
        max_model_len=max_model_len,
        gpu_memory_utilization=float(args.vllm_gpu_memory_utilization),
        tensor_parallel_size=int(args.vllm_tensor_parallel_size),
        seed=int(args.seed),
    )

    try:
        empty_recovered_plain = 0
        empty_dropped = 0

        def teacher_batch(message_batches: list[list[dict[str, str]]]) -> list[str]:
            kwargs = {
                "thinking_budget_tokens": args.thinking_budget_tokens,
                "answer_budget_tokens": args.answer_budget_tokens,
                "early_stopping_text": args.early_stopping_text,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "max_input_length": args.max_input_length,
                "model_name_or_path": args.model_name_or_path,
            }
            return generator.generate_with_thinking_budget_batch(message_batches, **kwargs)

        def retry_empty_with_plain(message_map: dict[int, list[dict[str, str]]]) -> int:
            empty_items = [
                (idx, messages)
                for idx, messages in message_map.items()
                if not has_nonempty_final_output(assistant_texts[idx])
            ]
            if not empty_items:
                return 0

            plain_kwargs = {
                "max_new_tokens": args.answer_budget_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "max_input_length": args.max_input_length,
                "enable_thinking": False,
                "model_name_or_path": args.model_name_or_path,
            }
            plain_outputs = generator.generate_chat_batch([messages for _, messages in empty_items], **plain_kwargs)

            recovered = 0
            for (idx, _), text in zip(empty_items, plain_outputs):
                assistant_texts[idx] = text.strip()
                if has_nonempty_final_output(assistant_texts[idx]):
                    recovered += 1
            return recovered

        if judge_enabled:
            print(
                f"[judge] enabled mode={judge_mode} rounds={judge_refine_rounds} "
                f"drop_failed={judge_drop_failed} max_new_tokens={args.judge_max_new_tokens} "
                f"method={args.judge_method} margin_threshold={args.judge_margin_threshold} "
                f"confidence_threshold={args.judge_confidence_threshold} "
                f"judge_thinking={bool(args.judge_enable_thinking)}"
            )

        print(f"[teacher] batched generation for {len(tasks)} tasks")
        assistant_texts = teacher_batch([task.teacher_messages for task in tasks])
        empty_recovered_plain += retry_empty_with_plain(
            {idx: task.teacher_messages for idx, task in enumerate(tasks)}
        )

        judge_info_by_idx: list[dict[str, Any] | None] = [None] * len(tasks)
        keep_mask = [True] * len(tasks)
        judge_summary: dict[str, int] | None = None

        if judge_enabled:
            judge_total = len(tasks)
            judge_passed = 0
            judge_recovered = 0
            judge_failed = 0
            judge_dropped = 0

            initial_passes: list[bool | None] = [None] * len(tasks)
            final_passes = [False] * len(tasks)
            attempts = [0] * len(tasks)
            last_feedbacks = [""] * len(tasks)
            last_judgments: list[str | None] = [None] * len(tasks)
            last_confidences: list[int | None] = [None] * len(tasks)
            last_yes_logprobs: list[float | None] = [None] * len(tasks)
            last_no_logprobs: list[float | None] = [None] * len(tasks)
            last_yes_probs: list[float | None] = [None] * len(tasks)
            last_no_probs: list[float | None] = [None] * len(tasks)
            last_margins: list[float | None] = [None] * len(tasks)
            last_raw_outputs = [""] * len(tasks)
            active_indices = list(range(len(tasks)))
            yes_continuations = ["yes", " yes", "Yes", " Yes"]
            no_continuations = ["no", " no", "No", " No"]
            judge_continuations = yes_continuations + no_continuations

            for refine_step in range(judge_refine_rounds + 1):
                if not active_indices:
                    break
                print(f"[judge] round={refine_step} pending={len(active_indices)}")
                judge_batches = []
                judge_eval_indices: list[int] = []
                next_refine_indices: list[int] = []
                next_refine_messages: list[list[dict[str, str]]] = []

                for idx in active_indices:
                    attempts[idx] = refine_step + 1
                    candidate_final = extract_final_output_text(assistant_texts[idx])
                    if not candidate_final:
                        if refine_step == 0:
                            initial_passes[idx] = False
                        last_feedbacks[idx] = "Candidate answer was empty."
                        if refine_step < judge_refine_rounds:
                            empty_result = JudgeResult(
                                passed=False,
                                feedback=last_feedbacks[idx],
                                corrected_answer=str(tasks[idx].metadata.get("reference_answer", "")).strip(),
                                raw_output="",
                            )
                            next_refine_indices.append(idx)
                            next_refine_messages.append(
                                build_refinement_messages(tasks[idx], assistant_texts[idx], empty_result)
                            )
                        continue
                    judge_eval_indices.append(idx)
                    if args.judge_method == "margin":
                        judge_batches.append(build_margin_judge_messages(tasks[idx], candidate_final))
                    else:
                        judge_batches.append(build_judge_messages(tasks[idx], candidate_final))

                judge_raws: list[str] = []
                if judge_batches:
                    if args.judge_method == "margin":
                        judge_scores = generator.score_continuations_batch(
                            judge_batches,
                            judge_continuations,
                            max_input_length=args.max_input_length,
                            enable_thinking=False,
                            model_name_or_path=args.model_name_or_path,
                            score_batch_size=args.judge_score_batch_size,
                        )
                        for score in judge_scores:
                            p_yes, yes_logprob = _continuation_probability_mass(score, yes_continuations)
                            p_no, no_logprob = _continuation_probability_mass(score, no_continuations)
                            judge_raws.append(
                                json.dumps(
                                    {
                                        "yes_logprob": yes_logprob,
                                        "no_logprob": no_logprob,
                                        "p_yes": p_yes,
                                        "p_no": p_no,
                                    },
                                    ensure_ascii=False,
                                )
                            )
                    else:
                        judge_raws = generator.generate_chat_batch(
                            judge_batches,
                            max_new_tokens=args.judge_max_new_tokens,
                            temperature=args.judge_temperature,
                            top_p=args.judge_top_p,
                            top_k=args.judge_top_k,
                            max_input_length=args.max_input_length,
                            enable_thinking=bool(args.judge_enable_thinking),
                            model_name_or_path=args.model_name_or_path,
                        )

                for pos, idx in enumerate(judge_eval_indices):
                    if args.judge_method == "margin":
                        parsed_raw = json.loads(judge_raws[pos])
                        judge_result = build_logprob_judge_result(
                            yes_logprob=float(parsed_raw["yes_logprob"]),
                            no_logprob=float(parsed_raw["no_logprob"]),
                            margin_threshold=float(args.judge_margin_threshold),
                            raw_output=judge_raws[pos],
                        )
                    else:
                        judge_result = parse_confidence_judge_result(
                            judge_raws[pos],
                            confidence_threshold=args.judge_confidence_threshold,
                        )
                    if refine_step == 0:
                        initial_passes[idx] = judge_result.passed
                    last_feedbacks[idx] = judge_result.feedback
                    last_judgments[idx] = judge_result.judgment
                    last_confidences[idx] = judge_result.confidence
                    last_yes_logprobs[idx] = judge_result.yes_logprob
                    last_no_logprobs[idx] = judge_result.no_logprob
                    last_yes_probs[idx] = judge_result.yes_prob
                    last_no_probs[idx] = judge_result.no_prob
                    last_margins[idx] = judge_result.logprob_margin
                    last_raw_outputs[idx] = judge_result.raw_output
                    if judge_result.passed:
                        final_passes[idx] = True
                        continue
                    if refine_step >= judge_refine_rounds:
                        continue
                    next_refine_indices.append(idx)
                    next_refine_messages.append(build_refinement_messages(tasks[idx], assistant_texts[idx], judge_result))

                if not next_refine_indices:
                    active_indices = []
                    continue

                refined_outputs = teacher_batch(next_refine_messages)
                for idx, refined in zip(next_refine_indices, refined_outputs):
                    assistant_texts[idx] = refined
                empty_recovered_plain += retry_empty_with_plain(
                    {idx: messages for idx, messages in zip(next_refine_indices, next_refine_messages)}
                )
                active_indices = next_refine_indices

            for idx in range(len(tasks)):
                if final_passes[idx]:
                    judge_passed += 1
                else:
                    judge_failed += 1
                if initial_passes[idx] is False and final_passes[idx]:
                    judge_recovered += 1
                if judge_drop_failed and not final_passes[idx]:
                    keep_mask[idx] = False
                    judge_dropped += 1
                judge_info_by_idx[idx] = {
                    "enabled": True,
                    "mode": judge_mode,
                    "rounds": judge_refine_rounds,
                    "attempts": attempts[idx],
                    "initial_pass": bool(initial_passes[idx]),
                    "final_pass": final_passes[idx],
                    "last_feedback": last_feedbacks[idx],
                    "last_judgment": last_judgments[idx],
                    "last_confidence": last_confidences[idx],
                    "last_yes_logprob": last_yes_logprobs[idx],
                    "last_no_logprob": last_no_logprobs[idx],
                    "last_p_yes": last_yes_probs[idx],
                    "last_p_no": last_no_probs[idx],
                    "last_logprob_margin": last_margins[idx],
                    "margin_threshold": args.judge_margin_threshold,
                    "confidence_threshold": args.judge_confidence_threshold,
                    "last_raw_output": last_raw_outputs[idx],
                    "judge_method": args.judge_method,
                    "judge_enable_thinking": bool(args.judge_enable_thinking),
                }

            judge_summary = {
                "total": judge_total,
                "pass": judge_passed,
                "failed": judge_failed,
                "recovered": judge_recovered,
                "dropped": judge_dropped,
                "empty_recovered_plain": empty_recovered_plain,
                "empty_dropped": 0,
            }

        generated_records: list[dict[str, Any]] = []
        for idx, task in enumerate(tasks):
            if not keep_mask[idx]:
                continue
            if not has_nonempty_final_output(assistant_texts[idx]):
                empty_dropped += 1
                continue
            judge_info = judge_info_by_idx[idx]
            assistant_message = {"role": "assistant", "content": assistant_texts[idx]}
            generated_records.append(
                {
                    "id": task.task_id,
                    "task_type": task.task_type,
                    "source_dataset": task.source_dataset,
                    "messages": task.sft_messages + [assistant_message],
                    "assistant_response": assistant_texts[idx],
                    "thinking_budget_tokens": args.thinking_budget_tokens,
                    "answer_budget_tokens": args.answer_budget_tokens,
                    "metadata": {
                        **task.metadata,
                        "sft_prompt_mode": "pragmatic" if args.keep_pragmatic_instruction_in_sft else "neutral",
                        "generation_backend": args.gen_backend,
                        **({"judge": judge_info} if judge_info is not None else {}),
                    },
                }
            )

        if empty_recovered_plain > 0 or empty_dropped > 0:
            print(f"[teacher] empty_recovered_plain={empty_recovered_plain} empty_dropped={empty_dropped}")
        if judge_summary is not None:
            judge_summary["empty_dropped"] = empty_dropped

        return generated_records, judge_summary
    finally:
        generator.close()

def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    if args.output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output_path}. Use --overwrite to replace it.")

    qa_rows = load_jsonl(args.qa_data_path)

    if not qa_rows:
        raise ValueError("Set --qa_data_path to a non-empty QA dataset.")

    tasks = build_qa_tasks(
        qa_rows,
        max_samples=args.max_qa_samples,
        keep_pragmatic_instruction_in_sft=args.keep_pragmatic_instruction_in_sft,
        teacher_prompt_mode=args.teacher_prompt_mode,
    )
    rng = random.Random(args.seed)
    rng.shuffle(tasks)

    if args.max_samples > 0:
        tasks = tasks[: args.max_samples]

    judge_requested = args.judge_refine_rounds > 0 or args.judge_drop_without_refine
    if judge_requested and args.judge_method == "margin" and args.gen_backend not in {
        "vllm_local_batched",
    }:
        raise NotImplementedError(
            "--judge_method margin currently requires --gen_backend vllm_local_batched "
            "so yes/no continuation logprobs can be computed in batch."
        )

    tokenizer_name = tokenizer_model_name_for_generation(args.model_name_or_path)
    tokenizer_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if is_ministral3_model_name(tokenizer_name):
        tokenizer_kwargs["fix_mistral_regex"] = True
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)
    except TypeError:
        tokenizer_kwargs.pop("fix_mistral_regex", None)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.gen_backend == "vllm_local_batched":
        if args.temperature > 0.0 or args.judge_temperature > 0.0:
            print(
                "[warn] sampling enabled; outputs will be stochastic across backends. "
                "This is expected for Phi-style inference."
            )
        generated_records, judge_summary = build_records_with_vllm_local_batched(args, tasks, tokenizer)
        if judge_summary is not None:
            print(
                f"[judge] total={judge_summary['total']} pass={judge_summary['pass']} failed={judge_summary['failed']} "
                f"recovered={judge_summary['recovered']} dropped={judge_summary['dropped']}"
            )

        write_jsonl(args.output_path, generated_records)
        train_records, val_records = split_train_val(generated_records, args.val_ratio, args.seed)
        write_jsonl(args.train_output_path, train_records)
        write_jsonl(args.val_output_path, val_records)

        print(f"[done] all={len(generated_records)} train={len(train_records)} val={len(val_records)}")
        print(f"[done] wrote: {args.output_path}")
        print(f"[done] wrote: {args.train_output_path}")
        print(f"[done] wrote: {args.val_output_path}")
        return

    model = None
    if args.gen_backend == "hf":
        dtype = resolve_dtype(args.dtype)
        model_kwargs: dict[str, Any] = {"torch_dtype": dtype}
        if args.device_map:
            model_kwargs["device_map"] = args.device_map
        if args.load_in_4bit:
            model_kwargs["load_in_4bit"] = True

        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
        if not args.device_map:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
        model.eval()

    vllm_model_name = args.vllm_model_name or args.model_name_or_path

    def generate_teacher(messages: list[dict[str, str]]) -> str:
        if args.gen_backend == "hf":
            assert model is not None
            return generate_with_thinking_budget_hf(
                model,
                tokenizer,
                messages,
                thinking_budget_tokens=args.thinking_budget_tokens,
                answer_budget_tokens=args.answer_budget_tokens,
                early_stopping_text=args.early_stopping_text,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_input_length=args.max_input_length,
                model_name_or_path=args.model_name_or_path,
            )
        return generate_with_thinking_budget_vllm(
            tokenizer=tokenizer,
            messages=messages,
            thinking_budget_tokens=args.thinking_budget_tokens,
            answer_budget_tokens=args.answer_budget_tokens,
            early_stopping_text=args.early_stopping_text,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_input_length=args.max_input_length,
            vllm_base_url=args.vllm_base_url,
            vllm_model_name=vllm_model_name,
            vllm_api_key=args.vllm_api_key,
            vllm_timeout=args.vllm_timeout,
            vllm_max_retries=args.vllm_max_retries,
            model_name_or_path=args.model_name_or_path,
        )

    def generate_teacher_plain(messages: list[dict[str, str]]) -> str:
        if args.gen_backend == "hf":
            assert model is not None
            return generate_chat_once_hf(
                model,
                tokenizer,
                messages,
                max_new_tokens=args.answer_budget_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_input_length=args.max_input_length,
                enable_thinking=False,
                model_name_or_path=args.model_name_or_path,
            )
        return generate_chat_once_vllm(
            tokenizer=tokenizer,
            messages=messages,
            max_new_tokens=args.answer_budget_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_input_length=args.max_input_length,
            vllm_base_url=args.vllm_base_url,
            vllm_model_name=vllm_model_name,
            vllm_api_key=args.vllm_api_key,
            vllm_timeout=args.vllm_timeout,
            vllm_max_retries=args.vllm_max_retries,
            enable_thinking=False,
        )

    def generate_judge(messages: list[dict[str, str]]) -> str:
        if args.gen_backend == "hf":
            assert model is not None
            return generate_chat_once_hf(
                model,
                tokenizer,
                messages,
                max_new_tokens=args.judge_max_new_tokens,
                temperature=args.judge_temperature,
                top_p=args.judge_top_p,
                top_k=args.judge_top_k,
                max_input_length=args.max_input_length,
                enable_thinking=bool(args.judge_enable_thinking),
                model_name_or_path=args.model_name_or_path,
            )
        if args.gen_backend == "vllm_api":
            return generate_chat_once_vllm(
                tokenizer=tokenizer,
                messages=messages,
                max_new_tokens=args.judge_max_new_tokens,
                temperature=args.judge_temperature,
                top_p=args.judge_top_p,
                top_k=args.judge_top_k,
                max_input_length=args.max_input_length,
                vllm_base_url=args.vllm_base_url,
                vllm_model_name=vllm_model_name,
                vllm_api_key=args.vllm_api_key,
                vllm_timeout=args.vllm_timeout,
                vllm_max_retries=args.vllm_max_retries,
                enable_thinking=bool(args.judge_enable_thinking),
            )
        raise NotImplementedError(
            "Generated confidence SFT judge is implemented for --gen_backend hf, "
            "vllm_api, and vllm_local_batched."
        )

    generated_records: list[dict[str, Any]] = []

    judge_refine_rounds = args.judge_refine_rounds
    judge_drop_failed = args.judge_drop_failed
    judge_mode = "refine"
    if args.judge_drop_without_refine:
        judge_refine_rounds = 0
        judge_drop_failed = True
        judge_mode = "drop_only"

    judge_enabled = judge_refine_rounds > 0 or args.judge_drop_without_refine
    judge_total = 0
    judge_passed = 0
    judge_recovered = 0
    judge_failed = 0
    judge_dropped = 0
    empty_recovered_plain = 0
    empty_dropped = 0

    if judge_enabled:
        print(
            f"[judge] enabled mode={judge_mode} rounds={judge_refine_rounds} "
            f"drop_failed={judge_drop_failed} max_new_tokens={args.judge_max_new_tokens} "
            f"confidence_threshold={args.judge_confidence_threshold} "
            f"judge_thinking={bool(args.judge_enable_thinking)}"
        )

    for task in tqdm(tasks, desc="Generating SFT targets"):
        assistant_text = generate_teacher(task.teacher_messages)
        if not has_nonempty_final_output(assistant_text):
            assistant_text = generate_teacher_plain(task.teacher_messages)
            if has_nonempty_final_output(assistant_text):
                empty_recovered_plain += 1

        judge_info: dict[str, Any] | None = None
        if judge_enabled:
            judge_total += 1
            initial_pass: bool | None = None
            final_pass = False
            attempts = 0
            last_feedback = ""
            last_judgment: str | None = None
            last_confidence: int | None = None
            last_raw_output = ""

            for refine_step in range(judge_refine_rounds + 1):
                attempts = refine_step + 1
                candidate_final = extract_final_output_text(assistant_text)
                if not candidate_final:
                    if refine_step == 0:
                        initial_pass = False
                    last_feedback = "Candidate answer was empty."
                    if refine_step >= judge_refine_rounds:
                        break
                    empty_result = JudgeResult(
                        passed=False,
                        feedback=last_feedback,
                        corrected_answer=str(task.metadata.get("reference_answer", "")).strip(),
                        raw_output="",
                    )
                    refine_messages = build_refinement_messages(task, assistant_text, empty_result)
                    assistant_text = generate_teacher(refine_messages)
                    if not has_nonempty_final_output(assistant_text):
                        assistant_text = generate_teacher_plain(refine_messages)
                        if has_nonempty_final_output(assistant_text):
                            empty_recovered_plain += 1
                    continue
                judge_messages = build_judge_messages(task, candidate_final)
                judge_raw = generate_judge(judge_messages)
                judge_result = parse_confidence_judge_result(
                    judge_raw,
                    confidence_threshold=args.judge_confidence_threshold,
                )

                if refine_step == 0:
                    initial_pass = judge_result.passed

                last_feedback = judge_result.feedback
                last_judgment = judge_result.judgment
                last_confidence = judge_result.confidence
                last_raw_output = judge_result.raw_output
                if judge_result.passed:
                    final_pass = True
                    break

                if refine_step >= judge_refine_rounds:
                    break

                refine_messages = build_refinement_messages(task, assistant_text, judge_result)
                assistant_text = generate_teacher(refine_messages)
                if not has_nonempty_final_output(assistant_text):
                    assistant_text = generate_teacher_plain(refine_messages)
                    if has_nonempty_final_output(assistant_text):
                        empty_recovered_plain += 1

            if final_pass:
                judge_passed += 1
            else:
                judge_failed += 1

            if initial_pass is False and final_pass:
                judge_recovered += 1

            if judge_drop_failed and not final_pass:
                judge_dropped += 1
                continue

            judge_info = {
                "enabled": True,
                "mode": judge_mode,
                "rounds": judge_refine_rounds,
                "attempts": attempts,
                "initial_pass": bool(initial_pass),
                "final_pass": final_pass,
                "last_feedback": last_feedback,
                "last_judgment": last_judgment,
                "last_confidence": last_confidence,
                "confidence_threshold": args.judge_confidence_threshold,
                "last_raw_output": last_raw_output,
                "judge_enable_thinking": bool(args.judge_enable_thinking),
            }

        if not has_nonempty_final_output(assistant_text):
            empty_dropped += 1
            continue

        record = {
            "id": task.task_id,
            "task_type": task.task_type,
            "source_dataset": task.source_dataset,
            "messages": task.sft_messages + [{"role": "assistant", "content": assistant_text}],
            "assistant_response": assistant_text,
            "thinking_budget_tokens": args.thinking_budget_tokens,
            "answer_budget_tokens": args.answer_budget_tokens,
            "metadata": {
                **task.metadata,
                "sft_prompt_mode": "pragmatic" if args.keep_pragmatic_instruction_in_sft else "neutral",
                "generation_backend": args.gen_backend,
                **({"judge": judge_info} if judge_info is not None else {}),
            },
        }
        generated_records.append(record)

    if judge_enabled:
        print(
            f"[judge] total={judge_total} pass={judge_passed} failed={judge_failed} "
            f"recovered={judge_recovered} dropped={judge_dropped}"
        )
    if empty_recovered_plain > 0 or empty_dropped > 0:
        print(f"[teacher] empty_recovered_plain={empty_recovered_plain} empty_dropped={empty_dropped}")

    write_jsonl(args.output_path, generated_records)

    train_records, val_records = split_train_val(generated_records, args.val_ratio, args.seed)
    write_jsonl(args.train_output_path, train_records)
    write_jsonl(args.val_output_path, val_records)

    print(f"[done] all={len(generated_records)} train={len(train_records)} val={len(val_records)}")
    print(f"[done] wrote: {args.output_path}")
    print(f"[done] wrote: {args.train_output_path}")
    print(f"[done] wrote: {args.val_output_path}")


if __name__ == "__main__":
    main()
