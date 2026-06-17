"""
Reward function for PragReST QA GRPO training.

Reward structure per sample (total range [0.0, 2.0]):
  - format reward in [0.0, 1.0]
  - correctness (outcome) reward in {0.0, 1.0} when judge P(yes) - P(no) >
    JUDGE_MARGIN_THRESHOLD

Judge protocol matches src/filter_prompts.py: ask the judge to emit a
single yes/no token, then read P(yes), P(no) from first-token top_logprobs.

Env vars:
  JUDGE_BASE_URL         - judge vLLM server (default http://localhost:8100)
  JUDGE_MODEL_NAME       - served-model-name on judge server (default "judge")
  JUDGE_MARGIN_THRESHOLD - min P(yes)-P(no) margin for correctness (default 0.8)
  JUDGE_LOG_PATH         - path to per-call jsonl log
"""

import json
import logging
import math
import os
import threading

import requests
from requests.adapters import HTTPAdapter

logger = logging.getLogger(__name__)

_log_lock = threading.Lock()
_judge_log_path = os.environ.get("JUDGE_LOG_PATH", "judge_log.jsonl")
JUDGE_MARGIN_THRESHOLD = float(os.environ.get("JUDGE_MARGIN_THRESHOLD", "0.8"))


def _log_judge_call(
    task_type: str,
    messages: list[dict],
    judge_token: str | None,
    p_yes: float,
    p_no: float,
    margin: float,
    confident_yes: bool,
):
    record = {
        "task_type": task_type,
        "judge_prompt": messages,
        "judge_token": judge_token,
        "p_yes": p_yes,
        "p_no": p_no,
        "margin": margin,
        "margin_threshold": JUDGE_MARGIN_THRESHOLD,
        "confident_yes": confident_yes,
    }
    with _log_lock:
        os.makedirs(os.path.dirname(_judge_log_path) or ".", exist_ok=True)
        with open(_judge_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def strip_thinking(raw: str) -> str:
    idx = raw.rfind("</think>")
    return raw[idx + len("</think>"):].strip() if idx != -1 else raw


def extract_final_response(raw: str) -> str | None:
    after = strip_thinking(raw)
    start = after.rfind("\\boxed{")
    if start == -1:
        return None
    i = start + len("\\boxed{")
    depth = 1
    while i < len(after):
        c = after[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                candidate = after[start + len("\\boxed{"):i].strip()
                return candidate or None
        i += 1
    return None


def compute_format_reward(text: str) -> float:
    if "<think>" not in text or "</think>" not in text:
        return 0.0

    reward = 0.5
    if text.count("<think>") > 1 or text.count("</think>") > 1:
        return reward

    open_pos = text.find("<think>")
    close_pos = text.rfind("</think>")
    if open_pos < close_pos:
        reasoning = text[open_pos + len("<think>"): close_pos].strip()
        answer_part = text[close_pos + len("</think>"):].strip()
        if reasoning and extract_final_response(answer_part) is not None:
            reward += 0.5
    return reward


def check_response_format(text: str) -> bool:
    return extract_final_response(text) is not None


def get_qa_judge_fields(extra_info: dict, ground_truth: str) -> tuple[str, str, str]:
    row = extra_info.get("source_row", {})
    if not isinstance(row, dict):
        row = {}

    content = str(row.get("content", extra_info.get("content", "")) or "").strip()
    question = str(row.get("question", extra_info.get("question", "")) or "").strip()
    reference = str(
        extra_info.get(
            "reference_answer",
            row.get("answer", extra_info.get("reference", ground_truth)),
        )
        or ""
    ).strip()

    return content, question, reference


def build_qa_judge_prompt(
    content: str,
    question: str,
    reference: str,
    candidate_output: str,
) -> list[dict]:
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


_session = None
_session_lock = threading.Lock()


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:
                _session = requests.Session()
                adapter = HTTPAdapter(pool_connections=64, pool_maxsize=64)
                _session.mount("http://", adapter)
                _session.mount("https://", adapter)
    return _session


def extract_yes_no_probs(response_json: dict) -> tuple[float, float]:
    """Return (P(yes), P(no)) from the first-token top_logprobs."""
    try:
        choice = response_json["choices"][0]
        logprobs_field = choice.get("logprobs") or {}
        content = logprobs_field.get("content") or []
        if not content:
            return 0.0, 0.0
        top = content[0].get("top_logprobs") or []
    except (KeyError, IndexError, TypeError):
        return 0.0, 0.0

    p_yes = 0.0
    p_no = 0.0
    for entry in top:
        token = (entry.get("token") or "").strip().lower()
        logprob = entry.get("logprob")
        if logprob is None:
            continue
        if token == "yes":
            p_yes += math.exp(logprob)
        elif token == "no":
            p_no += math.exp(logprob)
    return p_yes, p_no


def is_confident_yes(
    p_yes: float,
    p_no: float,
    threshold: float = JUDGE_MARGIN_THRESHOLD,
) -> bool:
    return (p_yes - p_no) > threshold


def _call_judge_api(messages: list[dict]) -> dict:
    base_url = os.environ.get("JUDGE_BASE_URL", "http://localhost:8100").rstrip("/")
    model_name = os.environ.get("JUDGE_MODEL_NAME", "judge")

    resp = _get_session().post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model_name,
            "messages": messages,
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def _call_judge_with_retry(messages: list[dict], max_retries: int = 2) -> dict:
    for attempt in range(1 + max_retries):
        try:
            response_json = _call_judge_api(messages)
            p_yes, p_no = extract_yes_no_probs(response_json)
            margin = p_yes - p_no
            try:
                judge_token = response_json["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                judge_token = None
            confident_yes = is_confident_yes(p_yes, p_no)
            result = {
                "judge_token": judge_token,
                "p_yes": p_yes,
                "p_no": p_no,
                "margin": margin,
                "confident_yes": confident_yes,
            }
            logger.info(
                "[Judge] token=%r p_yes=%.6f p_no=%.6f margin=%+.6f confident_yes=%s",
                judge_token,
                p_yes,
                p_no,
                margin,
                confident_yes,
            )
            return result
        except Exception as exc:
            logger.warning(f"[Judge] Attempt {attempt + 1}: API error: {exc}")
    logger.warning("[Judge] All retry attempts exhausted — returning no correctness reward")
    return {
        "judge_token": None,
        "p_yes": 0.0,
        "p_no": 0.0,
        "margin": 0.0,
        "confident_yes": False,
    }


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
    **kwargs,
) -> float:
    _ = data_source, kwargs
    extra_info = extra_info or {}
    task_type = "qa"
    conv_type = "qa"

    fmt_r = compute_format_reward(solution_str)

    def _default_result(fmt_val: float) -> dict:
        return {
            "score": fmt_val,
            "acc": 0,
            "task_type": task_type,
            "conv_type": conv_type,
            "format_reward": fmt_val,
            "correctness_reward": 0.0,
            "judge_p_yes": 0.0,
            "judge_p_no": 0.0,
            "judge_margin": 0.0,
            "judge_margin_threshold": JUDGE_MARGIN_THRESHOLD,
            "judge_confident_yes": False,
        }

    candidate_output = extract_final_response(solution_str)
    if candidate_output is None:
        return _default_result(fmt_r)

    print(f"[Extract][{task_type}] candidate_output fed to judge: {candidate_output!r}", flush=True)

    try:
        content, question, reference = get_qa_judge_fields(extra_info, ground_truth)

        if not content or not question or not reference:
            logger.warning("[Judge][QA] Missing content/question/reference; skipping judge.")
            return _default_result(fmt_r)

        correctness_messages = build_qa_judge_prompt(
            content, question, reference, candidate_output,
        )

        judge_result = _call_judge_with_retry(correctness_messages)
        _log_judge_call(
            task_type,
            correctness_messages,
            judge_result["judge_token"],
            judge_result["p_yes"],
            judge_result["p_no"],
            judge_result["margin"],
            judge_result["confident_yes"],
        )

        correctness = 1.0 if judge_result["confident_yes"] else 0.0
        total_reward = fmt_r + correctness
        return {
            "score": total_reward,
            "acc": 1 if judge_result["confident_yes"] else 0,
            "task_type": task_type,
            "conv_type": conv_type,
            "format_reward": fmt_r,
            "correctness_reward": correctness,
            "judge_p_yes": judge_result["p_yes"],
            "judge_p_no": judge_result["p_no"],
            "judge_margin": judge_result["margin"],
            "judge_margin_threshold": JUDGE_MARGIN_THRESHOLD,
            "judge_confident_yes": judge_result["confident_yes"],
        }
    except Exception as exc:
        logger.error(f"[Judge][{task_type}] Unexpected error: {exc}", exc_info=True)
        return _default_result(fmt_r)
