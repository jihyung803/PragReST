"""Shared helpers for benchmark evaluation scripts.

This module contains common result-directory handling, PragMEGA prompt loading,
chat prompt construction, answer extraction, and pass-state classification.
"""

import csv
import os
import re
from pathlib import Path
from typing import Dict, List, Optional


# ── Constants ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "Choose the best answer option based on the given context.\n"
    "Final answer of the question must be enclosed in \\boxed{...}."
)

THINK_END_TOKEN_ID = 151668   # </think>
IM_END_TOKEN_ID = 151645      # <|im_end|>

EARLY_STOPPING_TEXT = (
    "\n\nConsidering the limited time by the user, I have to give the "
    "solution based on the thinking directly now.\n</think>\n\n"
)

# Default generation hyper-parameters (can be overridden per-caller)
DEFAULT_MAX_PROMPT_LENGTH = 2048
DEFAULT_THINKING_BUDGET = 824
DEFAULT_ANSWER_MAX_NEW_TOKENS = 256


def get_eval_results_dir(root, benchmark: str) -> Path:
    """Return the standard output directory for a benchmark eval run."""
    results_root = os.environ.get("EVAL_RESULTS_ROOT", "").strip()
    base = Path(results_root).expanduser() if results_root else Path(root) / "results"
    path = base / "eval" / benchmark
    path.mkdir(parents=True, exist_ok=True)
    return path


# ── Dataset loading ──────────────────────────────────────────────────────────

def load_pragmega_data(
    data_dir: str,
    focus_phenomena: Optional[List[str]] = None,
) -> List[Dict]:
    """Load PragMEGA prompt CSV files.

    Filters:
      - Only ``*_prompts_seed0_examples0.csv`` files
      - Skips filenames containing ``no-story``
      - Skips rows where ``is_example == "True"``
      - Optionally restricts to *focus_phenomena*

    Each returned dict has keys:
      full_prompt, correct_answer_idx, phenomenon, item_id
    """
    data_dir = Path(data_dir)
    examples: List[Dict] = []
    csv_files = [
        f for f in data_dir.glob("*_prompts_seed0_examples0.csv")
        if "no-story" not in f.name
    ]
    for csv_file in sorted(csv_files):
        with open(csv_file, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if row.get("is_example", "False") == "True":
                    continue
                phenomenon = row.get("phenomenon", "")
                if focus_phenomena is not None and phenomenon not in focus_phenomena:
                    continue
                full_prompt = row.get("prompt", "").strip()
                if full_prompt.rstrip().endswith("Answer:"):
                    full_prompt = full_prompt.rstrip()[: -len("Answer:")].rstrip()
                examples.append({
                    "full_prompt": full_prompt,
                    "correct_answer_idx": int(row.get("randomized_true_answer", 0)),
                    "phenomenon": phenomenon,
                    "item_id": row.get("item_id", ""),
                })
    return examples


# ── Prompt construction ──────────────────────────────────────────────────────

def make_messages(
    example: Dict,
    system_prompt: Optional[str] = None,
    user_prompt_prefix: Optional[str] = None,
) -> List[Dict]:
    """Build the chat-message list consumed by ``apply_chat_template``.

    Optional overrides are accepted for backwards compatibility with callers
    that customize the system prompt or prepend extra user-side instructions.
    """
    active_system_prompt = str(system_prompt or SYSTEM_PROMPT).strip()
    user_message = str(example["full_prompt"]).strip()
    prefix = str(user_prompt_prefix or "").strip()
    if prefix:
        user_message = f"{prefix}\n\n{user_message}"
    return [
        {"role": "system", "content": active_system_prompt},
        {"role": "user", "content": user_message},
    ]


# ── Answer extraction ────────────────────────────────────────────────────────

def extract_answer(text: str) -> Optional[str]:
    r"""Regex stack: try ``\boxed{}`` first, then common text patterns."""
    boxed_patterns = [
        r"\\boxed\{\s*(\d+)\s*\}",
        r"\$\\boxed\{\s*(\d+)\s*\}\$",
    ]
    for pattern in boxed_patterns:
        m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).strip()

    fallback_patterns = [
        r"Answer:\s*(\d+)",
        r"answer:\s*(\d+)",
        r"Final Answer:\s*(\d+)",
        r"final answer:\s*(\d+)",
        r"Answer:\s*(\d+)\)",
        r"answer:\s*(\d+)\)",
    ]
    for pattern in fallback_patterns:
        m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).strip()

    return None


# ── Pass-state classification ────────────────────────────────────────────────

def classify_pass_state(output_ids: List[int]) -> str:
    """Classify a Pass-1 output into one of three states.

    Returns:
      ``"finished"``      – model produced ``<|im_end|>``  (answer complete)
      ``"think_done"``    – model produced ``</think>``     (thinking done, no answer yet)
      ``"budget_forced"`` – neither stop token found        (hit thinking budget)
    """
    if IM_END_TOKEN_ID in output_ids:
        return "finished"
    if THINK_END_TOKEN_ID in output_ids:
        return "think_done"
    return "budget_forced"


def trim_trailing_pad_tokens(token_ids, pad_token_id: Optional[int]) -> List[int]:
    """Strip trailing pad tokens from a 1-D token list / tensor."""
    token_ids = token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)
    if pad_token_id is None:
        return token_ids
    end = len(token_ids)
    while end > 0 and token_ids[end - 1] == pad_token_id:
        end -= 1
    return token_ids[:end]


# ── Judge prompt (for self-judge or external judge) ──────────────────────────

JUDGE_EXTRACTION_PROMPT_TEMPLATE = (
    "Extract the answer number from the text below. "
    "The answer is a single integer (1, 2, 3, or 4). "
    "Output ONLY the integer, nothing else.\n\n"
    "Text:\n{text}\n\n"
    "Answer number:"
)
