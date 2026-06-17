from __future__ import annotations

import json
import random
import re
from collections import Counter
from dataclasses import dataclass

from .open_judge_eval import strip_code_fence

ALTPRAG_DATASET_NAME = "Huangtubaye233/AltPrag"
ALTPRAG_MAXIM_ORDER = ["None", "Relation", "Quality", "Quantity", "Manner"]
ALTPRAG_TASK_ORDER = ["candidate1", "candidate2"]
ALTPRAG_TASK_DISPLAY = {
    "candidate1": "Cand1",
    "candidate2": "Cand2",
}

ALTPRAG_EARLY_STOPPING_TEXT = (
    "\n\nConsidering the limited time by the user, I have to give the final answer directly now.\n"
    "</think>\n\n"
    "Write the full answer enclosed in \\boxed{...}.\n"
)

ALTPRAG_SYSTEM_PROMPT = (
    "Enclose that exact output sentence in \\boxed{...}.\n"
)


@dataclass
class AltPragExample:
    id: str
    split: str
    row_index: int
    task: str
    source_candidate: int
    context: str
    root: str
    candidate_sentence_1: str
    candidate_sentence_2: str
    gold_intention: str
    gold_maxim: str


@dataclass
class AltPragStats:
    dataset_name: str
    hf_split: str
    task_mode: str
    total_rows: int
    selected_examples: int
    task_counts: dict[str, int]
    maxim_counts: dict[str, int]


def _clean(text) -> str:
    return str(text or "").strip()


def _json_line(key: str, value: str) -> str:
    return f'"{key}": {json.dumps(str(value), ensure_ascii=False)}'


def normalize_altprag_maxim(value) -> str:
    raw = str(value).strip() if value is not None else ""
    if not raw or raw.lower() == "none" or raw == "<None>":
        return "None"
    title = raw.capitalize()
    if title in ALTPRAG_MAXIM_ORDER:
        return title
    return raw


def build_altprag_system_prompt() -> str:
    return str(ALTPRAG_SYSTEM_PROMPT)


def _render_input_block(*, context: str, root: str, candidate_sentence_1: str, candidate_sentence_2: str) -> str:
    return "\n".join(
        [
            "Input:",
            _json_line("context", context),
            _json_line("root", root),
            _json_line("candidate_sentence_1", candidate_sentence_1),
            _json_line("candidate_sentence_2", candidate_sentence_2),
        ]
    )


def build_altprag_user_prompt(example: AltPragExample) -> str:
    return (
        "You will see a conversation with four parts:\n"
        "- context: background information\n"
        "- root: the first sentence in the conversation\n"
        "- candidate_sentence_1: one possible reply\n"
        "- candidate_sentence_2: another possible reply\n\n"
        "Your task is to analyze candidate_sentence_1 in context. \n"
        "Specifically:\n"
        "1. Identify what this reply is doing or expressing (its intention).\n"
        "2. Consider why or when someone might choose candidate_sentence_1 instead of candidate_sentence_2.\n\n"
        "Please follow this structure in your output, and your output should ONLY include this sentence. Do not output anything else:\n"
        '"candidate_sentence_1_intention": "This response [flouts the maxim of ... / does not flout any maxim]. It aims to [describe what the speaker is doing or expressing in this reply]. A speaker might prefer this reply over candidate_sentence_2 because [reason in context: explain why this reply fits better in the specific situation]. [Add any further insight or nuance here.]"\n\n'
        "Now look at the following conversation:\n\n"
        f"{_render_input_block(**{k: getattr(example, k) for k in ['context', 'root', 'candidate_sentence_1', 'candidate_sentence_2']})}\n\n"
        "Your Output:\n"
        '"candidate_sentence_1_intention": '
    )


def normalize_altprag_answer(text: str | None) -> str | None:
    raw = strip_code_fence(str(text or "")).strip()
    if not raw:
        return None

    if raw.startswith("{") and raw.endswith("}"):
        try:
            obj = json.loads(raw)
            candidate = obj.get("candidate_sentence_1_intention")
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        except Exception:
            pass

    if raw.startswith('"candidate_sentence_1_intention"') or raw.startswith("candidate_sentence_1_intention"):
        wrapped = raw if raw.startswith("{") else "{" + raw + "}"
        try:
            obj = json.loads(wrapped)
            candidate = obj.get("candidate_sentence_1_intention")
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        except Exception:
            pass

    match = re.search(r'"?candidate_sentence_1_intention"?\s*:\s*', raw, flags=re.IGNORECASE)
    if match:
        raw = raw[match.end() :].strip()

    raw = raw.strip().rstrip("}").strip()
    if len(raw) >= 2 and raw[0] == raw[-1] == '"':
        raw = raw[1:-1]
    raw = raw.replace('\\"', '"').strip()
    return raw or None


def extract_altprag_maxim(text: str | None) -> str | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    if re.search(r"does not flout any maxim", raw, flags=re.IGNORECASE):
        return "None"
    match = re.search(r"flouts the maxim of\s+(relation|quality|quantity|manner)", raw, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).capitalize()


def load_altprag_examples(
    dataset_name: str = ALTPRAG_DATASET_NAME,
    *,
    hf_split: str = "test",
    task_mode: str = "both",
    max_examples: int = 0,
    subset_seed: int = 42,
) -> tuple[list[AltPragExample], AltPragStats]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The 'datasets' package is required for AltPrag evaluation. Install it in the active environment first."
        ) from exc

    dataset = load_dataset(dataset_name)
    if hf_split == "all":
        split_names = list(dataset.keys())
    elif hf_split in dataset:
        split_names = [hf_split]
    else:
        available = sorted(dataset.keys())
        raise ValueError(f"Requested hf_split={hf_split!r} not found in {dataset_name}. Available: {available}")

    normalized_task_mode = str(task_mode or "both").strip().lower()
    if normalized_task_mode not in {"candidate1", "candidate2", "both"}:
        raise ValueError(f"Unsupported task_mode={task_mode!r}; expected candidate1, candidate2, or both")

    examples: list[AltPragExample] = []
    task_counts: Counter[str] = Counter()
    maxim_counts: Counter[str] = Counter()
    total_rows = 0

    for split_name in split_names:
        split_ds = dataset[split_name]
        total_rows += len(split_ds)
        for row_index, row in enumerate(split_ds):
            context = _clean(row.get("context"))
            root = _clean(row.get("root"))
            cand1 = _clean(row.get("candidate_sentence_1"))
            cand2 = _clean(row.get("candidate_sentence_2"))
            gold1 = _clean(row.get("candidate_sentence_1_intention"))
            gold2 = _clean(row.get("candidate_sentence_2_intention"))
            gm1 = normalize_altprag_maxim(row.get("human_annotation_sentence_1_GM"))
            gm2 = normalize_altprag_maxim(row.get("human_annotation_sentence_2_GM"))
            if not context or not root or not cand1 or not cand2:
                continue

            if normalized_task_mode in {"candidate1", "both"} and gold1:
                ex = AltPragExample(
                    id=f"{split_name}:{row_index}:candidate1",
                    split=str(split_name),
                    row_index=int(row_index),
                    task="candidate1",
                    source_candidate=1,
                    context=context,
                    root=root,
                    candidate_sentence_1=cand1,
                    candidate_sentence_2=cand2,
                    gold_intention=gold1,
                    gold_maxim=gm1,
                )
                examples.append(ex)
                task_counts[ex.task] += 1
                maxim_counts[ex.gold_maxim] += 1

            if normalized_task_mode in {"candidate2", "both"} and gold2:
                ex = AltPragExample(
                    id=f"{split_name}:{row_index}:candidate2",
                    split=str(split_name),
                    row_index=int(row_index),
                    task="candidate2",
                    source_candidate=2,
                    context=context,
                    root=root,
                    candidate_sentence_1=cand2,
                    candidate_sentence_2=cand1,
                    gold_intention=gold2,
                    gold_maxim=gm2,
                )
                examples.append(ex)
                task_counts[ex.task] += 1
                maxim_counts[ex.gold_maxim] += 1

    if max_examples and max_examples > 0 and len(examples) > max_examples:
        rng = random.Random(int(subset_seed))
        examples = rng.sample(examples, int(max_examples))
        task_counts = Counter(ex.task for ex in examples)
        maxim_counts = Counter(ex.gold_maxim for ex in examples)

    stats = AltPragStats(
        dataset_name=str(dataset_name),
        hf_split=str(hf_split),
        task_mode=str(normalized_task_mode),
        total_rows=int(total_rows),
        selected_examples=len(examples),
        task_counts={name: int(task_counts.get(name, 0)) for name in ALTPRAG_TASK_ORDER},
        maxim_counts={name: int(maxim_counts.get(name, 0)) for name in ALTPRAG_MAXIM_ORDER},
    )
    return examples, stats
