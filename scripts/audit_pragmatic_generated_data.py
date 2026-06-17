#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
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
from src.utils.io import set_seed
from src.utils.thinking_tags import ALL_CLOSE_TAGS, normalize_early_stopping_text, strip_prefixed_thinking

QA_SYSTEM_PROMPT = (
    "You are a dataset auditor for Pragmatic QA.\n"
    "Judge whether the item is high-quality for its SECTION.\n"
    "Answer yes for high-quality, no for low-quality."
)

SECTION_DEFINITIONS: dict[str, str] = {
    "Context and Deixis": (
        "Resolve meaning that depends on context by identifying the intended referent of "
        "underspecified or deictic expressions from the dialogue and current perspective."
    ),
    "Implicature and Presupposition": (
        "Recover non-explicit meaning by inferring what is implied under cooperative reasoning, "
        "and by identifying background assumptions required for an utterance to be interpretable."
    ),
    "Speech Acts and Intent Recognition": (
        "Infer what action an utterance performs in context, including cases where the intended "
        "act differs from the literal form."
    ),
    "Discourse and Coherence": (
        "Maintain a coherent interpretation across turns by linking utterances via discourse "
        "relations and tracking what has been established, updated, or answered."
    ),
    "Social Pragmatics": (
        "Interpret and respond appropriately when meaning is shaped by social context such as "
        "roles, power, culture, and interactional norms."
    ),
    "Metaphor": (
        "Interpret figurative language by mapping a non-literal description to the situation to "
        "recover the intended evaluation or action guidance, rather than a literal reading."
    ),
}

SECTION_ALIASES: dict[str, str] = {
    "context and deixis": "Context and Deixis",
    "implicature and presupposition": "Implicature and Presupposition",
    "speech acts and intent recognition": "Speech Acts and Intent Recognition",
    "discourse and coherence": "Discourse and Coherence",
    "social pragmatics": "Social Pragmatics",
    "metaphor": "Metaphor",
}

# Per-section calibration examples are hardcoded:
# - 2 good examples from pragmatic_mcq_examples.jsonl / pragmatic_examples.jsonl
# - 2 bad contrastive examples authored for calibration
SECTION_FEW_SHOTS: dict[str, dict[str, list[dict[str, str]]]] = {
    "Context and Deixis": {
        "qa": [
            {
                "answer": "yes",
                "content": "At 11:58 pm on Tuesday, Erin messages: \"Let’s meet this Friday.\" At 12:03 am (now Wednesday), Erin follows up: \"Sorry, I am little busy this week.\"",
                "question": "When does Erin want to meet?",
                "gold": "Next Friday.",
                "why": "High-quality in-domain QA example from pragmatic_mcq_examples.jsonl."
            },
            {
                "answer": "yes",
                "content": "You are on a video call. You and your coworker can see the shared screen; the client can also see it. Your coworker types in the shared document: \"This is a terrible idea lol.\" Then they message you privately: \"Can you move that somewhere safer?\"",
                "question": "What is the safest, most context-appropriate thing for you to do next?",
                "gold": "Move it off the shared doc to an internal-only place.",
                "why": "High-quality in-domain QA example from pragmatic_mcq_examples.jsonl."
            },
            {
                "answer": "no",
                "content": "Dialogue:\nA: \"I’m standing by the fountain. I’ll meet you here at 3.\"\nB: \"Got it.\"",
                "question": "What is B’s direct meaning?",
                "gold": "I’ll meet you at the fountain at 3.",
                "why": "Too explicit; little context-dependent resolution required."
            },
            {
                "answer": "no",
                "content": "Dialogue:\nA: \"Pick up the folder on the table. It’s the only blue one.\"\nB: \"Okay, the blue folder.\"",
                "question": "What is B’s direct meaning?",
                "gold": "I’ll pick up the blue folder.",
                "why": "Answer leakage from explicit cue."
            }
        ]
    },
    "Implicature and Presupposition": {
        "qa": [
            {
                "answer": "yes",
                "content": "A manager asks: \"Did the whole team sign off on the release?\" You reply: \"A few people signed off.\"",
                "question": "What is the safest conclusion the manager will draw from your reply?",
                "gold": "Not everyone signed off.",
                "why": "High-quality in-domain QA example from pragmatic_mcq_examples.jsonl."
            },
            {
                "answer": "yes",
                "content": "Donny is meeting a woman this evening.",
                "question": "Is Donny meeting his mother?",
                "gold": "No.",
                "why": "High-quality in-domain QA example from pragmatic_mcq_examples.jsonl."
            },
            {
                "answer": "no",
                "content": "Dialogue:\nA: \"Are you coming to the team dinner?\"\nB: \"No, I’m not.\"",
                "question": "What is B’s direct meaning?",
                "gold": "No, I’m not coming.",
                "why": "Literal explicit answer, no implicature needed."
            },
            {
                "answer": "no",
                "content": "Dialogue:\nA: \"Did you stop by the office today?\"\nB: \"I didn’t go to the office today.\"",
                "question": "What is B’s direct meaning?",
                "gold": "No, I didn’t stop by the office today.",
                "why": "Direct paraphrase without hidden meaning."
            }
        ]
    },
    "Speech Acts and Intent Recognition": {
        "qa": [
            {
                "answer": "yes",
                "content": "Kai and Jack are in a meeting room. The air feels stuffy. Windows are closed. Kai asked \"Is there anything you might need before we start the meeting?\" and Jack replied \"It’s stuffy in here.\"",
                "question": "What is Kai's next move?",
                "gold": "Open the window.",
                "why": "High-quality in-domain QA example from pragmatic_mcq_examples.jsonl."
            },
            {
                "answer": "yes",
                "content": "Luis and Danielle studying, and Luis tuned music on. Danielle says \"It is little hard to hear what my online lecture saying.",
                "question": "Is Danielle requesting something?",
                "gold": "Yes. She is indirectly asking Luis to turn down the music.",
                "why": "High-quality in-domain QA example from pragmatic_mcq_examples.jsonl."
            },
            {
                "answer": "no",
                "content": "Dialogue:\nA: \"It’s noisy.\"\nB: \"Close the door.\"",
                "question": "What is B’s direct meaning?",
                "gold": "Please close the door.",
                "why": "Speech act is already explicit imperative."
            },
            {
                "answer": "no",
                "content": "Dialogue:\nA: \"Should we start the meeting?\"\nB: \"It starts at 2.\"",
                "question": "What is B’s direct meaning?",
                "gold": "We should start at 2.",
                "why": "Mostly factual timing, weak intent inference."
            }
        ]
    },
    "Discourse and Coherence": {
        "qa": [
            {
                "answer": "yes",
                "content": "The migration runbook says, \"Disable the old endpoint only after Client A is confirmed on the new route.\" At 11:06, the ops channel gets a message: \"Client A should be fine now.\" At 11:08, the old endpoint is disabled. At 11:20, Client A reports that their traffic was still pointing to the old address.",
                "question": "What assumption caused the mistake?",
                "gold": "A guess was treated as confirmation.",
                "why": "High-quality in-domain QA example from pragmatic_mcq_examples.jsonl."
            },
            {
                "answer": "yes",
                "content": "The grant summary opens with \"No participant data left the lab.\" Appendix C lists an external vendor under \"transcription support.\" In the ethics response, the PI writes, \"Only de-identified audio was shared externally.\" The reviewer circles the opening sentence.",
                "question": "Why did the reviewer circle the opening sentence?",
                "gold": "It overstates the claim.",
                "why": "High-quality in-domain QA example from pragmatic_mcq_examples.jsonl."
            },
            {
                "answer": "no",
                "content": "Dialogue:\nA: \"Did you submit the report?\"\nB: \"Yes, I submitted it.\"",
                "question": "What is B’s direct meaning?",
                "gold": "Yes, I submitted it.",
                "why": "Literal answer, no discourse bridging needed."
            },
            {
                "answer": "no",
                "content": "Dialogue:\nA: \"Any update on the deployment?\"\nB: \"We merged the PR.\"\nA: \"Did the pipeline pass?\"\nB: \"It’s deployed.\"",
                "question": "What is B’s direct meaning?",
                "gold": "The deployment is complete.",
                "why": "Final line alone gives answer; low coherence demand."
            }
        ]
    },
    "Social Pragmatics": {
        "qa": [
            {
                "answer": "yes",
                "content": "A new intern says to the CEO in a hallway: \"Hey, can you approve my vacation request real quick?\" The CEO pauses and replies: \"Send it through the usual channel.\"",
                "question": "What is the CEO most likely communicating beyond the literal instruction?",
                "gold": "Follow the formal process.",
                "why": "High-quality in-domain QA example from pragmatic_mcq_examples.jsonl."
            },
            {
                "answer": "yes",
                "content": "You’re invited to a colleague’s wedding. The invitation says \"Black tie optional.\" Your friend texts: \"I’m thinking jeans and sneakers.\" You reply: \"it is wedding.\"",
                "question": "What are you most likely trying to get your friend to do?",
                "gold": "Dress more formally.",
                "why": "High-quality in-domain QA example from pragmatic_mcq_examples.jsonl."
            },
            {
                "answer": "no",
                "content": "Dialogue:\nA: \"Can you send me the file?\"\nB: \"Sure.\"",
                "question": "What is B’s direct meaning?",
                "gold": "Sure, I’ll send you the file.",
                "why": "Weak social-norm signal; generic reply."
            },
            {
                "answer": "no",
                "content": "Dialogue:\nA: \"I’ll take this call on speaker.\"\nB: \"In public spaces, speakerphone use violates social etiquette.\"",
                "question": "What is B’s direct meaning?",
                "gold": "Don’t use speaker here.",
                "why": "Unnatural lecture style, low realism."
            }
        ]
    },
    "Metaphor": {
        "qa": [
            {
                "answer": "yes",
                "content": "After a week of late-night debugging, the lead engineer said, \"We fixed the leak by painting the ceiling.\"",
                "question": "What is the lead engineer most likely criticizing about the team’s fix?",
                "gold": "A superficial patch, not the root cause.",
                "why": "High-quality in-domain QA example from pragmatic_mcq_examples.jsonl."
            },
            {
                "answer": "yes",
                "content": "During a design review, a senior designer looked at a complicated interface and said, \"This is a Swiss Army knife that only needs to cut bread.\"",
                "question": "What is the designer implying should change?",
                "gold": "It’s over-engineered; simplify to essentials.",
                "why": "High-quality in-domain QA example from pragmatic_mcq_examples.jsonl."
            },
            {
                "answer": "no",
                "content": "Dialogue:\nA: \"How stable is the system?\"\nB: \"It’s held together with duct tape, meaning it’s unreliable.\"",
                "question": "What is B’s direct meaning?",
                "gold": "It’s unreliable.",
                "why": "Metaphor is already glossed explicitly."
            },
            {
                "answer": "no",
                "content": "Dialogue:\nA: \"How’s the plan?\"\nB: \"It’s a train wreck, honestly.\"",
                "question": "What is B’s direct meaning?",
                "gold": "The plan is very bad.",
                "why": "Too transparent; little metaphor mapping needed."
            }
        ]
    }
}

QA_USER_PROMPT_TEMPLATE = """Is this pragmatic QA example high-quality?

Criteria:
Pragmatic dependency: the gold answer requires pragmatic interpretation, not just literal reading.
Question correctness: the question itself is not incorrect or ambiguous.
Gold answer: the gold answer is correct and the uniquely best-supported by given context.

Few-shot examples for this same SECTION:
{FEW_SHOT_EXAMPLES}

Example to judge:
SECTION: {SECTION}
SCENARIO: {CONTENT}
QUESTION: {QUESTION}
GOLD ANSWER: {GOLD_ANSWER}

Answer just yes or no with no other output.
Final answer:"""

# Difficulty: the item is challenging but solvable from the provided scenario.
# Section fidelity: the question genuinely tests the named pragmatic phenomenon.
# Few-shot examples for this same SECTION:
# {FEW_SHOT_EXAMPLES}


# Section definition:
# {SECTION_DEFINITION}


@dataclass
class AuditResult:
    decision: str
    prob_yes: float
    log_odds: float
    log_p_yes: float
    log_p_no: float
    model_answer: str
    scoring_mode: str
    score_valid: bool
    raw_output: str


def _score_field_value(audit: AuditResult, score_field: str) -> float:
    if score_field in {"log_odds", "keep_minus_drop"}:
        return float(audit.log_odds)
    if score_field == "drop_minus_keep":
        return float(-audit.log_odds)
    if score_field in {"prob_yes", "prob_keep"}:
        return float(audit.prob_yes)
    if score_field == "prob_drop":
        return float(1.0 - audit.prob_yes)
    raise ValueError(f"Unsupported score_field: {score_field}")


def _score_field_higher_is_better(score_field: str) -> bool:
    return score_field in {"log_odds", "prob_yes", "keep_minus_drop", "prob_keep"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit already-generated pragmatic QA datasets and filter to KEEP/DROP "
            "using a model that supports response_logprob."
        )
    )
    parser.add_argument("--config_path", type=Path, default=Path("configs/default.yaml"))

    parser.add_argument("--qa_input_path", type=Path, default=None)
    parser.add_argument("--qa_output_path", type=Path, default=None)
    parser.add_argument(
        "--summary_output_path",
        type=Path,
        default=ROOT_PATH / "data/audit/pragmatic_audit_summary.json",
    )
    parser.add_argument("--attach_audit_to_kept", action="store_true")
    parser.add_argument("--max_items_per_dataset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--filter_strategy",
        type=str,
        default="bottom_percent",
        choices=["threshold", "bottom_percent"],
        help="Filter method: fixed score threshold, or drop bottom X percent by score.",
    )
    parser.add_argument(
        "--score_field",
        type=str,
        default="drop_minus_keep",
        choices=[
            "log_odds",
            "prob_yes",
            "keep_minus_drop",
            "drop_minus_keep",
            "prob_keep",
            "prob_drop",
        ],
        help="Score used for filtering.",
    )
    parser.add_argument(
        "--quality_threshold",
        type=float,
        default=0.0,
        help="Keep items with score >= threshold when filter_strategy=threshold.",
    )
    parser.add_argument(
        "--drop_bottom_percent",
        type=float,
        default=50.0,
        help="Drop this percent of lowest-scored items when filter_strategy=bottom_percent.",
    )
    parser.add_argument(
        "--bottom_percent_scope",
        type=str,
        default="global",
        choices=["global", "per_section"],
        help="When filter_strategy=bottom_percent, drop globally or independently within each section.",
    )

    parser.add_argument("--model_role", type=str, default="")
    parser.add_argument("--model_name_or_path", type=str, default="")
    parser.add_argument("--dtype", type=str, default="")
    parser.add_argument("--device_map", type=str, default="")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument(
        "--gen_backend",
        type=str,
        default="vllm_local",
        choices=["hf", "vllm_api", "vllm_local"],
        help="Generation backend used for auditing.",
    )
    parser.add_argument("--vllm_base_url", type=str, default="http://localhost:8000")
    parser.add_argument("--vllm_model_name", type=str, default="")
    parser.add_argument("--vllm_api_key", type=str, default="EMPTY")
    parser.add_argument("--vllm_timeout", type=float, default=180.0)
    parser.add_argument("--vllm_max_retries", type=int, default=3)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--vllm_max_model_len", type=int, default=0)

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument(
        "--enable_thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--thinking_budget_tokens", type=int, default=0)
    parser.add_argument("--answer_budget_tokens", type=int, default=0)
    parser.add_argument(
        "--thinking_early_stopping_text",
        type=str,
        default=(
            "\n\nConsidering the limited time by the user, I have to give the solution "
            "based on the thinking directly now.\n</think>\n\n"
        ),
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


def _prepare_model_context(
    cfg_raw: dict[str, Any],
    args: argparse.Namespace,
) -> Any:
    models_cfg = cfg_raw.get("models", {}) if isinstance(cfg_raw, dict) else {}
    role = str(args.model_role or "speaker").strip() or "speaker"
    model_cfg = models_cfg.get(role) or models_cfg.get("speaker")
    if model_cfg is None:
        raise ValueError(f"Model config not found for role={role} and fallback=speaker")
    model_cfg = dict(model_cfg)

    if args.model_name_or_path:
        model_cfg["name"] = args.model_name_or_path
    if args.dtype:
        model_cfg["dtype"] = args.dtype
    if args.device_map:
        model_cfg["device_map"] = args.device_map
    if args.load_in_4bit:
        model_cfg["load_in_4bit"] = True
    if args.load_in_8bit:
        model_cfg["load_in_8bit"] = True

    if args.gen_backend == "vllm_api":
        from src.utils.vllm_api import VLLMApiModel

        vllm_model_name = (
            str(args.vllm_model_name).strip()
            or str(args.model_name_or_path).strip()
            or str(model_cfg.get("name", "")).strip()
        )
        if not vllm_model_name:
            raise ValueError("vLLM backend requires model name (--vllm_model_name or --model_name_or_path)")
        return VLLMApiModel(
            vllm_model_name,
            base_url=args.vllm_base_url,
            api_key=args.vllm_api_key,
            timeout=float(args.vllm_timeout),
            max_retries=int(args.vllm_max_retries),
        )
    if args.gen_backend == "vllm_local":
        from src.utils.vllm_local import VLLMLocalModel

        vllm_model_name = (
            str(args.vllm_model_name).strip()
            or str(args.model_name_or_path).strip()
            or str(model_cfg.get("name", "")).strip()
        )
        if not vllm_model_name:
            raise ValueError("Local vLLM backend requires model name (--vllm_model_name or --model_name_or_path)")
        vllm_max_model_len = int(args.vllm_max_model_len or 0)
        return VLLMLocalModel(
            vllm_model_name,
            dtype=str(model_cfg.get("dtype", "")).strip() or None,
            gpu_memory_utilization=float(args.vllm_gpu_memory_utilization),
            tensor_parallel_size=int(args.vllm_tensor_parallel_size),
            max_model_len=vllm_max_model_len if vllm_max_model_len > 0 else None,
            seed=int(cfg_raw.get("seed", 42)),
        )
    return _build_model(model_cfg, role)


def _set_all_seeds(seed: int) -> None:
    set_seed(seed)
    try:
        import random

        random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _strip_thinking(text: str) -> str:
    return strip_prefixed_thinking(text)


def _trim_thinking_prefix_through_close(
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


def _parse_jsonish(text: str) -> dict[str, Any] | None:
    cleaned = _strip_code_fence(_strip_thinking(text))
    if not cleaned:
        return None
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    span = _find_last_json_span(cleaned)
    if span is not None:
        start, end = span
        try:
            parsed = json.loads(cleaned[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return None
    return None


def _coerce_int(value: Any, default: int = 0) -> int:
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


def _clamp(v: int, lo: int, hi: int) -> int:
    return min(max(v, lo), hi)


def _ensure_list_str(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    s = str(value).strip()
    return [s] if s else []


def _normalize_hard_filters(values: list[str]) -> list[str]:
    neutral = {"none", "no", "null", "n/a", "na", "[]", "-"}
    out: list[str] = []
    for raw in values:
        s = str(raw).strip()
        if not s:
            continue
        if s.lower() in neutral:
            continue
        out.append(s)
    return out


def _canonical_section(section: str) -> str:
    key = re.sub(r"[^a-z0-9]+", " ", str(section or "").strip().lower()).strip()
    return SECTION_ALIASES.get(key, str(section or "").strip())

def _section_definition_for_prompt(section: str) -> str:
    canonical = _canonical_section(section)
    return SECTION_DEFINITIONS.get(
        canonical,
        "Interpret the item strictly by the provided SECTION label and require section-specific pragmatic reasoning.",
    )


def _few_shot_block_for_prompt(section: str, *, task_type: str) -> str:
    canonical = _canonical_section(section)
    section_pack = SECTION_FEW_SHOTS.get(canonical, {})
    shots = section_pack.get("qa", [])

    if not shots:
        return "No section-specific few-shot examples available."

    lines: list[str] = []
    for i, shot in enumerate(shots, start=1):
        answer_raw = str(shot.get("answer", "")).strip().lower()
        answer = answer_raw if answer_raw in {"yes", "no"} else "no"
        lines.append(f"Example {i}")
        lines.append(f"SCENARIO/DIALOGUE: {shot['content']}")
        if task_type == "qa":
            lines.append(f"QUESTION: {shot['question']}")
            lines.append(f"GOLD: {shot['gold']}")
            if answer == "no":
                lines.append(f"RATIONALE: {shot.get('why', 'Calibration example.')}")
        else:
            lines.append(f"GOLD: {shot['gold']}")
            if answer == "no":
                lines.append(f"RATIONALE: {shot.get('why', 'Calibration example.')}")
        lines.append(f"Final answer: {answer}")
        lines.append("")
    return "\n".join(lines).strip()


def _render_user_prompt(
    prompt_template: str,
    section: str,
    content: str,
    question: str,
    gold_answer: str,
    task_type: str,
) -> str:
    prompt = prompt_template
    prompt = prompt.replace("{SECTION}", section)
    prompt = prompt.replace("{CONTENT}", content)
    prompt = prompt.replace("{QUESTION}", question)
    prompt = prompt.replace("{GOLD_ANSWER}", gold_answer)
    prompt = prompt.replace("{SECTION_DEFINITION}", _section_definition_for_prompt(section))
    prompt = prompt.replace(
        "{FEW_SHOT_EXAMPLES}",
        _few_shot_block_for_prompt(section, task_type=task_type),
    )
    return prompt


def _to_float_scalar(value: Any) -> float:
    if value is None:
        return float("-inf")
    try:
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)
    except Exception:
        return float("-inf")


def _safe_sigmoid(x: float) -> float:
    # Stable sigmoid for large-magnitude log-odds.
    x = max(min(x, 60.0), -60.0)
    return 1.0 / (1.0 + math.exp(-x))


def _score_yes_no(
    *,
    model: Any,
    system_prompt: str,
    user_prompt: str,
    args: argparse.Namespace,
) -> AuditResult:
    if not hasattr(model, "response_logprob") or not callable(getattr(model, "response_logprob")):
        raise RuntimeError(
            "Audit requires response_logprob support. Refusing generation fallback by design."
        )

    assistant_prefix = ""
    if bool(args.enable_thinking):
        generation = model.generate(
            user_prompt,
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            system_prompt=system_prompt,
            enable_thinking=True,
        )
        assistant_prefix = _trim_thinking_prefix_through_close(
            generation.text,
            early_stopping_text=str(args.thinking_early_stopping_text),
            model_name_or_path=str(getattr(model, "name", "")),
        )

    try:
        lp_yes = _to_float_scalar(
            model.response_logprob(
                user_prompt,
                assistant_prefix + " yes",
                system_prompt=system_prompt,
                enable_thinking=bool(args.enable_thinking),
            )
        )
        lp_no = _to_float_scalar(
            model.response_logprob(
                user_prompt,
                assistant_prefix + " no",
                system_prompt=system_prompt,
                enable_thinking=bool(args.enable_thinking),
            )
        )
    except Exception as exc:
        raise RuntimeError(
            f"Audit score computation failed for response_logprob: {type(exc).__name__}: {exc}"
        ) from exc

    if not (math.isfinite(lp_yes) and math.isfinite(lp_no)):
        raise RuntimeError(
            f"Audit score computation returned non-finite log-probabilities: lp_yes={lp_yes}, lp_no={lp_no}"
        )

    log_odds = lp_yes - lp_no
    prob_yes = _safe_sigmoid(log_odds)
    model_answer = "yes" if prob_yes >= 0.5 else "no"
    return AuditResult(
        decision="",
        prob_yes=prob_yes,
        log_odds=log_odds,
        log_p_yes=lp_yes,
        log_p_no=lp_no,
        model_answer=model_answer,
        scoring_mode="logprob",
        score_valid=True,
        raw_output="",
    )


def _validate_audit_model_support(model: Any, *, backend_name: str, model_name: str) -> None:
    if hasattr(model, "response_logprob") and callable(getattr(model, "response_logprob")):
        return
    raise RuntimeError(
        "Audit requires response_logprob support and will not fall back to generation. "
        f"backend={backend_name!r} model={model_name!r}"
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {path}. Use --overwrite.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _default_output_path(input_path: Path, suffix: str) -> Path:
    return input_path.with_name(f"{input_path.stem}{suffix}{input_path.suffix}")


def _format_turns(turns: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        speaker = str(turn.get("speaker", "")).strip()
        utterance = str(turn.get("utterance", "")).strip()
        if not utterance:
            continue
        if speaker:
            lines.append(f"{speaker}: {utterance}")
        else:
            lines.append(utterance)
    return "\n".join(lines)


def _qa_fields(row: dict[str, Any]) -> tuple[str, str, str, str]:
    section = str(row.get("section", "")).strip()
    content = str(row.get("content", "")).strip()
    question = str(row.get("question", "")).strip()
    answer = str(row.get("answer", "")).strip()
    return section, content, question, answer


def _audit_dataset(
    *,
    dataset_name: str,
    rows: list[dict[str, Any]],
    max_items: int,
    model: Any,
    args: argparse.Namespace,
    field_builder,
    system_prompt: str,
    user_prompt_template: str,
    attach_audit_to_kept: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if max_items > 0:
        rows = rows[:max_items]

    section_stats: dict[str, dict[str, int]] = {}
    scored_items: list[dict[str, Any]] = []

    iterator: Any = enumerate(rows, start=1)
    if tqdm is not None:
        iterator = enumerate(tqdm(rows, desc=f"audit:{dataset_name}", unit="item"), start=1)

    for idx, row in iterator:
        section, content, question, gold_answer = field_builder(row)
        sec_key = section or "unknown"
        sec_bucket = section_stats.setdefault(sec_key, {"total": 0, "keep": 0, "drop": 0})
        sec_bucket["total"] += 1

        if dataset_name == "qa":
            missing_required = (not section or not content or not question or not gold_answer)
        else:
            missing_required = (not section or not content or not gold_answer)

        if missing_required:
            audit = AuditResult(
                decision="DROP",
                prob_yes=0.0,
                log_odds=-20.0,
                log_p_yes=-20.0,
                log_p_no=0.0,
                model_answer="invalid",
                scoring_mode="missing_fields",
                score_valid=False,
                raw_output="",
            )
        else:
            prompt = _render_user_prompt(
                user_prompt_template,
                section,
                content,
                question,
                gold_answer,
                task_type=dataset_name,
            )
            audit = _score_yes_no(
                model=model,
                system_prompt=system_prompt,
                user_prompt=prompt,
                args=args,
            )

        score_value = _score_field_value(audit, str(args.score_field))
        scored_items.append(
            {
                "index": idx,
                "section": sec_key,
                "row": row,
                "audit": audit,
                "score_value": float(score_value),
                "score_valid": bool(audit.score_valid),
            }
        )

    # Decide keep/drop from scores.
    keep_index_set: set[int] = set()
    hard_drop_index_set = {
        int(item["index"])
        for item in scored_items
        if str(item["audit"].scoring_mode) == "missing_fields"
    }
    eligible_items = [
        item for item in scored_items if int(item["index"]) not in hard_drop_index_set
    ]
    scorable_items = [item for item in eligible_items if bool(item["score_valid"])]
    unscored_items = [item for item in eligible_items if not bool(item["score_valid"])]

    n_total = len(scored_items)
    n_eligible = len(eligible_items)
    n_scorable = len(scorable_items)
    threshold_used: float | None = None
    threshold_used_by_section: dict[str, float] = {}

    if n_total == 0:
        summary = {
            "dataset": dataset_name,
            "total": 0,
            "eligible_total": 0,
            "scorable_total": 0,
            "unscored_total": 0,
            "hard_drop_total": 0,
            "keep": 0,
            "drop": 0,
            "keep_rate": 0.0,
            "avg_prob_yes": 0.0,
            "avg_prob_keep": 0.0,
            "avg_prob_drop": 0.0,
            "avg_log_odds": 0.0,
            "avg_keep_minus_drop": 0.0,
            "avg_drop_minus_keep": 0.0,
            "filter_strategy": args.filter_strategy,
            "score_field": args.score_field,
            "quality_threshold": float(args.quality_threshold),
            "drop_bottom_percent": float(args.drop_bottom_percent),
            "bottom_percent_scope": str(args.bottom_percent_scope),
            "effective_threshold": None,
            "effective_threshold_by_section": None,
            "section_stats": {},
        }
        return [], summary

    if args.filter_strategy == "threshold":
        threshold_used = float(args.quality_threshold)
        higher_is_better = _score_field_higher_is_better(str(args.score_field))
        for item in scorable_items:
            score_value = float(item["score_value"])
            if higher_is_better:
                if score_value >= threshold_used:
                    keep_index_set.add(int(item["index"]))
            else:
                if score_value <= threshold_used:
                    keep_index_set.add(int(item["index"]))
        keep_index_set.update(int(item["index"]) for item in unscored_items)
    else:
        drop_pct = max(0.0, min(100.0, float(args.drop_bottom_percent)))
        if n_scorable <= 0:
            raise RuntimeError(
                "No scorable items were produced by the auditor model; refusing bottom-percent filtering "
                "because it would cause arbitrary drops. Check vLLM/model output and prompts."
            )
        higher_is_better = _score_field_higher_is_better(str(args.score_field))
        drop_index_set: set[int] = set()
        if str(args.bottom_percent_scope) == "per_section":
            scorable_by_section: dict[str, list[dict[str, Any]]] = {}
            for item in scorable_items:
                sec_key = str(item["section"] or "unknown")
                scorable_by_section.setdefault(sec_key, []).append(item)

            for sec_key, section_items in scorable_by_section.items():
                section_sorted = sorted(
                    section_items,
                    key=lambda x: (x["score_value"], int(x["index"])),
                )
                section_drop_n = int((drop_pct / 100.0) * len(section_sorted))
                if higher_is_better:
                    dropped = section_sorted[:section_drop_n]
                    drop_index_set.update(int(x["index"]) for x in dropped)
                    if section_drop_n > 0:
                        threshold_used_by_section[sec_key] = float(
                            dropped[-1]["score_value"]
                        )
                else:
                    dropped = section_sorted[-section_drop_n:] if section_drop_n > 0 else []
                    drop_index_set.update(int(x["index"]) for x in dropped)
                    if section_drop_n > 0:
                        threshold_used_by_section[sec_key] = float(
                            dropped[0]["score_value"]
                        )
        else:
            drop_n = int((drop_pct / 100.0) * n_scorable)
            sorted_items = sorted(
                scorable_items,
                key=lambda x: (x["score_value"], int(x["index"])),
            )
            if higher_is_better:
                dropped = sorted_items[:drop_n]
                drop_index_set = {int(x["index"]) for x in dropped}
                if drop_n > 0:
                    threshold_used = float(dropped[-1]["score_value"])
            else:
                dropped = sorted_items[-drop_n:] if drop_n > 0 else []
                drop_index_set = {int(x["index"]) for x in dropped}
                if drop_n > 0:
                    threshold_used = float(dropped[0]["score_value"])

        keep_index_set = {
            int(x["index"]) for x in scorable_items if int(x["index"]) not in drop_index_set
        }
        keep_index_set.update(int(item["index"]) for item in unscored_items)

    kept: list[dict[str, Any]] = []
    drop_count = 0
    prob_yes_sum = 0.0
    log_odds_sum = 0.0

    for item in scored_items:
        idx = int(item["index"])
        sec_key = str(item["section"])
        row = item["row"]
        audit: AuditResult = item["audit"]
        prob_yes_sum += float(audit.prob_yes)
        log_odds_sum += float(audit.log_odds)

        decision = "KEEP" if idx in keep_index_set else "DROP"
        audit_payload = {
            "decision": decision,
            "prob_yes": float(audit.prob_yes),
            "prob_keep": float(audit.prob_yes),
            "prob_drop": float(1.0 - audit.prob_yes),
            "log_odds": float(audit.log_odds),
            "keep_minus_drop": float(audit.log_odds),
            "drop_minus_keep": float(-audit.log_odds),
            "log_p_yes": float(audit.log_p_yes),
            "log_p_no": float(audit.log_p_no),
            "log_p_keep": float(audit.log_p_yes),
            "log_p_drop": float(audit.log_p_no),
            "model_answer": audit.model_answer,
            "legacy_model_answer": "KEEP" if float(audit.prob_yes) >= 0.5 else "DROP",
            "scoring_mode": audit.scoring_mode,
            "score_valid": bool(audit.score_valid),
            "score_field": args.score_field,
            "score_value": float(item["score_value"]),
            "raw_output": audit.raw_output,
        }

        if decision == "KEEP":
            section_stats[sec_key]["keep"] += 1
            if attach_audit_to_kept:
                row_out = dict(row)
                row_out["audit"] = audit_payload
                kept.append(row_out)
            else:
                kept.append(row)
            continue

        section_stats[sec_key]["drop"] += 1
        drop_count += 1

    total = len(rows)
    keep_count = len(kept)
    summary = {
        "dataset": dataset_name,
        "total": total,
        "eligible_total": n_eligible,
        "scorable_total": n_scorable,
        "unscored_total": len(unscored_items),
        "hard_drop_total": len(hard_drop_index_set),
        "keep": keep_count,
        "drop": drop_count,
        "keep_rate": (keep_count / total) if total else 0.0,
        "avg_prob_yes": (prob_yes_sum / total) if total else 0.0,
        "avg_prob_keep": (prob_yes_sum / total) if total else 0.0,
        "avg_prob_drop": ((total - prob_yes_sum) / total) if total else 0.0,
        "avg_log_odds": (log_odds_sum / total) if total else 0.0,
        "avg_keep_minus_drop": (log_odds_sum / total) if total else 0.0,
        "avg_drop_minus_keep": ((-log_odds_sum) / total) if total else 0.0,
        "filter_strategy": args.filter_strategy,
        "score_field": args.score_field,
        "quality_threshold": float(args.quality_threshold),
        "drop_bottom_percent": float(args.drop_bottom_percent),
        "bottom_percent_scope": str(args.bottom_percent_scope),
        "effective_threshold": threshold_used,
        "effective_threshold_by_section": threshold_used_by_section or None,
        "section_stats": section_stats,
    }
    return kept, summary


def main() -> None:
    args = parse_args()
    if args.qa_input_path is None:
        raise ValueError("Set --qa_input_path.")

    cfg_raw = load_config(str(args.config_path))
    seed = int(cfg_raw.get("seed", 42)) if args.seed is None else int(args.seed)
    _set_all_seeds(seed)

    model = _prepare_model_context(cfg_raw, args)
    _validate_audit_model_support(
        model,
        backend_name=str(args.gen_backend),
        model_name=str(getattr(model, "name", getattr(args, "model_name_or_path", ""))),
    )

    summaries: list[dict[str, Any]] = []

    if args.qa_input_path is not None:
        qa_rows = _load_jsonl(args.qa_input_path)
        qa_output = args.qa_output_path or _default_output_path(args.qa_input_path, "_audited_keep")
        kept, summary = _audit_dataset(
            dataset_name="qa",
            rows=qa_rows,
            max_items=int(args.max_items_per_dataset or 0),
            model=model,
            args=args,
            field_builder=_qa_fields,
            system_prompt=QA_SYSTEM_PROMPT,
            user_prompt_template=QA_USER_PROMPT_TEMPLATE,
            attach_audit_to_kept=bool(args.attach_audit_to_kept),
        )
        _write_jsonl(qa_output, kept, overwrite=args.overwrite)
        summaries.append(
            {
                **summary,
                "input_path": str(args.qa_input_path),
                "keep_output_path": str(qa_output),
            }
        )
        print(f"[qa] keep={summary['keep']} drop={summary['drop']} -> {qa_output}")

    args.summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.summary_output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.summary_output_path}. Use --overwrite.")
    args.summary_output_path.write_text(
        json.dumps(
            {
                "seed": seed,
                "backend": args.gen_backend,
                "model_name": getattr(model, "name", ""),
                "summaries": summaries,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"[done] summary: {args.summary_output_path}")


if __name__ == "__main__":
    main()
