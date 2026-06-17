#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_common import get_eval_results_dir  # noqa: E402


PAIRWISE_SYSTEM_PROMPT = (
    "You are an expert evaluator of pragmatic reasoning. "
    "Compare two anonymized model analyses and judge only their content."
)


@dataclass(frozen=True)
class Comparison:
    label: str
    left_path: Path
    right_path: Path


def _safe_name(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")
    return cleaned or "comparison"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            rows.append(row)
    return rows


def _index_by_id(path: Path) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(path)
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        item_id = str(row.get("id") or "").strip()
        if not item_id:
            continue
        indexed[item_id] = row
    return indexed


def _model_name(rows: dict[str, dict[str, Any]], fallback: Path) -> str:
    for row in rows.values():
        name = str(row.get("model") or "").strip()
        if name:
            return name
    return fallback.stem


def _answer_text(row: dict[str, Any]) -> str:
    predicted = str(row.get("predicted_answer") or "").strip()
    if predicted:
        return predicted
    return str(row.get("raw_output") or "").strip()


def _truncate(text: str, max_chars: int) -> str:
    text = str(text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 32].rstrip() + "\n...[truncated]"


def _build_pairwise_prompt(
    *,
    row: dict[str, Any],
    response_a: str,
    response_b: str,
    max_response_chars: int,
    allow_invalid: bool,
) -> str:
    if allow_invalid:
        choice_instruction = (
            "Focus on content, not formatting. Return Invalid if either response is empty, "
            "nonsense, unrelated, not answering the question, or if you cannot make a clear choice.\n\n"
            "Return ONLY a JSON object with this exact schema:\n"
            "{\n"
            '  "choice": "1" or "2" or "Invalid",\n'
            '  "reason": "<brief explanation in 20 words or less>",\n'
            '  "category": "1" or "2" or "3" or "Invalid"\n'
            "}\n"
        )
    else:
        choice_instruction = (
            "Focus on content, not formatting. You must make a forced choice. "
            "Do not return Invalid. If both responses are flawed, choose the less flawed response. "
            "If one response is empty or unrelated, choose the other response. If both are empty or unrelated, "
            "choose the response that is marginally closer to the golden label.\n\n"
            "Return ONLY a JSON object with this exact schema:\n"
            "{\n"
            '  "choice": "1" or "2",\n'
            '  "reason": "<brief explanation in 20 words or less>",\n'
            '  "category": "1" or "2" or "3"\n'
            "}\n"
        )
    return (
        "You are comparing two model analyses of the intention of response_1, "
        "which is a response to the initial turn.\n\n"
        "Conversation structure:\n"
        f'"context": {json.dumps(row.get("context", ""), ensure_ascii=False)}\n'
        f'"initial turn": {json.dumps(row.get("root", ""), ensure_ascii=False)}\n'
        f'"response_1": {json.dumps(row.get("candidate_sentence_1", ""), ensure_ascii=False)}\n'
        f'"response_2": {json.dumps(row.get("candidate_sentence_2", ""), ensure_ascii=False)}\n'
        f'"golden_label": {json.dumps(row.get("gold_intention", ""), ensure_ascii=False)}\n\n'
        "The models were asked:\n"
        "What is the intention behind response_1? Why or when might someone "
        "prefer response_1 over response_2?\n\n"
        f"Model 1's response:\n{_truncate(response_a, max_response_chars)}\n\n"
        f"Model 2's response:\n{_truncate(response_b, max_response_chars)}\n\n"
        "Choose which response better captures the intention described in the golden label.\n\n"
        "Also categorize how the better response is better than the other:\n"
        "Category #1: Cognitive-Pragmatic Competence. The better response goes beyond "
        "literal meaning and identifies the speaker's underlying social goal, such as "
        "softening a refusal or signaling indirect disagreement.\n"
        "Category #2: Pragmalinguistic Competence. The better response identifies and "
        "explains rhetorical techniques, such as humor, irony, or self-deprecation, and "
        "clarifies how these strategies manage emotion or social tension.\n"
        "Category #3: Sociopragmatic Competence. The better response shows sensitivity "
        "to social context, including roles, relationships, or timing, and explains why "
        "the speaker's choice fits the situation.\n\n"
        f"{choice_instruction}"
    )


def _strip_code_fence(text: str) -> str:
    text = str(text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_judge_json(text: str) -> dict[str, Any]:
    raw = _strip_code_fence(text)
    try:
        parsed = json.loads(raw)
    except Exception:
        return {
            "choice": "Invalid",
            "category": "Invalid",
            "reason": f"judge_parse_error: {(raw or 'EMPTY')[:120]}",
            "raw": raw,
        }

    choice = str(parsed.get("choice") or "").strip()
    if choice not in {"1", "2", "Invalid"}:
        choice = "Invalid"

    category = str(parsed.get("category") or "").strip()
    if category not in {"1", "2", "3", "Invalid"}:
        category = "Invalid"

    reason = str(parsed.get("reason") or "").strip()
    return {
        "choice": choice,
        "category": category,
        "reason": reason,
        "raw": raw,
    }


class PairwiseJudge:
    def __init__(
        self,
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: float,
        max_retries: int,
    ) -> None:
        from openai import OpenAI

        self.client = OpenAI(timeout=float(timeout))
        self.model = str(model)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.max_retries = int(max_retries)

    def judge(self, prompt: str) -> dict[str, Any]:
        last_error = ""
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": PAIRWISE_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                content = str(response.choices[0].message.content or "")
                parsed = _parse_judge_json(content)
                if not str(parsed.get("reason") or "").startswith("judge_parse_error"):
                    return parsed
                last_error = str(parsed.get("reason") or "")
            except Exception as exc:
                last_error = str(exc)
            if attempt < self.max_retries:
                time.sleep(min(2.0 * (attempt + 1), 8.0))

        return {
            "choice": "Invalid",
            "category": "Invalid",
            "reason": f"judge_api_error: {last_error[:120]}",
            "raw": last_error,
        }


def _parse_comparison(spec: str) -> Comparison:
    if "=" not in spec or "::" not in spec:
        raise ValueError(
            "Comparison must be LABEL=LEFT_JSONL::RIGHT_JSONL, "
            f"got: {spec!r}"
        )
    label, paths = spec.split("=", 1)
    left, right = paths.split("::", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"Comparison label is empty: {spec!r}")
    return Comparison(label=label, left_path=Path(left.strip()), right_path=Path(right.strip()))


def _load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            item_id = str(row.get("id") or "").strip()
            if item_id:
                done.add(item_id)
    return done


def _winner_relation(choice: str, side_a_relation: str, side_b_relation: str) -> str:
    if choice == "1":
        return side_a_relation
    if choice == "2":
        return side_b_relation
    return "invalid"


def _judge_one(
    *,
    item_id: str,
    left_row: dict[str, Any],
    right_row: dict[str, Any],
    rng_seed: int,
    judge: PairwiseJudge,
    max_response_chars: int,
    allow_invalid: bool,
) -> dict[str, Any]:
    rng = random.Random(rng_seed)
    left_answer = _answer_text(left_row)
    right_answer = _answer_text(right_row)

    if rng.random() < 0.5:
        response_a = left_answer
        response_b = right_answer
        side_a_relation = "left"
        side_b_relation = "right"
    else:
        response_a = right_answer
        response_b = left_answer
        side_a_relation = "right"
        side_b_relation = "left"

    prompt = _build_pairwise_prompt(
        row=left_row,
        response_a=response_a,
        response_b=response_b,
        max_response_chars=max_response_chars,
        allow_invalid=allow_invalid,
    )
    result = judge.judge(prompt)
    choice = str(result.get("choice") or "Invalid")
    winner = _winner_relation(choice, side_a_relation, side_b_relation)

    return {
        "id": item_id,
        "row_index": left_row.get("row_index"),
        "task": left_row.get("task"),
        "source_candidate": left_row.get("source_candidate"),
        "gold_maxim": left_row.get("gold_maxim"),
        "context": left_row.get("context"),
        "root": left_row.get("root"),
        "candidate_sentence_1": left_row.get("candidate_sentence_1"),
        "candidate_sentence_2": left_row.get("candidate_sentence_2"),
        "gold_intention": left_row.get("gold_intention"),
        "side_a_relation": side_a_relation,
        "side_b_relation": side_b_relation,
        "choice": choice,
        "winner": winner,
        "category": str(result.get("category") or "Invalid"),
        "reason": str(result.get("reason") or ""),
        "judge_raw": str(result.get("raw") or ""),
        "left_answer": left_answer,
        "right_answer": right_answer,
    }


def _pct(num: int, den: int) -> float:
    return (num / den * 100.0) if den else 0.0


def _stable_u32(*parts: object) -> int:
    text = "\x1f".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def _summarize(label: str, left_model: str, right_model: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(row.get("winner") or "invalid") for row in rows)
    category_counts = Counter(str(row.get("category") or "Invalid") for row in rows)
    n = len(rows)
    decided = counts["left"] + counts["right"]
    return {
        "comparison": label,
        "left_model": left_model,
        "right_model": right_model,
        "n_total": n,
        "left_wins": counts["left"],
        "right_wins": counts["right"],
        "invalid_or_tie": counts["invalid"],
        "left_win_rate_all": _pct(counts["left"], n),
        "right_win_rate_all": _pct(counts["right"], n),
        "invalid_or_tie_rate_all": _pct(counts["invalid"], n),
        "left_win_rate_decided": _pct(counts["left"], decided),
        "right_win_rate_decided": _pct(counts["right"], decided),
        "decided_count": decided,
        "category_1": category_counts["1"],
        "category_2": category_counts["2"],
        "category_3": category_counts["3"],
        "category_invalid": category_counts["Invalid"],
    }


def _write_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    fieldnames = [
        "comparison",
        "left_model",
        "right_model",
        "n_total",
        "left_wins",
        "right_wins",
        "invalid_or_tie",
        "left_win_rate_all",
        "right_win_rate_all",
        "invalid_or_tie_rate_all",
        "left_win_rate_decided",
        "right_win_rate_decided",
        "decided_count",
        "category_1",
        "category_2",
        "category_3",
        "category_invalid",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_summary_md(path: Path, summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# AltPrag Pairwise Preference Summary",
        "",
        "| Comparison | Left wins | Right wins | Tie/Invalid | Left decided win % | Categories 1/2/3/Invalid |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {comparison} | {left_wins}/{n_total} ({left_win_rate_all:.1f}%) "
            "| {right_wins}/{n_total} ({right_win_rate_all:.1f}%) "
            "| {invalid_or_tie}/{n_total} ({invalid_or_tie_rate_all:.1f}%) "
            "| {left_win_rate_decided:.1f}% "
            "| {category_1}/{category_2}/{category_3}/{category_invalid} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_comparison(
    *,
    comparison: Comparison,
    output_dir: Path,
    judge: PairwiseJudge,
    seed: int,
    max_examples: int,
    max_workers: int,
    max_response_chars: int,
    allow_invalid: bool,
    resume: bool,
    dry_run: bool,
) -> dict[str, Any] | None:
    left_rows = _index_by_id(comparison.left_path)
    right_rows = _index_by_id(comparison.right_path)
    left_model = _model_name(left_rows, comparison.left_path)
    right_model = _model_name(right_rows, comparison.right_path)

    all_common_ids = sorted(set(left_rows) & set(right_rows))
    common_ids = list(all_common_ids)
    if max_examples > 0 and len(common_ids) > max_examples:
        rng = random.Random(seed)
        common_ids = sorted(rng.sample(common_ids, max_examples))

    detail_path = output_dir / f"{_safe_name(comparison.label)}_pairwise.jsonl"
    completed_ids = _load_completed(detail_path) if resume else set()
    pending_ids = [item_id for item_id in common_ids if item_id not in completed_ids]

    print(
        f"[{comparison.label}] left={left_model} right={right_model} "
        f"matched={len(all_common_ids)} selected={len(common_ids)} "
        f"pending={len(pending_ids)} output={detail_path}"
    )
    if dry_run:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    if resume and detail_path.exists():
        rows.extend(_read_jsonl(detail_path))

    def task(item_id: str) -> dict[str, Any]:
        return _judge_one(
            item_id=item_id,
            left_row=left_rows[item_id],
            right_row=right_rows[item_id],
            rng_seed=_stable_u32(seed, comparison.label, item_id),
            judge=judge,
            max_response_chars=max_response_chars,
            allow_invalid=allow_invalid,
        )

    with detail_path.open("a", encoding="utf-8") as f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(task, item_id): item_id for item_id in pending_ids}
            for i, fut in enumerate(concurrent.futures.as_completed(futures), start=1):
                item_id = futures[fut]
                try:
                    row = fut.result()
                except Exception as exc:
                    row = {
                        "id": item_id,
                        "winner": "invalid",
                        "choice": "Invalid",
                        "category": "Invalid",
                        "reason": f"local_error: {str(exc)[:120]}",
                    }
                row.update({
                    "comparison": comparison.label,
                    "left_model": left_model,
                    "right_model": right_model,
                    "left_path": str(comparison.left_path),
                    "right_path": str(comparison.right_path),
                })
                rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                if i % 25 == 0 or i == len(pending_ids):
                    counts = Counter(str(x.get("winner") or "invalid") for x in rows)
                    print(
                        f"  [{comparison.label}] {i}/{len(pending_ids)} pending done "
                        f"(left={counts['left']} right={counts['right']} invalid={counts['invalid']})",
                        flush=True,
                    )

    summary = _summarize(comparison.label, left_model, right_model, rows)
    summary_path = output_dir / f"{_safe_name(comparison.label)}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pairwise AltPrag preference judge over existing eval_altprag JSONL outputs."
    )
    parser.add_argument(
        "--comparison",
        action="append",
        default=[],
        help="Repeatable. Format: LABEL=LEFT_JSONL::RIGHT_JSONL. LEFT is the model counted as left wins.",
    )
    parser.add_argument("--left", help="Convenience single-comparison left JSONL.")
    parser.add_argument("--right", help="Convenience single-comparison right JSONL.")
    parser.add_argument("--label", default="pairwise", help="Label for --left/--right comparison.")
    parser.add_argument("--output-dir", default="", help="Output directory. Defaults to results/eval/altprag_pairwise/<timestamp>.")
    parser.add_argument("--judge-model", default=os.environ.get("ALTPRAG_PAIRWISE_JUDGE_MODEL", "gpt-4.1"))
    parser.add_argument("--judge-temperature", type=float, default=float(os.environ.get("ALTPRAG_PAIRWISE_JUDGE_TEMPERATURE", "0.0")))
    parser.add_argument("--judge-max-tokens", type=int, default=int(os.environ.get("ALTPRAG_PAIRWISE_JUDGE_MAX_TOKENS", "220")))
    parser.add_argument("--judge-timeout", type=float, default=float(os.environ.get("ALTPRAG_PAIRWISE_JUDGE_TIMEOUT", "180.0")))
    parser.add_argument("--judge-max-retries", type=int, default=int(os.environ.get("ALTPRAG_PAIRWISE_JUDGE_MAX_RETRIES", "3")))
    parser.add_argument("--max-workers", type=int, default=int(os.environ.get("ALTPRAG_PAIRWISE_MAX_WORKERS", "4")))
    parser.add_argument("--max-examples", type=int, default=int(os.environ.get("ALTPRAG_PAIRWISE_MAX_EXAMPLES", "0")))
    parser.add_argument("--max-response-chars", type=int, default=int(os.environ.get("ALTPRAG_PAIRWISE_MAX_RESPONSE_CHARS", "3000")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("ALTPRAG_PAIRWISE_SEED", "42")))
    parser.add_argument(
        "--force-choice",
        action="store_true",
        help="Do not allow the judge to return Invalid; force a 1/2 preference even when both outputs are flawed.",
    )
    parser.add_argument("--resume", action="store_true", help="Skip IDs already present in detail JSONL.")
    parser.add_argument("--dry-run", action="store_true", help="Only validate inputs and matched IDs; do not call the judge.")
    args = parser.parse_args()

    comparisons = [_parse_comparison(spec) for spec in args.comparison]
    if args.left or args.right:
        if not (args.left and args.right):
            raise SystemExit("--left and --right must be provided together")
        comparisons.append(Comparison(label=args.label, left_path=Path(args.left), right_path=Path(args.right)))
    if not comparisons:
        raise SystemExit("Provide at least one --comparison LABEL=LEFT_JSONL::RIGHT_JSONL or --left/--right.")

    for comp in comparisons:
        if not comp.left_path.exists():
            raise SystemExit(f"Left JSONL not found for {comp.label}: {comp.left_path}")
        if not comp.right_path.exists():
            raise SystemExit(f"Right JSONL not found for {comp.label}: {comp.right_path}")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = get_eval_results_dir(ROOT, "altprag_pairwise") / f"altprag_pairwise_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "judge_model": args.judge_model,
        "judge_temperature": args.judge_temperature,
        "max_examples": args.max_examples,
        "seed": args.seed,
        "force_choice": bool(args.force_choice),
        "comparisons": [
            {"label": c.label, "left_path": str(c.left_path), "right_path": str(c.right_path)}
            for c in comparisons
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    judge = None
    if not args.dry_run:
        judge = PairwiseJudge(
            model=args.judge_model,
            temperature=args.judge_temperature,
            max_tokens=args.judge_max_tokens,
            timeout=args.judge_timeout,
            max_retries=args.judge_max_retries,
        )

    summaries: list[dict[str, Any]] = []
    for comp in comparisons:
        summary = run_comparison(
            comparison=comp,
            output_dir=output_dir,
            judge=judge,  # type: ignore[arg-type]
            seed=args.seed,
            max_examples=args.max_examples,
            max_workers=max(1, args.max_workers),
            max_response_chars=max(200, args.max_response_chars),
            allow_invalid=not bool(args.force_choice),
            resume=bool(args.resume),
            dry_run=bool(args.dry_run),
        )
        if summary is not None:
            summaries.append(summary)

    if summaries:
        summary_json = output_dir / "pairwise_summary.json"
        summary_csv = output_dir / "pairwise_summary.csv"
        summary_md = output_dir / "pairwise_summary.md"
        summary_json.write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        _write_summary_csv(summary_csv, summaries)
        _write_summary_md(summary_md, summaries)
        print(f"Saved summary JSON: {summary_json}")
        print(f"Saved summary CSV:  {summary_csv}")
        print(f"Saved summary MD:   {summary_md}")
    else:
        print(f"Dry run complete. Manifest written to {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
