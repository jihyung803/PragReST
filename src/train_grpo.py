"""
Custom GRPO training entry point that adds lm-pragmatics evaluation during training.

Subclasses RayPPOTrainer to override _validate(): after each standard validation
step, also evaluates on lm-pragmatics using the same vLLM rollout worker that is
already running (no FSDP merge, no extra GPU memory needed).

Results are logged to W&B under val/lm_pragmatics_* keys.

Usage (drop-in replacement for python -m verl.trainer.main_ppo):
  python src/train_grpo.py \
      [all the same hydra args as train_grpo.slurm] \
      +trainer.pragmatics_eval.data_path="$BK_LINK_DIR/lm-pragmatics/prompts" \
      +trainer.pragmatics_eval.batch_size=16
"""

import csv
import os
import re
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import requests
from verl.trainer.main_ppo import TaskRunner, run_ppo  # type: ignore
from verl.trainer.ppo.ray_trainer import RayPPOTrainer  # type: ignore


# ── Prompt / data helpers (mirrors evaluate_verl_checkpoints_pragmatics.py) ───

SYSTEM_PROMPT = (
    "Choose the best answer option based on the given context.\n"
    "Final answer of the question must be enclosed in \\boxed{...}."
)
_THINK_END_TOKEN_ID = 151668   # </think>
_IM_END_TOKEN_ID = 151645      # <|im_end|>
_EARLY_STOPPING_TEXT = (
    "\n\nConsidering the limited time by the user, I have to give the "
    "solution based on the thinking directly now.\n</think>\n\n"
)


def load_lm_pragmatics_data(data_dir: str) -> List[Dict]:
    data_dir = Path(data_dir)
    examples = []
    csv_files = [
        f for f in data_dir.glob("*_prompts_seed0_examples0.csv")
        if "no-story" not in f.name
    ]
    for csv_file in sorted(csv_files):
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("is_example", "False") == "True":
                    continue
                full_prompt = row.get("prompt", "").strip()
                if full_prompt.rstrip().endswith("Answer:"):
                    full_prompt = full_prompt.rstrip()[: -len("Answer:")].rstrip()
                examples.append({
                    "full_prompt": full_prompt,
                    "correct_answer_idx": int(row.get("randomized_true_answer", 0)),
                    "phenomenon": row.get("phenomenon", ""),
                })
    return examples


def make_messages(example: Dict) -> List[Dict]:
    user_message = str(example["full_prompt"]).strip()
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


def extract_answer(text: str) -> Optional[str]:
    boxed_patterns = [
        r"\\boxed\{\s*(\d+)\s*\}",
        r"\$\\boxed\{\s*(\d+)\s*\}\$",
    ]
    for pattern in boxed_patterns:
        m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).strip()

    patterns = [
        r"Answer:\s*(\d+)",
        r"answer:\s*(\d+)",
        r"Final Answer:\s*(\d+)",
        r"final answer:\s*(\d+)",
        r"Answer:\s*(\d+)\)",
        r"answer:\s*(\d+)\)",
        r"^(\d+)\)?\s*$",
        r"option\s*(\d+)",
        r"Option\s*(\d+)",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).strip()
    return None


def extract_answer_with_judge(generated_text: str) -> Optional[str]:
    """
    Fallback: ask the judge vLLM API server to extract the answer number.
    Uses JUDGE_BASE_URL / JUDGE_MODEL_NAME env vars (same as reward_fn_verl.py).
    Returns None if the judge is unavailable or returns a non-integer.
    """
    base_url = os.environ.get("JUDGE_BASE_URL", "").rstrip("/")
    if not base_url:
        return None

    model_name = os.environ.get("JUDGE_MODEL_NAME", "judge")
    prompt = (
        "Extract the answer number from the text below. "
        "The answer is a single integer (1, 2, 3, or 4). "
        "Output ONLY the integer, nothing else.\n\n"
        f"Text:\n{generated_text}\n\n"
        "Answer number:"
    )
    messages = [{"role": "user", "content": prompt}]
    try:
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": model_name,
                "messages": messages,
                "max_tokens": 32,
                "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=30,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        m = re.search(r"(\d+)", content)
        if m:
            return m.group(1)
        print(f"[PragmaticsEval] Judge could not extract number. Judge raw response: {content!r}")
        return None
    except Exception as exc:
        print(f"[PragmaticsEval] Judge fallback failed: {exc}")
        return None


def _trim_trailing_pad_tokens(token_ids, pad_token_id: Optional[int]) -> List[int]:
    token_ids = token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)
    if pad_token_id is None:
        return token_ids
    end = len(token_ids)
    while end > 0 and token_ids[end - 1] == pad_token_id:
        end -= 1
    return token_ids[:end]


def _classify_budget_forcing_state(output_ids: List[int]) -> str:
    if _IM_END_TOKEN_ID in output_ids:
        return "finished"
    if _THINK_END_TOKEN_ID in output_ids:
        return "think_done"
    return "budget_forced"


# ── Custom trainer ────────────────────────────────────────────────────────────

class RayPPOTrainerWithPragmaticsEval(RayPPOTrainer):
    """
    Extends RayPPOTrainer to add lm-pragmatics evaluation at every validation step.
    Uses the already-running vLLM rollout worker — no FSDP merge required.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        rollout_cfg = getattr(
            getattr(self.config, "actor_rollout_ref", None), "rollout", None
        )
        rollout_name = getattr(rollout_cfg, "name", "unknown")
        hybrid = getattr(
            getattr(self.config, "actor_rollout_ref", None), "hybrid_engine", False
        )
        gpu_mem = getattr(rollout_cfg, "gpu_memory_utilization", "?")
        tp = getattr(rollout_cfg, "tensor_model_parallel_size", "?")
        print(
            f"[Trainer] Rollout engine: name={rollout_name!r}  "
            f"hybrid_engine={hybrid}  "
            f"gpu_memory_utilization={gpu_mem}  "
            f"tensor_parallel_size={tp}"
        )

        eval_cfg = getattr(self.config.trainer, "pragmatics_eval", None)
        data_path = getattr(eval_cfg, "data_path", None) if eval_cfg else None

        if data_path:
            self._pragmatics_examples = load_lm_pragmatics_data(data_path)
            self._pragmatics_batch_size = getattr(eval_cfg, "batch_size", 16)
            # Keep eval prompt truncation separate from training prompt length.
            self._pragmatics_max_prompt_length = getattr(eval_cfg, "max_prompt_length", 2048)
            self._pragmatics_thinking_budget = getattr(eval_cfg, "thinking_budget", 824)
            self._pragmatics_answer_max_new_tokens = getattr(eval_cfg, "answer_max_new_tokens", 64)
            self._pragmatics_max_samples = getattr(eval_cfg, "max_samples", None)
            print(f"[PragmaticsEval] Loaded {len(self._pragmatics_examples)} examples "
                  f"(batch_size={self._pragmatics_batch_size}, "
                  f"max_prompt_length={self._pragmatics_max_prompt_length}, "
                  f"thinking_budget={self._pragmatics_thinking_budget}, "
                  f"answer_max_new_tokens={self._pragmatics_answer_max_new_tokens}, "
                  f"max_samples={self._pragmatics_max_samples})")
        else:
            self._pragmatics_examples = None
            print("[PragmaticsEval] No data_path set — skipping lm-pragmatics eval.")

    def _compute_reward_colocate(self, batch):
        batch_reward = super()._compute_reward_colocate(batch)
        try:
            conv_types = batch_reward.non_tensor_batch.get("conv_type", None)
            if conv_types is None:
                conv_types = batch_reward.non_tensor_batch.get("task_type", None)
            if conv_types is not None:
                rewards = batch_reward.batch["rm_scores"].sum(-1).tolist()
                fmt_rewards = batch_reward.non_tensor_batch.get("format_reward", None)
                by_type = defaultdict(lambda: {"reward": [], "format_reward": []})
                for i, (r, ct) in enumerate(zip(rewards, conv_types)):
                    ct = str(ct)
                    by_type[ct]["reward"].append(r)
                    if fmt_rewards is not None:
                        by_type[ct]["format_reward"].append(float(fmt_rewards[i]))
                metrics = {}
                for ct, vals in by_type.items():
                    metrics[f"train/reward_{ct}"] = sum(vals["reward"]) / len(vals["reward"])
                    if vals["format_reward"]:
                        metrics[f"train/format_reward_{ct}"] = sum(vals["format_reward"]) / len(vals["format_reward"])
                import wandb
                if wandb.run is not None:
                    wandb.log({**metrics, "trainer/global_step": self.global_steps})
                print("[RewardByType] " + ", ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
        except Exception as exc:
            print(f"[RewardByType] Error logging per-conv_type rewards: {exc}")
        return batch_reward

    def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns, sample_response_lengths=None):
        metric_dict = super()._val_metrics_update(
            data_sources, sample_uids, reward_extra_infos_dict, sample_turns, sample_response_lengths
        )
        rewards = reward_extra_infos_dict.get("reward", [])
        conv_types = reward_extra_infos_dict.get("conv_type", [])
        if not conv_types:
            conv_types = reward_extra_infos_dict.get("task_type", [])
        fmt_rewards = reward_extra_infos_dict.get("format_reward", [])
        if rewards and conv_types:
            by_type = defaultdict(lambda: {"reward": [], "format_reward": []})
            for i, (r, ct) in enumerate(zip(rewards, conv_types)):
                ct = str(ct)
                by_type[ct]["reward"].append(r)
                if fmt_rewards:
                    by_type[ct]["format_reward"].append(float(fmt_rewards[i]))
            for ct, vals in by_type.items():
                metric_dict[f"val/reward_{ct}"] = sum(vals["reward"]) / len(vals["reward"])
                if vals["format_reward"]:
                    metric_dict[f"val/format_reward_{ct}"] = sum(vals["format_reward"]) / len(vals["format_reward"])
        return metric_dict

    def _validate(self, merged: bool = False):
        # Standard VERL validation on the configured validation set.
        metrics = super()._validate(merged=merged)

        if self._pragmatics_examples:
            pragmatics_metrics = self._validate_lm_pragmatics()
            metrics.update(pragmatics_metrics)

        return metrics

    def _validate_lm_pragmatics(self) -> Dict:
        from verl import DataProto
        import torch

        # ── Confirm which rollout backend is active ───────────────────────────
        use_async = hasattr(self, "async_rollout_manager")
        rollout_worker = (
            self.async_rollout_manager if use_async else self.actor_rollout_wg
        )
        rollout_cls = type(rollout_worker).__name__
        rollout_name = getattr(
            getattr(self.config, "actor_rollout_ref", None),
            "rollout", None,
        )
        rollout_name = getattr(rollout_name, "name", "unknown")
        print(
            f"[PragmaticsEval] Rollout backend: name={rollout_name!r}  "
            f"worker_type={rollout_cls!r}  use_async={use_async}"
        )

        examples = self._pragmatics_examples
        batch_size = self._pragmatics_batch_size
        max_prompt_length = self._pragmatics_max_prompt_length
        thinking_budget = self._pragmatics_thinking_budget
        answer_max_new_tokens = self._pragmatics_answer_max_new_tokens
        max_samples = self._pragmatics_max_samples

        if max_samples is not None and max_samples < len(examples):
            import random

            random.seed(42)
            examples = random.sample(examples, max_samples)

        correct = 0
        total = 0
        extraction_failures = 0
        results_by_phenomenon = defaultdict(lambda: {"correct": 0, "total": 0})
        pass1_counts = {"finished": 0, "think_done": 0, "budget_forced": 0}
        pass2_counts = {"continued_after_think": 0, "budget_forced": 0}

        num_batches = (len(examples) + batch_size - 1) // batch_size
        eval_start = time.perf_counter()

        def _pad_left_sequences(sequences: List[torch.Tensor]):
            max_len = max(seq.shape[0] for seq in sequences)
            pad_id = self.tokenizer.pad_token_id
            padded, masks = [], []
            for seq in sequences:
                pad_len = max_len - seq.shape[0]
                if pad_len > 0:
                    pad = torch.full((pad_len,), pad_id, dtype=seq.dtype)
                    padded.append(torch.cat([pad, seq], dim=0))
                    masks.append(
                        torch.cat(
                            [
                                torch.zeros(pad_len, dtype=torch.long),
                                torch.ones(seq.shape[0], dtype=torch.long),
                            ],
                            dim=0,
                        )
                    )
                else:
                    padded.append(seq)
                    masks.append(torch.ones(seq.shape[0], dtype=torch.long))

            input_ids = torch.stack(padded, dim=0)
            attention_mask = torch.stack(masks, dim=0)
            position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0) * attention_mask
            return input_ids, attention_mask, position_ids

        def _build_gen_batch(
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            position_ids: torch.Tensor,
            raw_prompts: List[List[Dict]],
            *,
            eos_token_id,
            max_new_tokens: int,
            do_sample: bool = False,
        ):
            orig_size = input_ids.shape[0]
            use_async = hasattr(self, "async_rollout_manager")
            chunk_size = 1
            if use_async:
                try:
                    chunk_size = len(self.async_rollout_manager.agent_loop_workers)
                except Exception:
                    chunk_size = 1

            pad_size = 0
            if chunk_size > 1:
                remainder = orig_size % chunk_size
                if remainder != 0:
                    pad_size = chunk_size - remainder

            if pad_size > 0:
                input_ids = torch.cat([input_ids, input_ids[-1:].expand(pad_size, -1)], dim=0)
                attention_mask = torch.cat([attention_mask, attention_mask[-1:].expand(pad_size, -1)], dim=0)
                position_ids = torch.cat([position_ids, position_ids[-1:].expand(pad_size, -1)], dim=0)
                raw_prompts = raw_prompts + [raw_prompts[-1]] * pad_size

            gen_batch = DataProto.from_dict(
                tensors={
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                }
            )
            padded_size = input_ids.shape[0]
            gen_batch.non_tensor_batch["raw_prompt"] = np.array(raw_prompts, dtype=object)
            gen_batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(padded_size)], dtype=object
            )
            gen_batch.non_tensor_batch["data_source"] = np.array(
                ["pragmatics_eval"] * padded_size, dtype=object
            )
            gen_batch.non_tensor_batch["reward_model"] = np.array(
                [{"ground_truth": "{}"}] * padded_size, dtype=object
            )
            gen_batch.meta_info = {
                "eos_token_id": eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": do_sample,
                "validate": True,
                "global_steps": self.global_steps,
                "max_response_length": max_new_tokens,
                "response_length": max_new_tokens,
            }
            return gen_batch, orig_size, use_async

        for batch_idx in range(num_batches):
            batch = examples[batch_idx * batch_size: (batch_idx + 1) * batch_size]

            # ── Tokenize prompts ──────────────────────────────────────────────
            prompt_texts = []
            messages_list = []
            for ex in batch:
                messages = make_messages(ex)
                messages_list.append(messages)
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
                prompt_texts.append(text)

            # Ensure left-padding regardless of tokenizer defaults.
            original_padding_side = self.tokenizer.padding_side
            self.tokenizer.padding_side = "left"
            try:
                encoded = self.tokenizer(
                    prompt_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_prompt_length,
                )
            finally:
                self.tokenizer.padding_side = original_padding_side

            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
            position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0) * attention_mask

            # ── Pass 1: think until </think> or <|im_end|>, or hit budget ─────
            gen_batch, orig_size, use_async = _build_gen_batch(
                input_ids,
                attention_mask,
                position_ids,
                messages_list.copy(),
                eos_token_id=[_THINK_END_TOKEN_ID, _IM_END_TOKEN_ID],
                max_new_tokens=thinking_budget,
                do_sample=False,
            )

            print(
                f"[PragmaticsEval] Pass 1 generate_sequences via "
                f"{'async_rollout_manager' if use_async else 'actor_rollout_wg'} "
                f"(batch_size={len(batch)}, max_new_tokens={thinking_budget})"
            )
            if use_async:
                output = self.async_rollout_manager.generate_sequences(gen_batch)
            else:
                output = self.actor_rollout_wg.generate_sequences(gen_batch)

            responses = output.batch["responses"][:orig_size]

            pass1_results = []
            second_pass_prefixes = []
            second_pass_prompt_ids = []
            second_pass_messages = []
            second_pass_example_indices = []
            second_pass_states = []
            early_ids = self.tokenizer(
                _EARLY_STOPPING_TEXT, return_tensors="pt", add_special_tokens=False
            )["input_ids"][0].to(dtype=input_ids.dtype)

            for i, ex in enumerate(batch):
                prompt_ids = input_ids[i][attention_mask[i].bool()]
                output_ids = _trim_trailing_pad_tokens(
                    responses[i], self.tokenizer.pad_token_id
                )
                pass1_state = _classify_budget_forcing_state(output_ids)
                pass1_counts[pass1_state] += 1

                if pass1_state == "finished":
                    generated = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                    pass1_results.append((generated, generated))  # no separate answer section
                    continue

                prefix_ids = torch.tensor(output_ids, dtype=input_ids.dtype)
                if pass1_state == "budget_forced":
                    print("[PragmaticsEval] thinking budget is reached")
                    prefix_ids = torch.cat([prefix_ids, early_ids], dim=0)

                combined_prompt_ids = torch.cat([prompt_ids, prefix_ids], dim=0)
                second_pass_prompt_ids.append(combined_prompt_ids)
                second_pass_prefixes.append(prefix_ids)
                second_pass_messages.append(messages_list[i])
                second_pass_example_indices.append(i)
                second_pass_states.append(pass1_state)
                pass1_results.append(None)  # will be replaced by (full, answer_only) tuple

            # ── Pass 2: short answer continuation ────────────────────────────
            if second_pass_prompt_ids:
                pass2_input_ids, pass2_attention_mask, pass2_position_ids = _pad_left_sequences(
                    second_pass_prompt_ids
                )
                pass2_batch, second_orig_size, use_async = _build_gen_batch(
                    pass2_input_ids,
                    pass2_attention_mask,
                    pass2_position_ids,
                    second_pass_messages.copy(),
                    eos_token_id=self.tokenizer.eos_token_id,
                    max_new_tokens=answer_max_new_tokens,
                    do_sample=False,
                )

                print(
                    f"[PragmaticsEval] Pass 2 generate_sequences via "
                    f"{'async_rollout_manager' if use_async else 'actor_rollout_wg'} "
                    f"(n={len(second_pass_prompt_ids)}, max_new_tokens={answer_max_new_tokens})"
                )
                if use_async:
                    pass2_output = self.async_rollout_manager.generate_sequences(pass2_batch)
                else:
                    pass2_output = self.actor_rollout_wg.generate_sequences(pass2_batch)

                pass2_responses = pass2_output.batch["responses"][:second_orig_size]

                for local_idx, example_idx in enumerate(second_pass_example_indices):
                    continuation_ids = _trim_trailing_pad_tokens(
                        pass2_responses[local_idx], self.tokenizer.pad_token_id
                    )
                    full_ids = torch.cat(
                        [
                            second_pass_prefixes[local_idx],
                            torch.tensor(continuation_ids, dtype=input_ids.dtype),
                        ],
                        dim=0,
                    )
                    generated = self.tokenizer.decode(full_ids, skip_special_tokens=True)
                    answer_only = self.tokenizer.decode(continuation_ids, skip_special_tokens=True)
                    pass1_results[example_idx] = (generated, answer_only)
                    if second_pass_states[local_idx] == "think_done":
                        pass2_counts["continued_after_think"] += 1
                    else:
                        pass2_counts["budget_forced"] += 1

            # ── Score ─────────────────────────────────────────────────────────
            for i, ex in enumerate(batch):
                generated, answer_only = pass1_results[i]
                predicted = extract_answer(generated)
                correct_idx = ex["correct_answer_idx"]
                phenomenon = ex["phenomenon"]

                is_correct = False
                if predicted is None:
                    # Try regex on answer-only text first (avoids thinking noise)
                    predicted = extract_answer(answer_only)
                if predicted is None:
                    # Judge sees only the answer section, not the full thinking
                    predicted = extract_answer_with_judge(answer_only)
                    if predicted is not None:
                        print(f"[PragmaticsEval] Judge fallback extracted: {predicted!r}")
                    else:
                        extraction_failures += 1
                        print(
                            f"[PragmaticsEval] EXTRACTION FAILURE #{extraction_failures} "
                            f"phenomenon={phenomenon!r} correct={correct_idx}\n"
                            f"  Answer section: {answer_only!r}"
                        )

                if predicted is not None:
                    try:
                        is_correct = int(predicted) == correct_idx
                    except ValueError:
                        pass

                if is_correct:
                    correct += 1
                total += 1
                results_by_phenomenon[phenomenon]["total"] += 1
                if is_correct:
                    results_by_phenomenon[phenomenon]["correct"] += 1

        accuracy = correct / total if total > 0 else 0.0
        elapsed = time.perf_counter() - eval_start

        SEP = "=" * 60
        print(f"\n{SEP}")
        print(f"[PragmaticsEval] step={self.global_steps}")
        print(SEP)
        print(f"  Eval seconds       : {elapsed:.1f}")
        print(
            "  Pass 1 counts      : "
            f"finished={pass1_counts['finished']} "
            f"think_done={pass1_counts['think_done']} "
            f"budget_forced={pass1_counts['budget_forced']}"
        )
        print(
            "  Pass 2 counts      : "
            f"continued_after_think={pass2_counts['continued_after_think']} "
            f"budget_forced={pass2_counts['budget_forced']}"
        )
        print(f"  Overall accuracy   : {accuracy:.4f}  ({correct}/{total})")
        print(f"  Extraction failures: {extraction_failures}/{total}")
        print(f"  Per-phenomenon:")
        for ph in sorted(results_by_phenomenon):
            d = results_by_phenomenon[ph]
            ph_acc = d["correct"] / d["total"]
            print(f"    {ph:<30s}  {ph_acc:.4f}  ({d['correct']}/{d['total']})")
        print(SEP + "\n")

        metrics = {
            "val/lm_pragmatics_accuracy": accuracy,
            "val/lm_pragmatics_extraction_failures": extraction_failures,
            "val/lm_pragmatics_eval_seconds": elapsed,
            "val/lm_pragmatics_pass1_finished": pass1_counts["finished"],
            "val/lm_pragmatics_pass1_think_done": pass1_counts["think_done"],
            "val/lm_pragmatics_pass1_budget_forced": pass1_counts["budget_forced"],
            "val/lm_pragmatics_pass2_continued_after_think": pass2_counts["continued_after_think"],
            "val/lm_pragmatics_pass2_budget_forced": pass2_counts["budget_forced"],
        }
        for ph, d in results_by_phenomenon.items():
            metrics[f"val/lm_pragmatics_{ph}"] = d["correct"] / d["total"]

        return metrics


# ── Custom TaskRunner ─────────────────────────────────────────────────────────

class TaskRunnerWithPragmaticsEval(TaskRunner):
    """Swaps in RayPPOTrainerWithPragmaticsEval instead of the base trainer."""

    def run(self, config):
        # Temporarily replace RayPPOTrainer in main_ppo's namespace
        import verl.trainer.main_ppo as main_ppo_module
        original = main_ppo_module.RayPPOTrainer
        main_ppo_module.RayPPOTrainer = RayPPOTrainerWithPragmaticsEval
        try:
            super().run(config)
        finally:
            main_ppo_module.RayPPOTrainer = original


# ── Entry point ───────────────────────────────────────────────────────────────

import hydra
import ray
import verl.trainer.main_ppo as _main_ppo_ref

# Resolve the config dir from the installed verl package (works regardless of
# where the repo is cloned or whether a local verl/ tree exists).
_VERL_CONFIG_DIR = str(Path(_main_ppo_ref.__file__).parent / "config")


@hydra.main(config_path=_VERL_CONFIG_DIR, config_name="ppo_trainer", version_base=None)
def main(config):
    from verl.utils.device import auto_set_device
    auto_set_device(config)

    task_runner_cls = ray.remote(num_cpus=1)(TaskRunnerWithPragmaticsEval)
    run_ppo(config, task_runner_class=task_runner_cls)


if __name__ == "__main__":
    main()
