from __future__ import annotations

import gc
import re
import os
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import requests
import torch
from tqdm import tqdm

from src.utils.thinking_tags import (
    contains_thinking_close,
    normalize_early_stopping_text,
    split_after_last_thinking_close,
)
from src.eval.vllm_boxed_mcqa import (
    DEFAULT_EARLY_STOPPING_TEXT,
    extract_boxed_answer,
    is_gemma4_model_spec,
    is_ministral3_model_spec,
    resolve_model_spec,
)
from src.eval.open_judge_eval import OPEN_ENDED_EARLY_STOPPING_TEXT, extract_open_answer


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_server(base_url: str, process: subprocess.Popen, timeout: float) -> None:
    deadline = time.time() + float(timeout)
    last_error: Exception | None = None
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"SGLang server exited early with code {process.returncode}")
        for path in ("/health", "/v1/models"):
            try:
                response = requests.get(base_url.rstrip("/") + path, timeout=2.0)
                if response.status_code < 500:
                    return
            except Exception as exc:
                last_error = exc
        time.sleep(1.0)
    raise TimeoutError(f"Timed out waiting for SGLang server at {base_url}: {last_error}")


def _read_tail(path: str, max_chars: int = 6000) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            data = fh.read()
    except Exception as exc:
        return f"<failed to read log: {exc}>"
    return data[-max_chars:]


def _write_sglang_compatible_lora_weights(source: Path, target: Path) -> bool:
    """Copy safetensors LoRA weights, adding zero q/v counterparts when needed.

    SGLang normalizes q_proj/v_proj LoRA weights into qkv_proj and assumes every
    layer with q_proj LoRA also has v_proj LoRA. PEFT adapters can legitimately
    target q without v, which is equivalent to a zero v LoRA update. Fill those
    missing tensors only in the temporary SGLang adapter directory.
    """
    if source.suffix != ".safetensors":
        return False

    try:
        from safetensors.torch import safe_open, save_file
    except Exception:
        return False

    tensors: dict[str, torch.Tensor] = {}
    metadata = None
    with safe_open(str(source), framework="pt", device="cpu") as f:
        metadata = f.metadata()
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    pattern = re.compile(
        r"^(?P<prefix>.*layers\.(?P<layer>\d+)\.self_attn\.)"
        r"(?P<proj>[qv])_proj\.(?P<ab>lora_[AB])\.weight$"
    )
    seen: dict[tuple[str, str, str], set[str]] = {}
    prototype: dict[tuple[str, str], torch.Tensor] = {}
    for key, tensor in tensors.items():
        match = pattern.match(key)
        if not match:
            continue
        prefix = match.group("prefix")
        layer = match.group("layer")
        proj = match.group("proj")
        ab = match.group("ab")
        seen.setdefault((prefix, layer, ab), set()).add(proj)
        prototype.setdefault((proj, ab), tensor)

    added = 0
    for (prefix, layer, ab), projs in list(seen.items()):
        for proj in ("q", "v"):
            if proj in projs:
                continue
            proto = prototype.get((proj, ab))
            if proto is None:
                continue
            key = f"{prefix}{proj}_proj.{ab}.weight"
            tensors[key] = torch.zeros_like(proto)
            added += 1

    save_file(tensors, str(target), metadata=metadata)
    if added:
        print(
            f"[sglang-lora] added {added} zero q/v LoRA tensors for SGLang compatibility: {target}",
            flush=True,
        )
    return True


def _make_sglang_lora_dir(adapter_path: str) -> tuple[str, tempfile.TemporaryDirectory | None]:
    if os.environ.get("SGLANG_STRIP_LORA_TOKENIZER", "1").strip().lower() in {"0", "false", "no"}:
        return adapter_path, None

    src = Path(adapter_path)
    tmp = tempfile.TemporaryDirectory(prefix=f"sglang_lora_{src.name}_")
    dst = Path(tmp.name)
    required = ["adapter_config.json"]
    for name in required:
        source = src / name
        if not source.exists():
            tmp.cleanup()
            raise FileNotFoundError(f"Missing required LoRA file: {source}")
        shutil.copy2(source, dst / name)

    copied_weights = False
    for pattern in ("adapter_model*.safetensors", "adapter_model*.bin"):
        for source in src.glob(pattern):
            target = dst / source.name
            if not _write_sglang_compatible_lora_weights(source, target):
                shutil.copy2(source, target)
            copied_weights = True
    if not copied_weights:
        tmp.cleanup()
        raise FileNotFoundError(f"No adapter_model weights found in {src}")

    return str(dst), tmp


def _sampling_params(
    *,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    top_p: float,
    top_k: int,
    min_p: float,
    seed: int,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "max_new_tokens": max(1, int(max_new_tokens)),
        "temperature": float(temperature) if do_sample else 0.0,
        "sampling_seed": int(seed),
    }
    if do_sample:
        params.update({
            "top_p": float(top_p),
            "top_k": int(top_k),
            "min_p": float(min_p),
        })
    return params


def _resolve_attention_backend(
    *,
    requested: str | None,
    env_value: str,
    dtype: str,
    model_path: str | dict[str, Any],
) -> str:
    explicit_backend = str(requested or env_value or "").strip()
    backend = explicit_backend or ("triton" if dtype == "float32" else "flashinfer")
    if is_gemma4_model_spec(model_path) and not explicit_backend:
        return "triton"
    return backend


def _extract_generate_texts(response_json: Any) -> list[str]:
    rows = response_json if isinstance(response_json, list) else [response_json]
    texts: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            texts.append(str(row))
            continue
        if "text" in row:
            texts.append(str(row.get("text") or ""))
        elif "output" in row:
            texts.append(str(row.get("output") or ""))
        else:
            texts.append(str(row))
    return texts


def _post_generate(
    *,
    base_url: str,
    prompt_texts: list[str],
    sampling_params: dict[str, Any],
    lora_name: str | None,
    timeout: float,
) -> list[str]:
    payload: dict[str, Any] = {
        "text": prompt_texts,
        "sampling_params": sampling_params,
    }
    if lora_name:
        payload["lora_path"] = [str(lora_name)] * len(prompt_texts)

    response = requests.post(
        base_url.rstrip("/") + "/generate",
        json=payload,
        timeout=float(timeout),
    )
    response.raise_for_status()
    texts = _extract_generate_texts(response.json())
    if len(texts) != len(prompt_texts):
        raise RuntimeError(
            f"SGLang returned {len(texts)} outputs for {len(prompt_texts)} prompts"
        )
    return texts


def _post_generate_batched(
    *,
    base_url: str,
    prompt_texts: list[str],
    sampling_params: dict[str, Any],
    lora_name: str | None,
    timeout: float,
    batch_size: int,
    desc: str,
    use_tqdm: bool,
) -> list[str]:
    if batch_size <= 0 or batch_size >= len(prompt_texts):
        return _post_generate(
            base_url=base_url,
            prompt_texts=prompt_texts,
            sampling_params=sampling_params,
            lora_name=lora_name,
            timeout=timeout,
        )

    outputs: list[str] = []
    starts = range(0, len(prompt_texts), batch_size)
    iterator = tqdm(
        starts,
        total=(len(prompt_texts) + batch_size - 1) // batch_size,
        desc=desc,
        unit="batch",
        disable=not use_tqdm,
    )
    for start in iterator:
        chunk = prompt_texts[start : start + batch_size]
        outputs.extend(_post_generate(
            base_url=base_url,
            prompt_texts=chunk,
            sampling_params=sampling_params,
            lora_name=lora_name,
            timeout=timeout,
        ))
    return outputs


class _SGLangServer:
    def __init__(
        self,
        *,
        model_path: str | dict[str, Any],
        max_model_len: int,
        seed: int,
        gpu_memory_utilization: float,
        tensor_parallel_size: int,
        attention_backend: str,
        sampling_backend: str | None,
        deterministic: bool,
        disable_cuda_graph: bool,
        startup_timeout: float,
        dtype: str,
        cpu_offload_gb: int = 0,
        reasoning_parser: str | None = None,
        moe_runner_backend: str | None = None,
    ) -> None:
        self.resolved = resolve_model_spec(model_path)
        self.base_model = str(self.resolved["base_model"])
        self.tokenizer = str(self.resolved["tokenizer"])
        self.lora_adapter = self.resolved["lora_adapter"]
        self._lora_tmpdir: tempfile.TemporaryDirectory | None = None
        if self.lora_adapter:
            self.lora_adapter, self._lora_tmpdir = _make_sglang_lora_dir(str(self.lora_adapter))
        self.lora_name = str(self.resolved["display_name"]) if self.lora_adapter else None
        self.max_model_len = int(max_model_len)
        self.seed = int(seed)
        self.gpu_memory_utilization = float(gpu_memory_utilization)
        self.tensor_parallel_size = int(tensor_parallel_size)
        self.attention_backend = str(attention_backend)
        self.sampling_backend = str(sampling_backend or "").strip() or None
        self.deterministic = bool(deterministic)
        self.disable_cuda_graph = bool(disable_cuda_graph)
        self.startup_timeout = float(startup_timeout)
        self.dtype = str(dtype)
        self.cpu_offload_gb = int(cpu_offload_gb)
        self.reasoning_parser = str(reasoning_parser or "").strip() or None
        self.moe_runner_backend = str(moe_runner_backend or "").strip() or None
        self.port = _find_free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.log_file = tempfile.NamedTemporaryFile(
            "w",
            prefix="sglang_eval_",
            suffix=".log",
            delete=False,
            encoding="utf-8",
        )
        self.process: subprocess.Popen | None = None

    def __enter__(self) -> "_SGLangServer":
        cmd = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            self.base_model,
            "--tokenizer-path",
            self.tokenizer,
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--dtype",
            self.dtype,
            "--context-length",
            str(self.max_model_len),
            "--mem-fraction-static",
            str(self.gpu_memory_utilization),
            "--tp-size",
            str(self.tensor_parallel_size),
            "--random-seed",
            str(self.seed),
            "--trust-remote-code",
        ]
        if self.attention_backend:
            cmd.extend(["--attention-backend", self.attention_backend])
        if self.sampling_backend:
            cmd.extend(["--sampling-backend", self.sampling_backend])
        if self.deterministic:
            cmd.append("--enable-deterministic-inference")
        if self.disable_cuda_graph:
            cmd.append("--disable-cuda-graph")
        if self.cpu_offload_gb > 0:
            cmd.extend(["--cpu-offload-gb", str(self.cpu_offload_gb)])
        if self.reasoning_parser:
            cmd.extend(["--reasoning-parser", self.reasoning_parser])
        if self.moe_runner_backend:
            cmd.extend(["--moe-runner-backend", self.moe_runner_backend])
        if self.lora_adapter:
            cmd.extend([
                "--enable-lora",
                "--lora-paths",
                f"{self.lora_name}={self.lora_adapter}",
                "--max-loras-per-batch",
                "2",
                "--max-loaded-loras",
                "2",
            ])
        extra_args = os.environ.get("SGLANG_EXTRA_ARGS", "").strip()
        if extra_args:
            cmd.extend(shlex.split(extra_args))

        env = os.environ.copy()
        venv_bin = Path(sys.executable).resolve().parent
        env["PATH"] = f"{venv_bin}{os.pathsep}{env.get('PATH', '')}"
        # Let torch use the wheel-bundled CUDA runtime instead of host overrides.
        env["LD_LIBRARY_PATH"] = ""
        self.process = subprocess.Popen(
            cmd,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        try:
            _wait_for_server(self.base_url, self.process, self.startup_timeout)
        except Exception:
            self.close()
            raise RuntimeError(
                f"SGLang startup failed; log: {self.log_file.name}\n"
                f"--- log tail ---\n{_read_tail(self.log_file.name)}"
            )
        return self

    def close(self) -> None:
        process = self.process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=30)
        try:
            self.log_file.close()
        except Exception:
            pass
        if self._lora_tmpdir is not None:
            self._lora_tmpdir.cleanup()
            self._lora_tmpdir = None

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(5)


def run_sglang_boxed_mcqa(
    *,
    model_path: str | dict[str, Any],
    prompt_texts: list[str],
    message_batches: list[list[dict[str, str]]] | None = None,
    example_keys: list[str],
    thinking_budget_tokens: int,
    answer_max_new_tokens: int,
    max_model_len: int,
    temperature: float,
    do_sample: bool | None = None,
    top_p: float = 1.0,
    top_k: int = -1,
    min_p: float = 0.0,
    seed: int = 0,
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
    early_stopping_text: str = DEFAULT_EARLY_STOPPING_TEXT,
    mode_label: str = "eval",
    use_tqdm: bool = True,
    answer_parser: Callable[[str], Any] = extract_boxed_answer,
    attention_backend: str | None = None,
    deterministic: bool | None = None,
    startup_timeout: float | None = None,
    request_timeout: float | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if len(prompt_texts) != len(example_keys):
        raise ValueError("prompt_texts and example_keys must have the same length")

    resolved = resolve_model_spec(model_path)
    base_model = str(resolved["base_model"])
    early_stopping_text = normalize_early_stopping_text(early_stopping_text, base_model)
    do_sample = (float(temperature) > 0.0) if do_sample is None else bool(do_sample)
    dtype = os.environ.get("SGLANG_DTYPE", "bfloat16").strip().lower() or "bfloat16"
    attention_backend_env = os.environ.get("SGLANG_ATTENTION_BACKEND", "").strip()
    attention_backend = _resolve_attention_backend(
        requested=attention_backend,
        env_value=attention_backend_env,
        dtype=dtype,
        model_path=model_path,
    )
    sampling_backend = os.environ.get("SGLANG_SAMPLING_BACKEND", "").strip() or None
    deterministic = (
        os.environ.get("SGLANG_DETERMINISTIC", "1").strip().lower()
        not in {"0", "false", "no"}
        if deterministic is None
        else bool(deterministic)
    )
    disable_cuda_graph_env = os.environ.get("SGLANG_DISABLE_CUDA_GRAPH", "").strip().lower()
    disable_cuda_graph = (
        dtype == "float32"
        if disable_cuda_graph_env == ""
        else disable_cuda_graph_env not in {"0", "false", "no", "off"}
    )
    startup_timeout = float(startup_timeout or os.environ.get("SGLANG_STARTUP_TIMEOUT", "600"))
    request_timeout = float(request_timeout or os.environ.get("SGLANG_REQUEST_TIMEOUT", "1800"))
    batch_size = int(os.environ.get("SGLANG_BATCH_SIZE", "128"))
    cpu_offload_gb = int(os.environ.get("SGLANG_CPU_OFFLOAD_GB", "0"))
    reasoning_parser = None
    moe_runner_backend = os.environ.get("SGLANG_MOE_RUNNER_BACKEND", "").strip() or None

    eval_start = time.perf_counter()
    with _SGLangServer(
        model_path=model_path,
        max_model_len=max_model_len,
        seed=seed,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        attention_backend=attention_backend,
        sampling_backend=sampling_backend,
        deterministic=deterministic,
        disable_cuda_graph=disable_cuda_graph,
        startup_timeout=startup_timeout,
        dtype=dtype,
        cpu_offload_gb=cpu_offload_gb,
        reasoning_parser=reasoning_parser,
        moe_runner_backend=moe_runner_backend,
    ) as server:
        think_params = _sampling_params(
            max_new_tokens=thinking_budget_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            seed=seed,
        )
        gen_texts = _post_generate_batched(
            base_url=server.base_url,
            prompt_texts=prompt_texts,
            sampling_params=think_params,
            lora_name=server.lora_name,
            timeout=request_timeout,
            batch_size=batch_size,
            desc=f"sglang:{mode_label}:think",
            use_tqdm=bool(use_tqdm),
        )
        reasoning_texts = [""] * len(gen_texts)

        results: dict[str, dict[str, Any]] = {}
        needs_forcing: list[tuple[int, str, str, str]] = []
        for idx, key in enumerate(example_keys):
            gen_text = gen_texts[idx]
            predicted = answer_parser(gen_text)
            force_reason = ""
            if predicted is None:
                if not contains_thinking_close(gen_text):
                    force_reason = "budget_forced_no_think_end"
                else:
                    force_reason = "think_done_missing_boxed"
                needs_forcing.append((idx, key, gen_text, force_reason))
            results[key] = {
                "raw_output": gen_text,
                "reasoning_output": reasoning_texts[idx],
                "predicted_answer": predicted,
                "budget_forced": False,
                "force_reason": force_reason,
            }

        if needs_forcing:
            forced_prompts: list[str] = []
            for idx, _, gen_text, _ in needs_forcing:
                thinking_only = gen_text
                if contains_thinking_close(gen_text):
                    answer_tail = split_after_last_thinking_close(gen_text)
                    if answer_tail and gen_text.endswith(answer_tail):
                        thinking_only = gen_text[: -len(answer_tail)].rstrip()
                forced_prompts.append(prompt_texts[idx] + thinking_only + str(early_stopping_text))

            answer_params = _sampling_params(
                max_new_tokens=answer_max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                seed=seed,
            )
            answer_texts = _post_generate_batched(
                base_url=server.base_url,
                prompt_texts=forced_prompts,
                sampling_params=answer_params,
                lora_name=server.lora_name,
                timeout=request_timeout,
                batch_size=batch_size,
                desc=f"sglang:{mode_label}:force",
                use_tqdm=bool(use_tqdm),
            )
            for j, (idx, key, gen_text, force_reason) in enumerate(needs_forcing):
                answer_text = answer_texts[j]
                full_text = gen_text + str(early_stopping_text) + answer_text
                predicted = answer_parser(full_text)
                results[key] = {
                    "raw_output": full_text,
                    "reasoning_output": "",
                    "predicted_answer": predicted,
                    "budget_forced": True,
                    "force_reason": force_reason,
                }

    elapsed = time.perf_counter() - eval_start
    meta = {
        "mode_label": mode_label,
        "num_examples": len(prompt_texts),
        "elapsed_seconds": float(elapsed),
        "budget_forced_count": int(sum(1 for x in results.values() if bool(x["budget_forced"]))),
        "backend": "sglang",
        "attention_backend": attention_backend,
        "deterministic": bool(deterministic),
        "base_model": str(resolve_model_spec(model_path)["base_model"]),
        "lora_adapter": resolve_model_spec(model_path)["lora_adapter"],
    }
    return results, meta


def run_sglang_open_ended(
    *,
    model_path: str | dict[str, Any],
    prompt_texts: list[str],
    example_keys: list[str],
    thinking_budget_tokens: int,
    answer_max_new_tokens: int,
    max_model_len: int,
    temperature: float,
    do_sample: bool | None = None,
    top_p: float = 1.0,
    top_k: int = -1,
    min_p: float = 0.0,
    seed: int = 0,
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
    early_stopping_text: str = OPEN_ENDED_EARLY_STOPPING_TEXT,
    mode_label: str = "eval",
    use_tqdm: bool = True,
    attention_backend: str | None = None,
    deterministic: bool | None = None,
    startup_timeout: float | None = None,
    request_timeout: float | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if len(prompt_texts) != len(example_keys):
        raise ValueError("prompt_texts and example_keys must have the same length")

    do_sample = (float(temperature) > 0.0) if do_sample is None else bool(do_sample)
    dtype = os.environ.get("SGLANG_DTYPE", "bfloat16").strip().lower() or "bfloat16"
    attention_backend_env = os.environ.get("SGLANG_ATTENTION_BACKEND", "").strip()
    attention_backend = _resolve_attention_backend(
        requested=attention_backend,
        env_value=attention_backend_env,
        dtype=dtype,
        model_path=model_path,
    )
    sampling_backend = os.environ.get("SGLANG_SAMPLING_BACKEND", "").strip() or None
    deterministic = (
        os.environ.get("SGLANG_DETERMINISTIC", "1").strip().lower()
        not in {"0", "false", "no"}
        if deterministic is None
        else bool(deterministic)
    )
    disable_cuda_graph_env = os.environ.get("SGLANG_DISABLE_CUDA_GRAPH", "").strip().lower()
    disable_cuda_graph = (
        dtype == "float32"
        if disable_cuda_graph_env == ""
        else disable_cuda_graph_env not in {"0", "false", "no", "off"}
    )
    startup_timeout = float(startup_timeout or os.environ.get("SGLANG_STARTUP_TIMEOUT", "600"))
    request_timeout = float(request_timeout or os.environ.get("SGLANG_REQUEST_TIMEOUT", "1800"))
    batch_size = int(os.environ.get("SGLANG_BATCH_SIZE", "128"))
    cpu_offload_gb = int(os.environ.get("SGLANG_CPU_OFFLOAD_GB", "0"))

    eval_start = time.perf_counter()
    with _SGLangServer(
        model_path=model_path,
        max_model_len=max_model_len,
        seed=seed,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        attention_backend=attention_backend,
        sampling_backend=sampling_backend,
        deterministic=deterministic,
        disable_cuda_graph=disable_cuda_graph,
        startup_timeout=startup_timeout,
        dtype=dtype,
        cpu_offload_gb=cpu_offload_gb,
    ) as server:
        think_params = _sampling_params(
            max_new_tokens=thinking_budget_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            seed=seed,
        )
        gen_texts = _post_generate_batched(
            base_url=server.base_url,
            prompt_texts=prompt_texts,
            sampling_params=think_params,
            lora_name=server.lora_name,
            timeout=request_timeout,
            batch_size=batch_size,
            desc=f"sglang:{mode_label}:think",
            use_tqdm=bool(use_tqdm),
        )

        results: dict[str, dict[str, Any]] = {}
        needs_forcing: list[tuple[int, str, str, str]] = []
        for idx, key in enumerate(example_keys):
            gen_text = str(gen_texts[idx])
            answer_text = extract_open_answer(gen_text)
            force_reason = ""
            if answer_text is None:
                if "</think>" not in gen_text:
                    force_reason = "budget_forced_no_think_end"
                else:
                    force_reason = "think_done_missing_answer"
                needs_forcing.append((idx, key, gen_text, force_reason))
            results[key] = {
                "raw_output": gen_text,
                "answer_text": answer_text,
                "budget_forced": False,
                "force_reason": force_reason,
            }

        if needs_forcing:
            forced_prompts: list[str] = []
            for idx, _, gen_text, _ in needs_forcing:
                if "</think>" in gen_text:
                    think_end = gen_text.rfind("</think>")
                    thinking_only = gen_text[:think_end]
                else:
                    thinking_only = gen_text
                forced_prompts.append(prompt_texts[idx] + thinking_only + str(early_stopping_text))

            answer_params = _sampling_params(
                max_new_tokens=answer_max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                seed=seed,
            )
            answer_texts = _post_generate_batched(
                base_url=server.base_url,
                prompt_texts=forced_prompts,
                sampling_params=answer_params,
                lora_name=server.lora_name,
                timeout=request_timeout,
                batch_size=batch_size,
                desc=f"sglang:{mode_label}:force",
                use_tqdm=bool(use_tqdm),
            )
            for j, (idx, key, gen_text, force_reason) in enumerate(needs_forcing):
                answer_tail = str(answer_texts[j])
                full_text = gen_text + str(early_stopping_text) + answer_tail
                results[key] = {
                    "raw_output": full_text,
                    "answer_text": extract_open_answer(full_text),
                    "budget_forced": True,
                    "force_reason": force_reason,
                }

    elapsed = time.perf_counter() - eval_start
    meta = {
        "mode_label": mode_label,
        "num_examples": len(prompt_texts),
        "elapsed_seconds": float(elapsed),
        "budget_forced_count": int(sum(1 for x in results.values() if bool(x["budget_forced"]))),
        "backend": "sglang",
        "attention_backend": attention_backend,
        "deterministic": bool(deterministic),
        "base_model": str(resolve_model_spec(model_path)["base_model"]),
        "lora_adapter": resolve_model_spec(model_path)["lora_adapter"],
    }
    return results, meta
