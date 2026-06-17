#!/usr/bin/env bash
set -eo pipefail
set -u

: "${WORK:?Set WORK to a writable work/cache directory}"
: "${SCRATCH:?Set SCRATCH to a scratch/checkpoint directory}"

export VLLM_USE_V1=0
export VLLM_FLASH_ATTN_VERSION=2
export OMP_NUM_THREADS=8
export MALLOC_ARENA_MAX=4

export PRAGREST_REPO_DIR="${PRAGREST_REPO_DIR:-$PWD}"
export VERL_SETUP_DIR="${VERL_SETUP_DIR:?Set VERL_SETUP_DIR to your VERL environment directory}"
export HF_HOME="$WORK/.cache/huggingface"
export WANDB_DIR="$WORK/.cache/wandb"
export PIP_CACHE_DIR="$WORK/.cache/pip"
export BASE_MODEL="${BASE_MODEL:-$SCRATCH/models/Qwen3-8B}"

# Use the SFT checkpoint as the rollout/policy generator so filtering reflects
# what the actual starting policy already finds trivial. Judge stays on the
# base model.
export SFT_MODEL_DIR="${SFT_MODEL_DIR:-$SCRATCH/models/Qwen3-8B-PragReST-SFT}"
export SFT_MODEL_HF="${SFT_MODEL_HF:-}"
export HF_TOKEN="${HF_TOKEN:-}"

POLICY_PORT=8200
JUDGE_PORT=8300
FILTER_MAX_WORKERS="${FILTER_MAX_WORKERS:-32}"
FILTER_MAX_TOKENS="${FILTER_MAX_TOKENS:-2048}"
FILTER_ROLLOUT_CHUNK_SIZE="${FILTER_ROLLOUT_CHUNK_SIZE:-4}"
FILTER_JUDGE_CHUNK_WORKERS="${FILTER_JUDGE_CHUNK_WORKERS:-4}"
FILTER_JUDGE_MARGIN_THRESHOLD="${FILTER_JUDGE_MARGIN_THRESHOLD:-0.8}"
FILTER_EVAL_PER_SECTION="${FILTER_EVAL_PER_SECTION:-10}"
POLICY_GPU_MEMORY_UTIL="${POLICY_GPU_MEMORY_UTIL:-0.46}"
JUDGE_GPU_MEMORY_UTIL="${JUDGE_GPU_MEMORY_UTIL:-0.46}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-64}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"

module purge
module load gcc/13.2.0 cuda/12.8

if [ ! -f "$VERL_SETUP_DIR/env_vars.sh" ]; then
    echo "ERROR: Missing VERL env setup script: $VERL_SETUP_DIR/env_vars.sh"
    exit 1
fi

if [ ! -f "$VERL_SETUP_DIR/.venv/bin/activate" ]; then
    echo "ERROR: Missing VERL virtualenv activate script: $VERL_SETUP_DIR/.venv/bin/activate"
    exit 1
fi

if [ ! -f "$BASE_MODEL/config.json" ]; then
    echo "ERROR: Missing local base model at $BASE_MODEL"
    echo "Expected file: $BASE_MODEL/config.json"
    exit 1
fi

# Use the requested SFT checkpoint as the rollout/policy generator.
if [ -f "$SFT_MODEL_DIR/config.json" ]; then
    echo "Using cached SFT model: $SFT_MODEL_DIR"
    POLICY_MODEL="$SFT_MODEL_DIR"
elif [ -n "$SFT_MODEL_HF" ]; then
    if [ -z "$HF_TOKEN" ]; then
        echo "ERROR: Set HF_TOKEN in the job environment to download $SFT_MODEL_HF."
        exit 1
    fi
    if huggingface-cli download "$SFT_MODEL_HF" \
            --local-dir "$SFT_MODEL_DIR" \
            --token "$HF_TOKEN" 2>&1 | tee hf_download_${SLURM_JOB_ID}.log; then
        echo "SFT model downloaded to $SFT_MODEL_DIR"
        POLICY_MODEL="$SFT_MODEL_DIR"
    else
        echo "ERROR: Failed to download requested SFT checkpoint: $SFT_MODEL_HF"
        exit 1
    fi
else
    echo "ERROR: Missing requested SFT checkpoint: $SFT_MODEL_DIR"
    echo "Expected file: $SFT_MODEL_DIR/config.json"
    exit 1
fi
echo "Policy model: $POLICY_MODEL"
echo "Judge  model: $BASE_MODEL"

# env_vars.sh and virtualenv activation may not be nounset-safe.
had_nounset=0
case $- in
    *u*)
        had_nounset=1
        set +u
        ;;
esac

cd "$VERL_SETUP_DIR"
source env_vars.sh
source .venv/bin/activate
export TIKTOKEN_ENCODINGS_BASE="$WORK/encodings"

if [ "$had_nounset" -eq 1 ]; then
    set -u
fi

cd "$PRAGREST_REPO_DIR"

DATA_DIR="${DATA_DIR:-$PRAGREST_REPO_DIR/data/verl-mixed-qwen-8b}"
NO_EASY_NO_HARD_DIR="$DATA_DIR/no_easy_no_hard"
NO_EASY_DIR="$DATA_DIR/no_easy"
INPUT_PARQUET="$DATA_DIR/all.parquet"
NO_EASY_NO_HARD_ALL_FILTERED="$NO_EASY_NO_HARD_DIR/all_filtered.parquet"
NO_EASY_ALL_FILTERED="$NO_EASY_DIR/all_filtered.parquet"
NO_EASY_NO_HARD_STATS="${NO_EASY_NO_HARD_ALL_FILTERED%.parquet}_stats.json"
mkdir -p "$NO_EASY_NO_HARD_DIR" "$NO_EASY_DIR"

VLLM_EAGER_ARGS=()
if [ "$VLLM_ENFORCE_EAGER" = "1" ]; then
    VLLM_EAGER_ARGS=(--enforce-eager)
fi

echo "Node: $(hostname)"
echo "Working dir: $(pwd)"
echo "Python: $(which python)"
echo "Filter max workers: $FILTER_MAX_WORKERS"
echo "Filter max tokens: $FILTER_MAX_TOKENS"
echo "Filter rollout chunk size: $FILTER_ROLLOUT_CHUNK_SIZE"
echo "Filter judge chunk workers: $FILTER_JUDGE_CHUNK_WORKERS"
echo "Filter judge margin threshold: $FILTER_JUDGE_MARGIN_THRESHOLD"
echo "Filter eval per section: $FILTER_EVAL_PER_SECTION"
echo "Input parquet: $INPUT_PARQUET"
echo "No easy + no hard parquet: $NO_EASY_NO_HARD_ALL_FILTERED"
echo "No easy parquet: $NO_EASY_ALL_FILTERED"
echo "vLLM max num seqs: $VLLM_MAX_NUM_SEQS"
echo "vLLM max batched tokens: $VLLM_MAX_NUM_BATCHED_TOKENS"
echo "vLLM enforce eager: $VLLM_ENFORCE_EAGER"
nvidia-smi

start_ts=$(date +%s)
echo "Start time: $(date)"
POLICY_PID=""
JUDGE_PID=""

finish() {
  for pid in "${POLICY_PID:-}" "${JUDGE_PID:-}"; do
    if [ -n "$pid" ]; then
      echo "Stopping vLLM pid $pid ..."
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
  end_ts=$(date +%s)
  echo "End time: $(date)"
  echo "Elapsed: $((end_ts - start_ts)) seconds"
}
trap finish EXIT

# 1. Preprocess: QA JSONL -> single all.parquet (no split yet)
# echo "Preprocessing data (no split, QA only)..."
# PYTHONUNBUFFERED=1 python src/data_preprocess_verl_mixed.py \
#     --qa_input data/processed/pragmaticQA_qwen3_8b_cycle1_audited.jsonl \
#     --output data/verl-mixed-RSA8B-final \
#     --no_split

# 2a. Start vLLM POLICY server (SFT model) on $POLICY_PORT
# Sequential startup: judge waits until policy finished profiling, so its own
# memory profile sees a stable baseline instead of racing the policy's load.
echo "Starting vLLM POLICY server with SFT model: $POLICY_MODEL on port $POLICY_PORT"
PYTHONUNBUFFERED=1 python -m vllm.entrypoints.openai.api_server \
    --model "$POLICY_MODEL" \
    --host 0.0.0.0 \
    --port $POLICY_PORT \
    --served-model-name policy \
    --dtype bfloat16 \
    --trust-remote-code \
    --max-model-len 4096 \
    --gpu-memory-utilization "$POLICY_GPU_MEMORY_UTIL" \
    --max-num-seqs "$VLLM_MAX_NUM_SEQS" \
    --max-num-batched-tokens "$VLLM_MAX_NUM_BATCHED_TOKENS" \
    --enable-prefix-caching \
    "${VLLM_EAGER_ARGS[@]}" \
    --disable-log-requests > vllm_policy_${SLURM_JOB_ID}.log 2>&1 &
POLICY_PID=$!

echo "Waiting for vLLM POLICY server to be ready before starting JUDGE..."
MAX_WAIT=900
ELAPSED=0
policy_ready=0
while [ $policy_ready -eq 0 ]; do
    if curl -s "http://localhost:$POLICY_PORT/v1/models" > /dev/null 2>&1; then
        echo "  policy server ready (after ~${ELAPSED}s)"
        policy_ready=1
        break
    fi
    if ! kill -0 "$POLICY_PID" 2>/dev/null; then
        echo "ERROR: policy server process died. See vllm_policy_${SLURM_JOB_ID}.log"
        exit 1
    fi
    if [ $ELAPSED -ge $MAX_WAIT ]; then
        echo "ERROR: policy server did not start within ${MAX_WAIT}s."
        echo "  See vllm_policy_${SLURM_JOB_ID}.log"
        exit 1
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
done

# 2b. Start vLLM JUDGE server (base model) on $JUDGE_PORT
echo "Starting vLLM JUDGE server with base model: $BASE_MODEL on port $JUDGE_PORT"
PYTHONUNBUFFERED=1 python -m vllm.entrypoints.openai.api_server \
    --model "$BASE_MODEL" \
    --host 0.0.0.0 \
    --port $JUDGE_PORT \
    --served-model-name judge \
    --dtype bfloat16 \
    --trust-remote-code \
    --max-model-len 4096 \
    --gpu-memory-utilization "$JUDGE_GPU_MEMORY_UTIL" \
    --max-num-seqs "$VLLM_MAX_NUM_SEQS" \
    --max-num-batched-tokens "$VLLM_MAX_NUM_BATCHED_TOKENS" \
    --enable-prefix-caching \
    "${VLLM_EAGER_ARGS[@]}" \
    --disable-log-requests > vllm_judge_${SLURM_JOB_ID}.log 2>&1 &
JUDGE_PID=$!

# 3. Wait for judge server to be ready
echo "Waiting for vLLM JUDGE server to be ready..."
ELAPSED=0
judge_ready=0
while [ $judge_ready -eq 0 ]; do
    if curl -s "http://localhost:$JUDGE_PORT/v1/models" > /dev/null 2>&1; then
        echo "  judge  server ready (after ~${ELAPSED}s)"
        judge_ready=1
        break
    fi
    if ! kill -0 "$JUDGE_PID" 2>/dev/null; then
        echo "ERROR: judge server process died. See vllm_judge_${SLURM_JOB_ID}.log"
        exit 1
    fi
    if [ $ELAPSED -ge $MAX_WAIT ]; then
        echo "ERROR: judge server did not start within ${MAX_WAIT}s."
        echo "  See vllm_judge_${SLURM_JOB_ID}.log"
        exit 1
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
done
echo "Both vLLM servers ready (took ~${ELAPSED}s)."

# 4. Run the filtering script once, dropping both easy and hard prompts.
echo "Running the filtering script on all samples (drop easy + hard)..."
PYTHONUNBUFFERED=1 python src/filter_prompts.py \
    --input "$INPUT_PARQUET" \
    --output "$NO_EASY_NO_HARD_ALL_FILTERED" \
    --base_url "http://localhost:$POLICY_PORT" \
    --model_name policy \
    --judge_base_url "http://localhost:$JUDGE_PORT" \
    --judge_model_name judge \
    --n_samples 8 \
    --rollout_chunk_size "$FILTER_ROLLOUT_CHUNK_SIZE" \
    --judge_chunk_workers "$FILTER_JUDGE_CHUNK_WORKERS" \
    --max_workers "$FILTER_MAX_WORKERS" \
    --max_tokens "$FILTER_MAX_TOKENS" \
    --filter_hard \
    --judge_margin_threshold "$FILTER_JUDGE_MARGIN_THRESHOLD"

# 5. Shut down both servers
# `|| true` is required: `wait $pid` returns 128+SIGTERM (=143) for a killed
# process, and `set -e` would otherwise abort the script before steps 6/7 run
# (this is what previously left no_easy/ empty even though step 4 succeeded).
echo "Filtering complete. Shutting down vLLM servers..."
kill $POLICY_PID $JUDGE_PID 2>/dev/null || true
wait $POLICY_PID 2>/dev/null || true
wait $JUDGE_PID 2>/dev/null || true
POLICY_PID=""
JUDGE_PID=""

# 6. Derive the easy-only variant from the same filter stats.
echo "Building easy-only filtered parquet from $NO_EASY_NO_HARD_STATS..."
PYTHONUNBUFFERED=1 python src/build_easy_only_from_filter_stats.py \
    --input "$INPUT_PARQUET" \
    --stats "$NO_EASY_NO_HARD_STATS" \
    --output "$NO_EASY_ALL_FILTERED"

# 7. Split both filtered datasets into train/val.
echo "Splitting no_easy_no_hard samples into train/val..."
PYTHONUNBUFFERED=1 python src/split_filtered_parquet.py \
    --input "$NO_EASY_NO_HARD_ALL_FILTERED" \
    --output "$NO_EASY_NO_HARD_DIR" \
    --eval_per_section "$FILTER_EVAL_PER_SECTION"

echo "Splitting no_easy samples into train/val..."
PYTHONUNBUFFERED=1 python src/split_filtered_parquet.py \
    --input "$NO_EASY_ALL_FILTERED" \
    --output "$NO_EASY_DIR" \
    --eval_per_section "$FILTER_EVAL_PER_SECTION"

echo "Done!"
