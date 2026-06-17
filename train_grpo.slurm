#!/usr/bin/env bash
# GRPO batch job template (QA-only pragmatic task, Qwen3-8B).
# Nodes 1-4 run policy training/rollout workers; node 5 runs the judge server.

set -eo pipefail

: "${WORK:?Set WORK to a writable work/cache directory}"
: "${SCRATCH:?Set SCRATCH to a scratch/checkpoint directory}"

export REWARD_DEBUG=1
export OMP_NUM_THREADS=8
export MALLOC_ARENA_MAX=4
export MAX_CONCURRENCY=8
export RAY_memory_monitor_refresh_ms=0
export WANDB_FINISH_TIMEOUT=120
export WANDB__SERVICE_WAIT=120

# 1. Environment and paths
export VLLM_USE_V1=0
export VLLM_FLASH_ATTN_VERSION=2
export PRAGREST_REPO_DIR="${PRAGREST_REPO_DIR:-$PWD}"
export VERL_SETUP_DIR="${VERL_SETUP_DIR:?Set VERL_SETUP_DIR to your VERL environment directory}"
export HF_HOME="$WORK/.cache/huggingface"
export WANDB_DIR="$WORK/.cache/wandb"
export PIP_CACHE_DIR="$WORK/.cache/pip"
export HF_TOKEN="${HF_TOKEN:-}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"

if [ -z "$WANDB_API_KEY" ]; then
    echo "ERROR: Set WANDB_API_KEY in the job environment before submitting."
    exit 1
fi

export SFT_MODEL_DIR="${SFT_MODEL_DIR:-$SCRATCH/models/Qwen3-8B-PragReST-SFT}"
export SFT_MODEL_HF="${SFT_MODEL_HF:-}"
export BASE_MODEL="${BASE_MODEL:-$SCRATCH/models/Qwen3-8B}"

# Use the requested SFT checkpoint as the policy model.
if [ -f "$SFT_MODEL_DIR/config.json" ]; then
    echo "Using cached SFT model: $SFT_MODEL_DIR"
    export POLICY_MODEL="$SFT_MODEL_DIR"
elif [ -n "$SFT_MODEL_HF" ]; then
    if [ -z "$HF_TOKEN" ]; then
        echo "ERROR: Set HF_TOKEN in the job environment to download $SFT_MODEL_HF."
        exit 1
    fi
    if huggingface-cli download "$SFT_MODEL_HF" \
            --local-dir "$SFT_MODEL_DIR" \
            --token "$HF_TOKEN" 2>&1 | tee hf_download_${SLURM_JOB_ID}.log; then
        echo "SFT model downloaded to $SFT_MODEL_DIR"
        export POLICY_MODEL="$SFT_MODEL_DIR"
    else
        echo "ERROR: Failed to download requested SFT checkpoint: $SFT_MODEL_HF"
        exit 1
    fi
else
    echo "ERROR: Missing requested SFT checkpoint: $SFT_MODEL_DIR"
    echo "Expected file: $SFT_MODEL_DIR/config.json"
    exit 1
fi

export DATA_DIR="${DATA_DIR:-$PRAGREST_REPO_DIR/data/verl-mixed-qwen-8b}"
export TRAIN_FILE="$DATA_DIR/no_easy/train_filtered.parquet"
export VAL_FILE="$DATA_DIR/no_easy/val_filtered.parquet"
export REWARD_FN_PATH="${REWARD_FN_PATH:-$PRAGREST_REPO_DIR/src/reward_fn.py}"
export PRAG_EVAL_BATCH_SIZE="${PRAG_EVAL_BATCH_SIZE:-169}"
export PRAG_EVAL_MAX_PROMPT_LENGTH="${PRAG_EVAL_MAX_PROMPT_LENGTH:-2048}"
export PRAG_EVAL_THINKING_BUDGET="${PRAG_EVAL_THINKING_BUDGET:-2048}"
export PRAG_EVAL_ANSWER_MAX_NEW_TOKENS="${PRAG_EVAL_ANSWER_MAX_NEW_TOKENS:-64}"
export PRAG_EVAL_MAX_SAMPLES="${PRAG_EVAL_MAX_SAMPLES:-169}"

export OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH/pragrest_grpo/batch_job_${SLURM_JOB_ID}}"
export JUDGE_MODEL_NAME="judge"
export JUDGE_MARGIN_THRESHOLD=0.8
export JUDGE_LOG_PATH="$OUTPUT_DIR/judge_log.jsonl"
JUDGE_PORT=8100

# 2. Load modules and activate the VERL environment
module purge
module load gcc/13.2.0 cuda/12.8

cd "$VERL_SETUP_DIR"
source env_vars.sh
source .venv/bin/activate
export TIKTOKEN_ENCODINGS_BASE="$WORK/encodings"

# 3. Node discovery
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
policy_node2=${nodes_array[1]}
policy_node3=${nodes_array[2]}
policy_node4=${nodes_array[3]}
judge_node=${nodes_array[4]}

head_node_ip=$(getent hosts "$head_node"     | awk '{print $1}')
policy_node2_ip=$(getent hosts "$policy_node2" | awk '{print $1}')
policy_node3_ip=$(getent hosts "$policy_node3" | awk '{print $1}')
policy_node4_ip=$(getent hosts "$policy_node4" | awk '{print $1}')
judge_node_ip=$(getent hosts "$judge_node"   | awk '{print $1}')

echo "Policy node 1: $head_node ($head_node_ip)"
echo "Policy node 2: $policy_node2 ($policy_node2_ip)"
echo "Policy node 3: $policy_node3 ($policy_node3_ip)"
echo "Policy node 4: $policy_node4 ($policy_node4_ip)"
echo "Judge  node:   $judge_node ($judge_node_ip)"

# 4. Cleanup stale processes on all nodes
for node in "$head_node" "$policy_node2" "$policy_node3" "$policy_node4" "$judge_node"; do
    echo "Cleaning up stale processes on $node ..."
    srun --overlap --nodes=1 --ntasks=1 -w "$node" bash -c '
        ray stop --force 2>/dev/null || true
        nvidia-smi --query-compute-apps=pid --format=csv,noheader \
            | xargs -r kill -9 2>/dev/null || true
        pkill -f vllm 2>/dev/null || true
        pkill -f "ray::" 2>/dev/null || true
        rm -f /tmp/vllm* /dev/shm/vllm* 2>/dev/null || true
    '
done
sleep 5

# 5. Judge vLLM server
echo "Starting judge vLLM server on $judge_node:$JUDGE_PORT ..."
VLLM_USE_V1=0 srun --overlap --nodes=1 --ntasks=1 -w "$judge_node" \
    python -m vllm.entrypoints.openai.api_server \
        --model "$BASE_MODEL" \
        --served-model-name "$JUDGE_MODEL_NAME" \
        --host 0.0.0.0 \
        --port $JUDGE_PORT \
        --gpu-memory-utilization 0.8 \
        --max-model-len 2048 \
        --dtype bfloat16 \
        --trust-remote-code \
        --enforce-eager \
        --no-enable-log-requests > vllm_judge_server_${SLURM_JOB_ID}.log 2>&1 &

export JUDGE_BASE_URL="http://$judge_node_ip:$JUDGE_PORT"

echo "Waiting for judge server at $JUDGE_BASE_URL ..."
for i in {1..120}; do
    if curl -sf "$JUDGE_BASE_URL/health" > /dev/null 2>&1; then
        echo "Judge server ready (attempt $i)."
        break
    fi
    [ $i -eq 120 ] && { echo "ERROR: judge server did not start. See vllm_judge_server_${SLURM_JOB_ID}.log"; exit 1; }
    echo "  $i/120 – waiting 5s..."
    sleep 5
done

echo "Judge server GPU sanity check on $judge_node ..."
srun --overlap --nodes=1 --ntasks=1 -w "$judge_node" bash -lc '
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
    echo "Active GPU compute processes:"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader || true
'

# 6. Ray cluster
echo "Starting Ray cluster across policy nodes 1, 2, and 3..."
ray stop --force
export RAY_memory_usage_threshold=0.99

port=6379
export ip_head=$head_node_ip:$port
export RAY_ADDRESS=$ip_head

srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head \
        --node-ip-address="$head_node_ip" \
        --port=$port \
        --num-cpus=64 \
        --num-gpus=1 \
        --block > ray_head_${SLURM_JOB_ID}.log 2>&1 &

echo "Waiting 15s for Ray head to start..."
sleep 15

srun --overlap --nodes=1 --ntasks=1 -w "$policy_node2" \
    ray start \
        --address="$ip_head" \
        --node-ip-address="$policy_node2_ip" \
        --num-cpus=64 \
        --num-gpus=1 \
        --block > ray_worker2_${SLURM_JOB_ID}.log 2>&1 &

srun --overlap --nodes=1 --ntasks=1 -w "$policy_node3" \
    ray start \
        --address="$ip_head" \
        --node-ip-address="$policy_node3_ip" \
        --num-cpus=64 \
        --num-gpus=1 \
        --block > ray_worker3_${SLURM_JOB_ID}.log 2>&1 &

srun --overlap --nodes=1 --ntasks=1 -w "$policy_node4" \
    ray start \
        --address="$ip_head" \
        --node-ip-address="$policy_node4_ip" \
        --num-cpus=64 \
        --num-gpus=1 \
        --block > ray_worker4_${SLURM_JOB_ID}.log 2>&1 &

echo "Waiting 15s for Ray workers to join..."
sleep 15
ray status

# 7. GPU sanity check on head node
echo "Head node GPU state before training:"
nvidia-smi --query-gpu=index,memory.used,memory.free,memory.total --format=csv,noheader
echo "Active GPU processes on head node:"
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader || true

# 8. Training
echo "Final judge health check before training..."
curl -sf "$JUDGE_BASE_URL/health" > /dev/null 2>&1 \
    || { echo "ERROR: judge server not reachable at $JUDGE_BASE_URL. Aborting."; exit 1; }
echo "Judge server confirmed reachable. Starting training..."
cd "$PRAGREST_REPO_DIR"
PYTHONUNBUFFERED=1 VLLM_USE_V1=0 python src/train_grpo.py \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.train_batch_size=128 \
    data.max_prompt_length=512 \
    data.max_response_length=1536 \
    data.filter_overlong_prompts=True \
    data.truncation=left \
    data.return_raw_chat=True \
    "+data.apply_chat_template_kwargs={enable_thinking: true}" \
    custom_reward_function.path="$REWARD_FN_PATH" \
    custom_reward_function.name=compute_score \
    reward_model.num_workers=16 \
    actor_rollout_ref.model.path="$POLICY_MODEL" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.optim.lr=4e-6 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.02 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.offload_policy=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.60 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.max_num_seqs=256 \
    actor_rollout_ref.rollout.max_model_len=2048 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.offload_policy=False \
    trainer.critic_warmup=0 \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=4 \
    trainer.total_epochs=4 \
    trainer.save_freq=1 \
    +trainer.save_start_step=10 \
    trainer.test_freq=0 \
    trainer.val_before_train=False \
    trainer.project_name="${WANDB_PROJECT:-pragrest_grpo}" \
    trainer.experiment_name="${WANDB_RUN_NAME:-qa_only_qwen3_8b_margin${JUDGE_MARGIN_THRESHOLD}_kl_0.02_${SLURM_JOB_ID}}" \
    'trainer.logger=["console", "wandb"]' \
    trainer.default_local_dir="$OUTPUT_DIR"

# 9. Write experiment config summary
mkdir -p "$OUTPUT_DIR"
cat > "$OUTPUT_DIR/config.txt" <<EOF
job_id        = ${SLURM_JOB_ID}
task          = pragmatic_qa_only
sft_model     = ${SFT_MODEL_DIR}
policy_model  = ${POLICY_MODEL}
train_file    = ${TRAIN_FILE}
val_file      = ${VAL_FILE}
reward_fn     = ${REWARD_FN_PATH}
judge_reward  = P(yes) - P(no) > ${JUDGE_MARGIN_THRESHOLD}
lr            = 4e-6
batch_size    = 128
kl_coef       = 0.02
EOF
