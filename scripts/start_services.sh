#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

mkdir -p log
export PYTHONPATH="${PYTHONPATH:-}:$PROJECT_ROOT"

if [ "${START_MONGODB:-0}" = "1" ]; then
  "${MONGODB_BIN:-mongodb-7.0.20/bin/mongod}" \
    --port="${MONGO_PORT:-27017}" \
    --dbpath="${MONGODB_DBPATH:-data/mongodb/data}" \
    --logpath="${MONGODB_LOGPATH:-data/mongodb/log/mongodb.log}" \
    --bind_ip="${MONGO_BIND_IP:-0.0.0.0}" \
    --fork
fi

nohup python src/server/semantic_chunk.py > log/semantic_chunk.log 2>&1 &
echo "semantic chunk service started on :6000"

VLLM_ARGS=(
  serve "${VLLM_MODEL_PATH:-LLaMA-Factory-main/output/qwen3_lora_sft_int4}"
  --host "${VLLM_HOST:-0.0.0.0}"
  --port "${VLLM_PORT:-8000}"
  --max-model-len "${VLLM_MAX_MODEL_LEN:-8192}"
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.7}"
)

if [ -n "${VLLM_SERVED_MODEL_NAME:-}" ]; then
  VLLM_ARGS+=(--served-model-name "${VLLM_SERVED_MODEL_NAME}")
fi

if [ "${VLLM_TENSOR_PARALLEL_SIZE:-1}" != "1" ]; then
  VLLM_ARGS+=(--tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE}")
fi

if [ "${VLLM_TRUST_REMOTE_CODE:-0}" = "1" ]; then
  VLLM_ARGS+=(--trust-remote-code)
fi

nohup vllm "${VLLM_ARGS[@]}" > log/qwen3-7b.log 2>&1 &
echo "vLLM service started on :${VLLM_PORT:-8000}"
