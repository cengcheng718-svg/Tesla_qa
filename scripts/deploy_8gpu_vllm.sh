#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export RAG_MAX_INFLIGHT_REQUESTS="${RAG_MAX_INFLIGHT_REQUESTS:-8}"

echo "Starting Tesla RAG stack with ${VLLM_TENSOR_PARALLEL_SIZE}-GPU tensor parallel vLLM..."
docker compose -f docker-compose.yml -f deploy/docker-compose.8gpu.yml up -d --build
docker compose -f docker-compose.yml -f deploy/docker-compose.8gpu.yml ps
