#!/bin/bash
# vllm OpenAI 兼容服务 - Sinong1.0-8B
# 用法: bash start_vllm.sh [模型路径] [端口] [GPU数量]

set -e

MODEL_PATH="${1:-/app/models/Sinong1.0-8B}"
PORT="${2:-8000}"
TP_SIZE="${3:-1}"

echo "============================================"
echo " Sinong1.0-8B vLLM OpenAI 服务"
echo " 模型: ${MODEL_PATH}"
echo " 端口: ${PORT}"
echo " 张量并行: ${TP_SIZE}"
echo "============================================"

vllm serve "${MODEL_PATH}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --gpu-memory-utilization 0.9 \
    --max-model-len 40960 \
    --dtype bfloat16 \
    --trust-remote-code \
    --enable-reasoning \
    --reasoning-parser deepseek_r1
