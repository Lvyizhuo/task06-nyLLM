#!/bin/bash
# vllm OpenAI 兼容服务 - Sinong1.0-32B
# 用法: bash start_vllm.sh [模型路径] [端口] [TP并行数]
#
# 示例:
#   bash start_vllm.sh                                    # 默认: /app/models/Sinong1.0-32B:8000 TP=2
#   bash start_vllm.sh /data/models/Sinong1.0-32B 8000 4  # 自定义路径 + 4卡并行
#   bash start_vllm.sh /app/models/Sinong1.0-32B 8000 2   # 2卡并行（最低推荐）

set -e

MODEL_PATH="${1:-/app/models/Sinong1.0-32B}"
PORT="${2:-8000}"
TP_SIZE="${3:-2}"

echo "============================================"
echo " Sinong1.0-32B vLLM OpenAI 服务"
echo " 模型: ${MODEL_PATH}"
echo " 端口: ${PORT}"
echo " 张量并行: ${TP_SIZE} (4xA100-40GB 推荐 TP=2 或 TP=4)"
echo "============================================"

vllm serve "${MODEL_PATH}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192 \
    --dtype bfloat16 \
    --trust-remote-code
