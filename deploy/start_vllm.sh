#!/bin/bash
# vllm OpenAI 服务部署脚本
# 启动 Sinong1.0-8B 模型服务

set -e

MODEL_PATH="${1:-../models/Sinong1.0-8B}"
PORT="${2:-8000}"

echo "启动 Sinong1.0-8B 服务"
echo "模型路径: $MODEL_PATH"
echo "端口: $PORT"
echo ""

vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --gpu-memory-utilization 0.9 \
    --max-model-len 40960 \
    --dtype bfloat16 \
    --trust-remote-code
