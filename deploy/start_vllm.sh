#!/bin/bash

# ============================================================
# vllm 原生 OpenAI 服务启动脚本
# 使用 vllm serve 命令直接启动 OpenAI 兼容的 API 服务
# ============================================================

set -e

# 默认配置
MODEL_PATH="${MODEL_PATH:-./models/Sinong1.0-8B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  Sinong1.0-8B vllm OpenAI 服务启动${NC}"
echo -e "${GREEN}============================================================${NC}"
echo -e "${YELLOW}模型路径:${NC} $MODEL_PATH"
echo -e "${YELLOW}监听地址:${NC} $HOST:$PORT"
echo -e "${YELLOW}GPU 显存利用率:${NC} $GPU_MEMORY_UTILIZATION"
echo -e "${YELLOW}最大上下文长度:${NC} $MAX_MODEL_LEN"
echo -e "${YELLOW}Tensor 并行数:${NC} $TENSOR_PARALLEL_SIZE"
echo -e "${GREEN}============================================================${NC}"

# 检查模型路径是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo -e "${RED}错误: 模型路径不存在: $MODEL_PATH${NC}"
    exit 1
fi

# 检查 vllm 是否安装
if ! command -v vllm &> /dev/null; then
    echo -e "${RED}错误: vllm 未安装，请运行: pip install vllm${NC}"
    exit 1
fi

# 启动 vllm serve
echo -e "${GREEN}正在启动 vllm serve...${NC}"

exec vllm serve "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --trust-remote-code \
    --dtype bfloat16 \
    --enforce-eager \
    --chat-template "$MODEL_PATH/tokenizer_config.json"
