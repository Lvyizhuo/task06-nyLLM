#!/bin/bash

# ============================================================
# 快速启动脚本
# 根据环境自动选择启动方式
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  Sinong1.0-8B 服务启动${NC}"
echo -e "${GREEN}============================================================${NC}"

# 检查模型路径
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Sinong1.0-8B}"

if [ ! -d "$MODEL_PATH" ]; then
    echo -e "${RED}错误: 模型路径不存在: $MODEL_PATH${NC}"
    echo -e "${YELLOW}请设置 MODEL_PATH 环境变量或下载模型到 models/Sinong1.0-8B${NC}"
    exit 1
fi

# 检查 GPU
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}检测到 NVIDIA GPU${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo -e "${YELLOW}未检测到 nvidia-smi，将尝试使用 CPU 模式${NC}"
fi

# 检查 vllm 是否安装
if python -c "import vllm" 2>/dev/null; then
    echo -e "${GREEN}vllm 已安装，使用原生 OpenAI 服务${NC}"
    echo -e "${YELLOW}启动命令: python start_vllm.py --model-path $MODEL_PATH${NC}"
    echo ""

    # 启动 vllm 服务
    cd "$SCRIPT_DIR"
    python start_vllm.py --model-path "$MODEL_PATH" "$@"
else
    echo -e "${YELLOW}vllm 未安装，使用 FastAPI 服务${NC}"
    echo -e "${YELLOW}建议安装 vllm 以获得更好的性能: pip install vllm${NC}"
    echo -e "${YELLOW}启动命令: python server.py --model-path $MODEL_PATH --backend transformers${NC}"
    echo ""

    # 启动 FastAPI 服务
    cd "$SCRIPT_DIR"
    python server.py --model-path "$MODEL_PATH" --backend transformers "$@"
fi
