#!/bin/bash
# ============================================================
# Sinong1.0-8B 服务器部署脚本
# 使用场景：GPU 服务器上一键部署
# ============================================================

set -e

# ---- 配置 ----
MODEL_DIR="/data/models/Sinong1.0-8B"   # 服务器上模型存放路径，按需修改
PORT=8000
GPU=0

echo "========================================="
echo " Sinong1.0-8B 服务部署"
echo "========================================="

# ---- 1. 下载模型（如果不存在）----
if [ ! -d "$MODEL_DIR" ] || [ -z "$(ls -A $MODEL_DIR)" ]; then
    echo "[INFO] 模型目录不存在，开始下载..."
    pip install modelscope
    modelscope download --model NAULLM/Sinong1.0-8B --local_dir "$MODEL_DIR"
else
    echo "[INFO] 模型已存在: $MODEL_DIR"
    du -sh "$MODEL_DIR"
fi

# ---- 2. Docker 部署 ----
echo "[INFO] 使用 Docker Compose 启动服务..."

# 修改 docker-compose 中的模型挂载路径
export MODEL_DIR

# 启动
cd "$(dirname "$0")"
docker compose up -d --build

echo "[INFO] 等待服务启动..."
sleep 10

# 检查
if curl -s http://localhost:${PORT}/health | grep -q "ok"; then
    echo "[SUCCESS] 服务已启动！"
    echo "  API 地址: http://localhost:${PORT}/v1/chat/completions"
    echo "  健康检查: http://localhost:${PORT}/health"
else
    echo "[WAIT] 模型仍在加载中，请稍候..."
    echo "  查看日志: docker compose logs -f"
fi
