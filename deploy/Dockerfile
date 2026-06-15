FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# 避免交互式安装
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 安装系统依赖（含 curl 用于健康检查）
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    git \
    git-lfs \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 设置 Python3.10 为默认
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# 安装 Python 依赖 - PyTorch (CUDA 12.4)
RUN pip3 install --no-cache-dir --break-system-packages \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 安装 Python 依赖 - 推理和 API
RUN pip3 install --no-cache-dir --break-system-packages \
    "transformers>=4.51.0" \
    accelerate \
    safetensors \
    sentencepiece \
    tiktoken \
    fastapi \
    uvicorn \
    pydantic \
    modelscope \
    loguru

# 安装 vLLM（GPU 推理加速）
RUN pip3 install --no-cache-dir --break-system-packages \
    "vllm>=0.9.0"

# 创建目录
RUN mkdir -p /app/models /app/deploy /app/logs

# 复制服务代码
COPY deploy/server.py /app/deploy/server.py

# 工作目录
WORKDIR /app

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 默认使用 vLLM 异步引擎启动（生产推荐）
# 本地调试可改用 --backend transformers --device auto
CMD ["python3", "/app/deploy/server.py", \
     "--model-path", "/app/models/Sinong1.0-8B", \
     "--backend", "vllm", \
     "--vllm-async", \
     "--host", "0.0.0.0", \
     "--port", "8000"]
