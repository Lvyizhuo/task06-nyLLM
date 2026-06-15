#!/usr/bin/env python3
"""
vllm OpenAI 服务启动脚本
使用模型默认参数启动 Sinong1.0-8B 服务
"""

import sys
import subprocess
from pathlib import Path


def check_environment():
    """检查运行环境"""
    try:
        import vllm
        print(f"✓ vllm {vllm.__version__}")
    except ImportError:
        print("✗ vllm 未安装: pip install vllm")
        sys.exit(1)

    try:
        import torch
        if not torch.cuda.is_available():
            print("✗ 需要 NVIDIA GPU")
            sys.exit(1)
        gpu_count = torch.cuda.device_count()
        print(f"✓ {gpu_count} GPU(s) 可用")
    except ImportError:
        print("✗ PyTorch 未安装")
        sys.exit(1)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="启动 vllm OpenAI 服务")
    parser.add_argument("--model-path", type=str, default="../models/Sinong1.0-8B", help="模型路径")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU 显存利用率")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor 并行数")
    args = parser.parse_args()

    check_environment()

    model_path = Path(args.model_path).resolve()
    if not model_path.exists():
        print(f"✗ 模型路径不存在: {model_path}")
        sys.exit(1)

    # 使用模型默认参数启动
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(model_path),
        "--host", args.host,
        "--port", str(args.port),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--max-model-len", "40960",  # 模型默认最大上下文
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--trust-remote-code",
        "--dtype", "bfloat16",
    ]

    print(f"\n启动服务: {model_path.name}")
    print(f"地址: {args.host}:{args.port}")
    print(f"命令: {' '.join(cmd)}\n")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n服务已停止")


if __name__ == "__main__":
    main()
