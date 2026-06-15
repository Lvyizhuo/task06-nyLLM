#!/usr/bin/env python3
"""
vllm 原生 OpenAI 服务启动脚本（Python 版本）
提供更灵活的配置选项和启动参数管理
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def check_requirements():
    """检查依赖是否满足"""
    try:
        import vllm
        print(f"✓ vllm 版本: {vllm.__version__}")
    except ImportError:
        print("✗ vllm 未安装，请运行: pip install vllm")
        sys.exit(1)

    try:
        import torch
        if not torch.cuda.is_available():
            print("✗ CUDA 不可用，vllm 需要 NVIDIA GPU")
            sys.exit(1)
        print(f"✓ CUDA 可用，GPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    except ImportError:
        print("✗ PyTorch 未安装")
        sys.exit(1)


def start_vllm_server(args):
    """启动 vllm serve 服务"""
    model_path = Path(args.model_path).resolve()

    if not model_path.exists():
        print(f"✗ 模型路径不存在: {model_path}")
        sys.exit(1)

    # 构建启动命令
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(model_path),
        "--host", args.host,
        "--port", str(args.port),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--max-model-len", str(args.max_model_len),
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--trust-remote-code",
        "--dtype", args.dtype,
    ]

    # 可选参数
    if args.enforce_eager:
        cmd.append("--enforce-eager")

    if args.chat_template:
        cmd.extend(["--chat-template", args.chat_template])

    if args.served_model_name:
        cmd.extend(["--served-model-name", args.served_model_name])

    if args.api_key:
        cmd.extend(["--api-key", args.api_key])

    # 打印启动信息
    print("\n" + "=" * 60)
    print("  Sinong1.0-8B vllm OpenAI 服务启动")
    print("=" * 60)
    print(f"模型路径: {model_path}")
    print(f"监听地址: {args.host}:{args.port}")
    print(f"GPU 显存利用率: {args.gpu_memory_utilization}")
    print(f"最大上下文长度: {args.max_model_len}")
    print(f"Tensor 并行数: {args.tensor_parallel_size}")
    print(f"数据类型: {args.dtype}")
    print(f"Chat 模板: {args.chat_template or '自动检测'}")
    print(f"模型名称: {args.served_model_name or model_path.name}")
    print("=" * 60)
    print(f"\n启动命令: {' '.join(cmd)}\n")

    # 启动服务
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n服务已停止")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ 服务启动失败: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="启动 vllm 原生 OpenAI 服务",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本启动
  python start_vllm.py --model-path ./models/Sinong1.0-8B

  # 自定义配置
  python start_vllm.py --model-path ./models/Sinong1.0-8B --port 8001 --gpu-memory-utilization 0.8

  # 多 GPU 并行
  python start_vllm.py --model-path ./models/Sinong1.0-8B --tensor-parallel-size 2

  # 启用 API Key 认证
  python start_vllm.py --model-path ./models/Sinong1.0-8B --api-key your-secret-key
        """
    )

    # 必需参数
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="模型本地路径"
    )

    # 服务配置
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址 (默认: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="监听端口 (默认: 8000)")
    parser.add_argument("--api-key", type=str, default=None, help="API Key 认证 (可选)")

    # 模型配置
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU 显存利用率 (默认: 0.9)")
    parser.add_argument("--max-model-len", type=int, default=40960, help="最大上下文长度 (默认: 40960)")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor 并行数 (默认: 1)")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "auto"], help="数据类型 (默认: bfloat16)")
    parser.add_argument("--enforce-eager", action="store_true", help="强制使用 eager 模式 (调试用)")

    # 模型名称和模板
    parser.add_argument("--served-model-name", type=str, default=None, help="对外提供的模型名称 (默认: 使用目录名)")
    parser.add_argument("--chat-template", type=str, default=None, help="Chat 模板文件路径 (默认: 自动检测)")

    args = parser.parse_args()

    # 检查依赖
    check_requirements()

    # 启动服务
    start_vllm_server(args)


if __name__ == "__main__":
    main()
