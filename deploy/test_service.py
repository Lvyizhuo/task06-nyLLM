#!/usr/bin/env python3
"""
测试 vllm OpenAI 服务的脚本
用于验证服务是否正常工作
"""

import requests
import json
import sys
from typing import Optional


def test_health(base_url: str = "http://localhost:8000") -> bool:
    """测试健康检查接口"""
    print("1. 测试健康检查接口...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ 服务状态: {data.get('status')}")
            print(f"   ✓ 模型: {data.get('model')}")
            return True
        else:
            print(f"   ✗ 健康检查失败: HTTP {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"   ✗ 连接失败: {e}")
        return False


def test_list_models(base_url: str = "http://localhost:8000") -> bool:
    """测试列出模型接口"""
    print("\n2. 测试列出模型接口...")
    try:
        response = requests.get(f"{base_url}/v1/models", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = data.get("data", [])
            print(f"   ✓ 可用模型数量: {len(models)}")
            for model in models:
                print(f"   ✓ 模型: {model.get('id')}")
            return True
        else:
            print(f"   ✗ 获取模型列表失败: HTTP {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"   ✗ 连接失败: {e}")
        return False


def test_chat_completion(
    base_url: str = "http://localhost:8000",
    model: str = "Sinong1.0-8B",
    message: str = "你好，请简单介绍一下自己。",
    max_tokens: int = 100,
) -> bool:
    """测试非流式对话接口"""
    print("\n3. 测试非流式对话接口...")
    try:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": message}],
            "max_tokens": max_tokens,
            "temperature": 0.6,
        }

        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=60,
        )

        if response.status_code == 200:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})

            print(f"   ✓ 响应成功")
            print(f"   ✓ 回复内容: {content[:100]}...")
            print(f"   ✓ Token 使用: prompt={usage.get('prompt_tokens')}, completion={usage.get('completion_tokens')}")
            return True
        else:
            print(f"   ✗ 对话失败: HTTP {response.status_code}")
            print(f"   ✗ 错误信息: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"   ✗ 连接失败: {e}")
        return False


def test_streaming_chat(
    base_url: str = "http://localhost:8000",
    model: str = "Sinong1.0-8B",
    message: str = "请简述水稻的种植流程。",
    max_tokens: int = 200,
) -> bool:
    """测试流式对话接口"""
    print("\n4. 测试流式对话接口...")
    try:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": message}],
            "max_tokens": max_tokens,
            "temperature": 0.6,
            "stream": True,
        }

        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=60,
        )

        if response.status_code == 200:
            print("   ✓ 流式响应开始")
            full_content = ""
            chunk_count = 0

            for line in response.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            delta = data["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                full_content += content
                                chunk_count += 1
                                print(content, end="", flush=True)
                        except json.JSONDecodeError:
                            continue

            print(f"\n   ✓ 流式接收完成，共 {chunk_count} 个 chunk")
            print(f"   ✓ 完整回复: {full_content[:100]}...")
            return True
        else:
            print(f"   ✗ 流式对话失败: HTTP {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"   ✗ 连接失败: {e}")
        return False


def run_all_tests(base_url: str = "http://localhost:8000") -> bool:
    """运行所有测试"""
    print("=" * 60)
    print("  vllm OpenAI 服务测试")
    print("=" * 60)
    print(f"服务地址: {base_url}")
    print("=" * 60)

    results = []

    # 测试健康检查
    results.append(test_health(base_url))

    # 测试列出模型
    results.append(test_list_models(base_url))

    # 测试非流式对话
    results.append(test_chat_completion(base_url))

    # 测试流式对话
    results.append(test_streaming_chat(base_url))

    # 汇总结果
    print("\n" + "=" * 60)
    print("  测试结果汇总")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"通过: {passed}/{total}")

    if passed == total:
        print("✓ 所有测试通过！服务正常工作。")
        return True
    else:
        print("✗ 部分测试失败，请检查服务状态。")
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="测试 vllm OpenAI 服务")
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000",
        help="服务基础地址 (默认: http://localhost:8000)"
    )
    parser.add_argument(
        "--test",
        type=str,
        choices=["health", "models", "chat", "stream", "all"],
        default="all",
        help="运行指定测试 (默认: all)"
    )

    args = parser.parse_args()

    if args.test == "all":
        success = run_all_tests(args.base_url)
    elif args.test == "health":
        success = test_health(args.base_url)
    elif args.test == "models":
        success = test_list_models(args.base_url)
    elif args.test == "chat":
        success = test_chat_completion(args.base_url)
    elif args.test == "stream":
        success = test_streaming_chat(args.base_url)
    else:
        print(f"未知测试: {args.test}")
        sys.exit(1)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
