#!/usr/bin/env python3
"""
流式输出示例
展示如何在 Python 中实现流式调用，并处理 SSE 事件
"""

import requests
import json
from typing import Generator


def stream_chat(
    base_url: str = "http://localhost:8000",
    model: str = "Sinong1.0-8B",
    message: str = "请详细介绍水稻的种植流程，包括选种、育秧、插秧、田间管理、收割等环节。",
    max_tokens: int = 1000,
    temperature: float = 0.6,
) -> Generator[str, None, None]:
    """
    流式调用聊天接口

    Yields:
        str: 每个 chunk 的内容
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": message}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }

    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json=payload,
        stream=True,
        timeout=60,
    )

    if response.status_code != 200:
        raise Exception(f"请求失败: HTTP {response.status_code} - {response.text}")

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
                        yield content
                except json.JSONDecodeError:
                    continue


def main():
    """主函数：演示流式输出"""
    print("=" * 60)
    print("  流式输出示例")
    print("=" * 60)
    print("问题：请详细介绍水稻的种植流程")
    print("-" * 60)
    print("回答：", end="", flush=True)

    full_response = ""
    chunk_count = 0

    try:
        for chunk in stream_chat():
            print(chunk, end="", flush=True)
            full_response += chunk
            chunk_count += 1
    except Exception as e:
        print(f"\n错误: {e}")
        return

    print("\n" + "-" * 60)
    print(f"接收完成，共 {chunk_count} 个 chunk")
    print(f"总字符数: {len(full_response)}")


def demo_with_langchain():
    """使用 LangChain 的流式调用示例"""
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage

    print("\n" + "=" * 60)
    print("  LangChain 流式输出示例")
    print("=" * 60)

    # 创建 LLM 实例
    llm = ChatOpenAI(
        model="Sinong1.0-8B",
        base_url="http://localhost:8000/v1",
        api_key="not-needed",
        temperature=0.6,
        max_tokens=500,
        streaming=True,
    )

    print("问题：小麦赤霉病怎么防治？")
    print("-" * 60)
    print("回答：", end="", flush=True)

    # 流式调用
    for chunk in llm.stream([HumanMessage(content="小麦赤霉病怎么防治？")]):
        print(chunk.content, end="", flush=True)

    print()


if __name__ == "__main__":
    # 运行原生 requests 示例
    main()

    # 运行 LangChain 示例（需要安装 langchain）
    try:
        demo_with_langchain()
    except ImportError:
        print("\n提示：安装 langchain 以运行 LangChain 示例")
        print("pip install langchain langchain-openai")
