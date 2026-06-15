"""
LangGraph 调用 Sinong1.0-8B 的示例
使用 ChatOpenAI 适配器对接 OpenAI 兼容 API
支持流式和非流式两种调用方式
"""

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage


# ============ 1. 连接 Sinong1.0-8B 服务 ============

# 本地调试
BASE_URL = "http://localhost:8000/v1"

# 服务器部署（替换为实际 IP）
# BASE_URL = "http://your-server-ip:8000/v1"

llm = ChatOpenAI(
    model="Sinong1.0-8B",
    base_url=BASE_URL,
    api_key="not-needed",           # 本地部署不需要真实 key
    temperature=0.6,
    max_tokens=2048,
    # streaming=True,               # 如需流式，取消注释
)


# ============ 2. 定义工具（示例） ============

@tool
def search_agriculture_knowledge(query: str) -> str:
    """搜索农业领域知识，包括作物种植、病虫害防治、畜牧养殖等。"""
    # 实际项目中替换为你的 RAG 检索或数据库查询
    return f"关于「{query}」的农业知识：这是模拟检索结果，请替换为实际知识库。"


@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息。"""
    # 实际项目中对接天气 API
    return f"{city}今日天气：晴，温度 25°C，适合农事操作。"


@tool
def calculate(expression: str) -> str:
    """计算数学表达式，用于农药配比、面积计算等。"""
    try:
        result = eval(expression)
        return f"计算结果：{expression} = {result}"
    except Exception as e:
        return f"计算失败：{e}"


# ============ 3. 创建 ReAct 智能体 ============

tools = [search_agriculture_knowledge, get_weather, calculate]

agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt="你是司农(Sinong)，一个专业的农业智能助手。请直接回答用户问题，不要提及参考资料。",
)


# ============ 4. 运行示例 ============

if __name__ == "__main__":
    # ---- 示例1：直接用 LLM 对话（不经过 Agent）----
    print("=" * 60)
    print("示例1：直接 LLM 对话")
    print("=" * 60)

    response = llm.invoke([HumanMessage(content="介绍一下济南？")])
    print(response.content)

    # ---- 示例2：通过 Agent 调用（带工具）----
    print("\n" + "=" * 60)
    print("示例2：Agent 调用（带工具）")
    print("=" * 60)

    result = agent.invoke({
        "messages": [{"role": "user", "content": "小麦赤霉病怎么防治？"}]
    })
    print(result["messages"][-1].content)

    # ---- 示例3：多轮对话 ----
    print("\n" + "=" * 60)
    print("示例3：多轮对话")
    print("=" * 60)

    result2 = agent.invoke({
        "messages": [
            {"role": "user", "content": "山东地区适合种什么小麦品种？"},
            {"role": "assistant", "content": "山东地区适合种植的小麦品种包括济麦22、良星99等品种。"},
            {"role": "user", "content": "这些品种的亩产大概是多少？"},
        ]
    })
    print(result2["messages"][-1].content)

    # ---- 示例4：流式输出 ----
    print("\n" + "=" * 60)
    print("示例4：流式输出")
    print("=" * 60)

    streaming_llm = ChatOpenAI(
        model="Sinong1.0-8B",
        base_url=BASE_URL,
        api_key="not-needed",
        temperature=0.6,
        streaming=True,
    )

    for chunk in streaming_llm.stream([HumanMessage(content="请简述水稻的种植流程")]):
        print(chunk.content, end="", flush=True)
    print()
