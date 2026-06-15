"""
LangGraph 调用 Sinong1.0-8B 的示例
使用 ChatOpenAI 适配器对接 OpenAI 兼容 API
"""

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool


# ============ 1. 连接本地 Sinong1.0-8B 服务 ============

llm = ChatOpenAI(
    model="Sinong1.0-8B",
    base_url="http://localhost:8000/v1",   # 本地服务地址
    api_key="not-needed",                   # 本地部署不需要真实 key
    temperature=0.6,
    max_tokens=2048,
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
        result = eval(expression)  # 生产环境中请用更安全的方式
        return f"计算结果：{expression} = {result}"
    except Exception as e:
        return f"计算失败：{e}"


# ============ 3. 创建 ReAct 智能体 ============

tools = [search_agriculture_knowledge, get_weather, calculate]

agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt="你是一个专业的农业智能助手，基于司农大语言模型。你可以帮助用户解答农业相关问题，"
           "包括作物种植、病虫害防治、畜牧养殖、农业经济等。请使用中文回答。",
)


# ============ 4. 运行示例 ============

if __name__ == "__main__":
    # 简单对话
    result = agent.invoke({
        "messages": [{"role": "user", "content": "小麦赤霉病怎么防治？"}]
    })
    print("=== 智能体回复 ===")
    print(result["messages"][-1].content)

    # 多轮对话
    result2 = agent.invoke({
        "messages": [
            {"role": "user", "content": "山东地区适合种什么小麦品种？"},
            {"role": "assistant", "content": "山东地区适合种植的小麦品种包括济麦22、良星99等..."},
            {"role": "user", "content": "这些品种的亩产大概是多少？"},
        ]
    })
    print("\n=== 多轮对话回复 ===")
    print(result2["messages"][-1].content)
