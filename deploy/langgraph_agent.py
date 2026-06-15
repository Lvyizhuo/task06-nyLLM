"""
LangGraph 智能体调用 vllm 原生 OpenAI 服务
支持流式和非流式两种调用方式
适配 Sinong1.0-8B 模型
"""

from typing import Annotated, TypedDict, Sequence
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
import operator


# ============================================================
# 1. 配置连接参数
# ============================================================

# vllm 原生 OpenAI 服务地址
# 本地调试
BASE_URL = "http://localhost:8000/v1"

# 服务器部署（替换为实际 IP）
# BASE_URL = "http://your-server-ip:8000/v1"

# 模型名称（需要与 vllm serve --served-model-name 一致）
MODEL_NAME = "Sinong1.0-8B"

# API Key（如果 vllm serve 启用了 --api-key）
API_KEY = "not-needed"  # 如果没有设置 API Key，可以使用任意值


# ============================================================
# 2. 创建 LLM 实例
# ============================================================

def create_llm(
    base_url: str = BASE_URL,
    model_name: str = MODEL_NAME,
    api_key: str = API_KEY,
    temperature: float = 0.6,
    max_tokens: int = 2048,
    streaming: bool = False,
) -> ChatOpenAI:
    """创建 ChatOpenAI 实例，连接 vllm 原生服务"""
    return ChatOpenAI(
        model=model_name,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=streaming,
    )


# 非流式 LLM
llm = create_llm(streaming=False)

# 流式 LLM
streaming_llm = create_llm(streaming=True)


# ============================================================
# 3. 定义工具（示例）
# ============================================================

@tool
def search_agriculture_knowledge(query: str) -> str:
    """搜索农业领域知识，包括作物种植、病虫害防治、畜牧养殖等。

    Args:
        query: 搜索关键词
    """
    # 实际项目中替换为你的 RAG 检索或数据库查询
    return f"关于「{query}」的农业知识：这是模拟检索结果，请替换为实际知识库。"


@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息。

    Args:
        city: 城市名称
    """
    # 实际项目中对接天气 API
    return f"{city}今日天气：晴，温度 25°C，适合农事操作。"


@tool
def calculate(expression: str) -> str:
    """计算数学表达式，用于农药配比、面积计算等。

    Args:
        expression: 数学表达式
    """
    try:
        result = eval(expression)
        return f"计算结果：{expression} = {result}"
    except Exception as e:
        return f"计算失败：{e}"


# 工具列表
tools = [search_agriculture_knowledge, get_weather, calculate]

# 创建工具节点
tool_node = ToolNode(tools)


# ============================================================
# 4. 定义 LangGraph 状态和节点
# ============================================================

class AgentState(TypedDict):
    """智能体状态"""
    messages: Annotated[Sequence[BaseMessage], operator.add]


def should_continue(state: AgentState) -> str:
    """判断是否继续调用工具"""
    messages = state["messages"]
    last_message = messages[-1]

    # 如果最后一条消息包含工具调用，则继续
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    # 否则结束
    return END


def call_model(state: AgentState) -> dict:
    """调用 LLM"""
    messages = state["messages"]

    # 添加系统提示词
    system_message = SystemMessage(
        content="你是司农(Sinong)，一个专业的农业智能助手。请直接回答用户的问题，不要提及或引用任何参考资料。"
    )
    messages = [system_message] + list(messages)

    # 绑定工具到 LLM
    llm_with_tools = llm.bind_tools(tools)
    response = llm_with_tools.invoke(messages)

    return {"messages": [response]}


def call_model_streaming(state: AgentState):
    """调用 LLM（流式版本）"""
    messages = state["messages"]

    # 添加系统提示词
    system_message = SystemMessage(
        content="你是司农(Sinong)，一个专业的农业智能助手。请直接回答用户的问题，不要提及或引用任何参考资料。"
    )
    messages = [system_message] + list(messages)

    # 绑定工具到 LLM
    llm_with_tools = streaming_llm.bind_tools(tools)

    # 流式生成
    for chunk in llm_with_tools.stream(messages):
        yield {"messages": [chunk]}


# ============================================================
# 5. 构建 LangGraph 图
# ============================================================

# 创建状态图
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# 设置入口点
workflow.set_entry_point("agent")

# 添加条件边
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END,
    }
)

# 从工具回到 agent
workflow.add_edge("tools", "agent")

# 编译图
agent = workflow.compile()


# ============================================================
# 6. 流式输出图（用于前端流式展示）
# ============================================================

# 创建流式状态图
streaming_workflow = StateGraph(AgentState)

# 添加节点（使用流式版本）
streaming_workflow.add_node("agent", call_model_streaming)
streaming_workflow.add_node("tools", tool_node)

# 设置入口点
streaming_workflow.set_entry_point("agent")

# 添加条件边
streaming_workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END,
    }
)

# 从工具回到 agent
streaming_workflow.add_edge("tools", "agent")

# 编译流式图
streaming_agent = streaming_workflow.compile()


# ============================================================
# 7. 运行示例
# ============================================================

def run_non_streaming_example():
    """非流式调用示例"""
    print("=" * 60)
    print("示例1：非流式 Agent 调用（带工具）")
    print("=" * 60)

    result = agent.invoke({
        "messages": [HumanMessage(content="小麦赤霉病怎么防治？")]
    })

    print(result["messages"][-1].content)


def run_streaming_example():
    """流式调用示例"""
    print("\n" + "=" * 60)
    print("示例2：流式 Agent 调用")
    print("=" * 60)

    for event in streaming_agent.stream({
        "messages": [HumanMessage(content="请简述水稻的种植流程")]
    }):
        for node, value in event.items():
            if node == "agent":
                for msg in value["messages"]:
                    if hasattr(msg, "content") and msg.content:
                        print(msg.content, end="", flush=True)
    print()


def run_direct_llm_example():
    """直接调用 LLM（不经过 Agent）"""
    print("\n" + "=" * 60)
    print("示例3：直接 LLM 对话（非流式）")
    print("=" * 60)

    response = llm.invoke([HumanMessage(content="介绍一下济南？")])
    print(response.content)


def run_direct_streaming_example():
    """直接流式调用 LLM"""
    print("\n" + "=" * 60)
    print("示例4：直接 LLM 流式对话")
    print("=" * 60)

    for chunk in streaming_llm.stream([HumanMessage(content="请简述水稻的种植流程")]):
        print(chunk.content, end="", flush=True)
    print()


def run_multi_turn_example():
    """多轮对话示例"""
    print("\n" + "=" * 60)
    print("示例5：多轮对话")
    print("=" * 60)

    result = agent.invoke({
        "messages": [
            HumanMessage(content="山东地区适合种什么小麦品种？"),
            AIMessage(content="山东地区适合种植的小麦品种包括济麦22、良星99等品种。"),
            HumanMessage(content="这些品种的亩产大概是多少？"),
        ]
    })
    print(result["messages"][-1].content)


if __name__ == "__main__":
    # 运行所有示例
    run_non_streaming_example()
    run_direct_llm_example()
    run_direct_streaming_example()
    run_multi_turn_example()
    # 流式示例需要前端配合，这里注释掉
    # run_streaming_example()
