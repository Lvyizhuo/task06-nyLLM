# Sinong1.0-8B 部署与接口封装指南

> 模型架构：**Qwen3ForCausalLM**（基于 Qwen3-8B LoRA 微调）| 参数量：**8.2B** | 权重：**~16GB (BF16)**
> 上下文长度：**40960 tokens** | 思考模式：支持（Qwen3 thinking，默认关闭）
> 训练框架：ms-swift + LoRA (rank=8) + DeepSpeed ZeRO-3
> 流式输出：**已支持（SSE）** | 推理后端：**Transformers + vLLM 双后端**
> 日志：**loguru**（控制台彩色 + 文件轮转）

---

## 一、环境信息

### 已创建的 conda 环境

```
环境名: agent
Python: 3.10
路径: /opt/miniconda3/envs/agent
```

### 已安装的核心依赖

| 包 | 版本 | 用途 |
|---|---|---|
| torch | 2.12.0 | 推理引擎 |
| transformers | 5.12.0 | 模型加载与推理 |
| modelscope | 1.37.1 | 模型下载 |
| accelerate | 1.14.0 | 多GPU/设备映射 |
| fastapi | 0.137.0 | API服务框架 |
| uvicorn | 0.49.0 | ASGI服务器 |
| langchain | 1.3.9 | Agent框架 |
| langchain-openai | 1.3.2 | OpenAI兼容适配 |
| langgraph | 1.2.5 | Agent编排 |
| loguru | 0.7.3 | 日志框架 |

### 激活环境

```bash
conda activate agent
```

---

## 二、模型文件

已下载至：
```
/Users/lvyizhuo/project/i/task06-nyLLM/models/Sinong1.0-8B/  (15GB)
```

---

## 三、本地调试（Mac）— Transformers 后端

### 3.1 启动 API 服务

```bash
conda activate agent

# Transformers 后端（Mac/CPU/GPU 均可）
python deploy/server.py \
  --model-path ./models/Sinong1.0-8B \
  --backend transformers \
  --host 0.0.0.0 \
  --port 8000 \
  --device mps    # Mac Apple Silicon 使用 MPS 加速
                  # 如果 MPS 不可用，改用 cpu
```

> **注意**：Mac 上 8B 模型推理速度较慢（CPU约1-3 tok/s，MPS约5-10 tok/s），仅用于调试。
> 生产环境请在 GPU 服务器上部署。

### 3.2 验证服务

```bash
# 健康检查（返回后端信息）
curl http://localhost:8000/health

# 非流式请求（默认关闭思考模式，推荐）
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Sinong1.0-8B",
    "messages": [
      {"role": "user", "content": "小麦赤霉病的防治方法有哪些？"}
    ],
    "temperature": 0.6,
    "max_tokens": 512
  }'

# 流式请求（SSE 格式）
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Sinong1.0-8B",
    "messages": [
      {"role": "user", "content": "小麦赤霉病的防治方法有哪些？"}
    ],
    "temperature": 0.6,
    "max_tokens": 512,
    "stream": true
  }'

# 开启思考模式（谨慎使用，可能导致"根据参考资料"幻觉）
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Sinong1.0-8B",
    "messages": [
      {"role": "user", "content": "小麦赤霉病的防治方法有哪些？"}
    ],
    "temperature": 0.6,
    "max_tokens": 512,
    "enable_thinking": true,
    "include_think": true
  }'
```

### 3.3 LangGraph 调用

```python
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# 连接本地服务
llm = ChatOpenAI(
    model="Sinong1.0-8B",
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
    temperature=0.6,
)

# 创建智能体
agent = create_react_agent(
    model=llm,
    tools=[your_tool_1, your_tool_2],
)

# 调用
result = agent.invoke({
    "messages": [{"role": "user", "content": "你的问题"}]
})
```

详见 `deploy/langgraph_agent.py`。

---

## 四、GPU 服务器部署 — vLLM 后端（推荐生产）

### 4.1 启动参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--backend vllm` | transformers | 使用 vLLM 后端 |
| `--vllm-async` | false | 启用异步引擎（**推荐，流式性能更好**） |
| `--gpu-memory-utilization` | 0.9 | GPU 显存利用率 |
| `--max-model-len` | 40960 | 最大上下文长度 |

### 4.2 裸机部署

```bash
conda activate agent

# vLLM 同步引擎
python deploy/server.py \
  --model-path ./models/Sinong1.0-8B \
  --backend vllm \
  --gpu-memory-utilization 0.9 \
  --max-model-len 40960 \
  --port 8000

# vLLM 异步引擎（推荐，流式输出性能更好）
python deploy/server.py \
  --model-path ./models/Sinong1.0-8B \
  --backend vllm \
  --vllm-async \
  --gpu-memory-utilization 0.9 \
  --max-model-len 40960 \
  --port 8000
```

### 4.3 Docker 部署

```bash
cd deploy/

# 构建并启动（Dockerfile 默认使用 vllm + async）
docker compose up -d --build

# 查看日志
docker compose logs -f

# 停止
docker compose down
```

### 4.4 自定义启动命令

如需使用 Transformers 后端（Docker 内）：

```yaml
# 修改 docker-compose.yml 的 command
command: >
  python3 /app/deploy/server.py
  --model-path /app/models/Sinong1.0-8B
  --backend transformers
  --device auto
  --host 0.0.0.0
  --port 8000
```

### 4.5 多GPU部署

```yaml
# docker-compose.yml
environment:
  - CUDA_VISIBLE_DEVICES=0,1
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 2
          capabilities: [gpu]
```

---

## 五、双后端对比

| 特性 | Transformers | vLLM |
|------|-------------|------|
| **适用场景** | 本地调试、Mac、CPU | 生产环境、GPU 服务器 |
| **推理速度** | 慢（1-10 tok/s） | 快（50-200 tok/s） |
| **流式输出** | TextIteratorStreamer | AsyncEngine 逐 token |
| **批处理** | 单请求 | 连续批处理（continuous batching） |
| **GPU 显存优化** | 无 | PagedAttention |
| **安装复杂度** | 简单 | 需 GPU + CUDA |
| **Mac 支持** | ✅ | ❌ |
| **GPU 要求** | 可选 | 必须（16GB+ VRAM） |

---

## 六、LangGraph 完整接入方案

### 6.1 接口格式

无论用哪种后端（Transformers/vLLM），暴露的都是 **OpenAI 兼容 API**：

```
POST http://<host>:8000/v1/chat/completions   # 支持流式/非流式
GET  http://<host>:8000/v1/models
GET  http://<host>:8000/health                 # 返回后端类型和加载状态
```

### 6.2 LangGraph 接入代码

```python
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

llm = ChatOpenAI(
    model="Sinong1.0-8B",
    base_url="http://<your-server-ip>:8000/v1",
    api_key="not-needed",
    temperature=0.6,
    max_tokens=2048,
    streaming=True,  # 流式输出
)

@tool
def your_tool(query: str) -> str:
    """你的工具描述"""
    return "result"

agent = create_react_agent(model=llm, tools=[your_tool])
result = agent.invoke({"messages": [{"role": "user", "content": "你的问题"}]})
```

### 6.3 多 Agent 编排

```python
from langgraph.graph import StateGraph, MessagesState

def researcher(state: MessagesState):
    return {"messages": [llm.invoke(state["messages"])]}

def analyst(state: MessagesState):
    return {"messages": [llm.invoke(state["messages"])]}

workflow = StateGraph(MessagesState)
workflow.add_node("researcher", researcher)
workflow.add_node("analyst", analyst)
workflow.add_edge("researcher", "analyst")
workflow.set_entry_point("researcher")

app = workflow.compile()
```

---

## 七、常见问题

### Q1: 模型回答"根据参考资料"但没接 RAG？
训练数据偏差导致。已内置系统提示词防止此问题，如需自定义：`--system-prompt "你的提示词"`

### Q2: 流式输出
**已支持**。`"stream": true` 返回 SSE 格式，vLLM 后端推荐加 `--vllm-async` 获得真正的逐 token 流式。

### Q3: Mac 上 MPS 报错？
改为 `--device cpu`，Mac 上 MPS 对 bfloat16 支持有限。

### Q4: GPU 显存不足？
- 减小 `--max-model-len`（如改为 8192）
- 使用 `--gpu-memory-utilization 0.95`
- 使用 4-bit 量化版（需转换 GGUF 格式）

### Q5: vLLM 启动报模型格式错误？
确保使用 vllm >= 0.9.0，Qwen3ForCausalLM 是较新架构。

### Q6: 如何关闭思考模式？
默认已关闭。请求中设置 `"enable_thinking": false`，或在消息中添加 `/no_think`。

### Q7: 思考模式输出乱码/标签？
server.py 已自动解析 `<think!>...</think!>` 和 `<answer>...</answer>` 标签。
- 默认只返回回答内容，思考过程被过滤
- 如需获取思考过程，设置 `"include_think": true`

### Q8: Docker 中 GPU 不可用？
安装 NVIDIA Container Toolkit：
```bash
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Q9: 日志在哪里？
- 控制台：彩色 loguru 输出
- 文件：`deploy/logs/server_YYYY-MM-DD.log`，自动按天轮转，保留 30 天

---

## 八、部署架构图

```
┌──────────────────────────────────────────────────────────────┐
│                        你的应用层                              │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │  LangGraph   │  │  LangChain   │  │  其他客户端        │  │
│  │  Agent 编排  │  │  Chain/Tool  │  │  (curl/SDK)       │  │
│  └──────┬───────┘  └──────┬───────┘  └───────┬───────────┘  │
│         └─────────────────┼──────────────────┘              │
│                           │ OpenAI 兼容 API                  │
│                           ▼                                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         Sinong1.0-8B API Server (FastAPI)            │   │
│  │  ┌────────────────┐    ┌───────────────────────┐    │   │
│  │  │ Transformers   │    │ vLLM AsyncEngine      │    │   │
│  │  │ (本地调试)      │    │ (生产加速+流式)        │    │   │
│  │  └───────┬────────┘    └───────────┬───────────┘    │   │
│  │          └─────────┬───────────────┘                │   │
│  │                    ▼                                 │   │
│  │          Qwen3ForCausalLM (8.2B)                    │   │
│  └──────────────────────────────────────────────────────┘   │
│                           │                                  │
│                           ▼                                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Docker 容器 / 裸机   |   NVIDIA GPU (16GB+ VRAM)    │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

---

## 九、启动命令速查

```bash
# Mac 本地调试
python deploy/server.py --model-path ./models/Sinong1.0-8B --backend transformers --device mps

# GPU 服务器 - Transformers 后端
python deploy/server.py --model-path ./models/Sinong1.0-8B --backend transformers --device auto

# GPU 服务器 - vLLM 同步引擎
python deploy/server.py --model-path ./models/Sinong1.0-8B --backend vllm

# GPU 服务器 - vLLM 异步引擎（生产推荐）
python deploy/server.py --model-path ./models/Sinong1.0-8B --backend vllm --vllm-async

# 自定义系统提示词
python deploy/server.py --model-path ./models/Sinong1.0-8B --backend vllm --vllm-async --system-prompt "你是一个农业专家"

# Docker
cd deploy && docker compose up -d --build
```
