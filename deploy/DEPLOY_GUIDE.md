# Sinong1.0-8B 部署与接口封装指南

> 模型架构：**Qwen3ForCausalLM** | 参数量：**8.2B** | 权重：**~16GB (BF16)**
> 上下文长度：**40960 tokens** | 思考模式：支持（Qwen3 thinking）

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

文件结构：
```
Sinong1.0-8B/
├── config.json                          # Qwen3ForCausalLM, hidden_size=4096, 36层
├── generation_config.json               # temperature=0.6, top_k=20, top_p=0.95
├── model-00001-of-00004.safetensors     # 4.57GB
├── model-00002-of-00004.safetensors     # 4.91GB
├── model-00003-of-00004.safetensors     # 4.98GB
├── model-00004-of-00004.safetensors     # 1.58GB
├── model.safetensors.index.json
├── tokenizer.json
├── tokenizer_config.json
├── vocab.json & merges.txt
└── ...
```

---

## 三、本地调试（Mac）

### 3.1 启动 API 服务

```bash
conda activate agent

python deploy/server.py \
  --model-path ./models/Sinong1.0-8B \
  --host 0.0.0.0 \
  --port 8000 \
  --device mps    # Mac Apple Silicon 使用 MPS 加速
                  # 如果 MPS 不可用，改用 cpu
```

> **注意**：Mac 上 8B 模型推理速度较慢（CPU约1-3 tok/s，MPS约5-10 tok/s），仅用于调试。
> 生产环境请在 GPU 服务器上部署。

### 3.2 验证服务

```bash
# 健康检查
curl http://localhost:8000/health

# 模型列表
curl http://localhost:8000/v1/models

# 聊天请求
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Sinong1.0-8B",
    "messages": [
      {"role": "user", "content": "小麦赤霉病的防治方法有哪些？"}
    ],
    "temperature": 0.6,
    "max_tokens": 512,
    "enable_thinking": true
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

## 四、服务器部署（GPU + Docker）

### 4.1 前提条件

- GPU 服务器：至少 **1x NVIDIA GPU，16GB+ 显存**（如 A10、V100、A100）
- Docker + NVIDIA Container Toolkit
- 磁盘空间：30GB+

### 4.2 上传模型到服务器

```bash
# 方式一：在服务器上直接下载
pip install modelscope
modelscope download --model NAULLM/Sinong1.0-8B --local_dir /data/models/Sinong1.0-8B

# 方式二：从本机 scp 传输
scp -r ./models/Sinong1.0-8B user@server:/data/models/Sinong1.0-8B
```

### 4.3 Docker 部署

```bash
cd deploy/

# 构建并启动
docker compose up -d --build

# 查看日志
docker compose logs -f

# 停止
docker compose down
```

### 4.4 修改模型路径

编辑 `docker-compose.yml` 中的 volumes 挂载路径：

```yaml
volumes:
  - /data/models/Sinong1.0-8B:/app/models/Sinong1.0-8B:ro
```

### 4.5 多GPU部署

如果需要指定特定GPU，修改 `docker-compose.yml`：

```yaml
environment:
  - CUDA_VISIBLE_DEVICES=0,1   # 使用GPU 0和1
```

---

## 五、vLLM 高性能部署（推荐生产使用）

> vLLM 仅支持 NVIDIA GPU，不支持 Mac。

### 5.1 安装 vLLM

```bash
conda activate agent
pip install vllm>=0.9.0
```

### 5.2 启动 vLLM 服务

```bash
# 设置国内源
export VLLM_USE_MODELSCOPE=false

# 启动服务
python -m vllm.entrypoints.openai.api_server \
  --model ./models/Sinong1.0-8B \
  --served-model-name Sinong1.0-8B \
  --port 8000 \
  --max-model-len 40960 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.9 \
  --enable-reasoning \
  --reasoning-parser qwen3
```

> **关键参数说明**：
> - `--max-model-len 40960`：与模型 config.json 中 max_position_embeddings 一致
> - `--enable-reasoning`：启用 Qwen3 思考模式
> - `--reasoning-parser qwen3`：使用 Qwen3 推理解析器
> - `--gpu-memory-utilization 0.9`：GPU显存利用率

### 5.3 vLLM Docker 部署

```bash
docker run -d \
  --name sinong-vllm \
  --gpus all \
  -v /data/models/Sinong1.0-8B:/app/models/Sinong1.0-8B \
  -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model /app/models/Sinong1.0-8B \
  --served-model-name Sinong1.0-8B \
  --port 8000 \
  --max-model-len 40960 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.9 \
  --enable-reasoning \
  --reasoning-parser qwen3
```

---

## 六、LangGraph 完整接入方案

### 6.1 接口格式

无论用哪种方式部署（FastAPI/vLLM），最终暴露的都是 **OpenAI 兼容 API**：

```
POST http://<host>:8000/v1/chat/completions
GET  http://<host>:8000/v1/models
GET  http://<host>:8000/health
```

### 6.2 LangGraph 接入代码

```python
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

# ============ 连接远程/本地模型服务 ============
llm = ChatOpenAI(
    model="Sinong1.0-8B",
    base_url="http://<your-server-ip>:8000/v1",  # 替换为实际地址
    api_key="not-needed",                          # 本地部署无需真实key
    temperature=0.6,
    max_tokens=2048,
)

# ============ 定义工具 ============
@tool
def your_tool(query: str) -> str:
    """你的工具描述"""
    return "result"

# ============ 创建 Agent ============
agent = create_react_agent(
    model=llm,
    tools=[your_tool],
    prompt="你是一个农业智能助手，基于司农大语言模型。",
)

# ============ 调用 ============
result = agent.invoke({
    "messages": [{"role": "user", "content": "你的问题"}]
})
print(result["messages"][-1].content)
```

### 6.3 多 Agent 编排

```python
from langgraph.graph import StateGraph, MessagesState

# 定义多个节点
def researcher(state: MessagesState):
    """研究节点：负责知识检索"""
    return {"messages": [llm.invoke(state["messages"])]}

def analyst(state: MessagesState):
    """分析节点：负责数据分析"""
    return {"messages": [llm.invoke(state["messages"])]}

# 构建工作流
workflow = StateGraph(MessagesState)
workflow.add_node("researcher", researcher)
workflow.add_node("analyst", analyst)
workflow.add_edge("researcher", "analyst")
workflow.set_entry_point("researcher")

app = workflow.compile()
result = app.invoke({"messages": [{"role": "user", "content": "分析今年小麦种植趋势"}]})
```

---

## 七、常见问题

### Q1: Mac 上 MPS 报错？
改为 `--device cpu`，Mac 上 MPS 对 bfloat16 支持有限。

### Q2: GPU 显存不足？
- 减小 `--max-model-len`（如改为 8192）
- 使用 `--gpu-memory-utilization 0.95`
- 使用 4-bit 量化版（需转换 GGUF 格式）

### Q3: vLLM 启动报模型格式错误？
确保使用 vllm >= 0.9.0，Qwen3ForCausalLM 是较新架构。

### Q4: 如何关闭思考模式？
请求中设置 `"enable_thinking": false`，或在消息中添加 `/no_think`。

### Q5: Docker 中 GPU 不可用？
确保安装了 NVIDIA Container Toolkit：
```bash
# Ubuntu
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

---

## 八、部署架构图

```
┌─────────────────────────────────────────────────────────────┐
│                      你的应用层                               │
│  ┌────────────────┐  ┌────────────────┐  ┌───────────────┐ │
│  │  LangGraph     │  │  LangChain     │  │  其他客户端    │ │
│  │  Agent 编排    │  │  Chain/Tool    │  │  (curl/SDK)   │ │
│  └───────┬────────┘  └───────┬────────┘  └───────┬───────┘ │
│          │                   │                   │         │
│          └───────────────────┼───────────────────┘         │
│                              │ OpenAI 兼容 API              │
│                              ▼                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │          Sinong1.0-8B API Service                     │  │
│  │  ┌──────────────┐  或  ┌──────────────┐             │  │
│  │  │ FastAPI +    │      │ vLLM Server  │             │  │
│  │  │ Transformers │      │ (高性能)      │             │  │
│  │  └──────┬───────┘      └──────┬───────┘             │  │
│  │         │                      │                      │  │
│  │         ▼                      ▼                      │  │
│  │    Qwen3ForCausalLM (8.2B)                           │  │
│  └──────────────────────────────────────────────────────┘  │
│                              │                              │
│                              ▼                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Docker 容器 / 裸机部署                               │  │
│  │  NVIDIA GPU (16GB+ VRAM)                             │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```
