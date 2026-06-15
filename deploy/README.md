# Sinong1.0-8B API 服务

vllm 原生 OpenAI 兼容 API 服务，用于部署 Sinong1.0-8B 农业大模型。

## 快速启动

### 方式一：直接启动

```bash
cd deploy
pip install -r requirements.txt
python start_vllm.py
```

### 方式二：Docker 部署

```bash
cd deploy
docker-compose up -d
```

## API 接口

服务启动后，默认地址：`http://localhost:8000`

### 1. 健康检查

```bash
GET /health
```

**响应示例：**
```json
{
  "status": "ok",
  "model": "Sinong1.0-8B"
}
```

### 2. 列出模型

```bash
GET /v1/models
```

**响应示例：**
```json
{
  "object": "list",
  "data": [
    {
      "id": "Sinong1.0-8B",
      "object": "model",
      "owned_by": "vllm"
    }
  ]
}
```

### 3. 聊天补全

```bash
POST /v1/chat/completions
Content-Type: application/json
```

**请求参数：**

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| model | string | 是 | - | 模型名称：`Sinong1.0-8B` |
| messages | array | 是 | - | 消息列表 |
| max_tokens | int | 否 | 512 | 最大生成 token 数 |
| temperature | float | 否 | 0.6 | 采样温度 |
| top_p | float | 否 | 0.95 | 核采样参数 |
| top_k | int | 否 | 20 | Top-K 采样参数 |
| stream | bool | 否 | false | 是否流式输出 |

**消息格式：**
```json
{
  "messages": [
    {"role": "system", "content": "你是农业智能助手"},
    {"role": "user", "content": "小麦赤霉病怎么防治？"}
  ]
}
```

**非流式请求示例：**
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Sinong1.0-8B",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 100
  }'
```

**非流式响应示例：**
```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "Sinong1.0-8B",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "你好！我是司农..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 50,
    "total_tokens": 60
  }
}
```

**流式请求示例：**
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Sinong1.0-8B",
    "messages": [{"role": "user", "content": "你好"}],
    "stream": true
  }'
```

**流式响应示例：**
```
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"delta":{"role":"assistant"}}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"delta":{"content":"你"}}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"delta":{"content":"好"}}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

## 模型默认参数

| 参数 | 值 | 说明 |
|------|-----|------|
| temperature | 0.6 | 采样温度 |
| top_p | 0.95 | 核采样参数 |
| top_k | 20 | Top-K 采样参数 |
| max_model_len | 40960 | 最大上下文长度 |
| dtype | bfloat16 | 数据类型 |

## LangGraph 集成

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="Sinong1.0-8B",
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
    temperature=0.6,
    max_tokens=2048,
    streaming=True,
)
```

## 配置说明

### 环境变量

复制 `.env.example` 为 `.env` 并修改：

```bash
MODEL_PATH=../models/Sinong1.0-8B
HOST=0.0.0.0
PORT=8000
GPU_MEMORY_UTILIZATION=0.9
```

### 启动参数

```bash
python start_vllm.py \
  --model-path ../models/Sinong1.0-8B \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  --tensor-parallel-size 1
```

## 故障排查

**显存不足：**
```bash
python start_vllm.py --gpu-memory-utilization 0.8
```

**端口占用：**
```bash
lsof -i :8000
kill -9 <PID>
```

**查看日志：**
```bash
docker-compose logs -f  # Docker 部署
```
