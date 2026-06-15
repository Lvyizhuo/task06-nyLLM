# Sinong1.0-8B 部署指南

vllm OpenAI 兼容 API 服务，部署 Sinong1.0-8B 农业大模型。

## 文件说明

```
deploy/
├── start_vllm.sh      # vllm 启动脚本
├── docker-compose.yml # Docker 部署
├── requirements.txt   # Python 依赖
└── README.md          # 本文档
```

## 快速启动

### 方式一：本地部署

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务
bash start_vllm.sh

# 自定义端口
bash start_vllm.sh ../models/Sinong1.0-8B 8001
```

### 方式二：Docker 部署

```bash
docker-compose up -d
```

## API 接口

服务地址：`http://localhost:8000`

### 健康检查

```bash
GET /health
```

### 列出模型

```bash
GET /v1/models
```

### 聊天补全

```bash
POST /v1/chat/completions
Content-Type: application/json
```

**请求参数：**

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| model | string | 是 | - | `Sinong1.0-8B` |
| messages | array | 是 | - | 消息列表 |
| max_tokens | int | 否 | 512 | 最大生成 token 数 |
| temperature | float | 否 | 0.6 | 采样温度 |
| top_p | float | 否 | 0.95 | 核采样参数 |
| top_k | int | 否 | 20 | Top-K 采样参数 |
| stream | bool | 否 | false | 是否流式输出 |

**请求示例：**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Sinong1.0-8B",
    "messages": [
      {"role": "system", "content": "你是农业智能助手"},
      {"role": "user", "content": "小麦赤霉病怎么防治？"}
    ],
    "max_tokens": 512,
    "temperature": 0.6,
    "stream": false
  }'
```

**响应示例：**

```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "model": "Sinong1.0-8B",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "小麦赤霉病防治方法..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 150,
    "total_tokens": 175
  }
}
```

**流式请求：**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Sinong1.0-8B",
    "messages": [{"role": "user", "content": "你好"}],
    "stream": true
  }'
```

**流式响应：**

```
data: {"id":"chatcmpl-xxx","choices":[{"delta":{"role":"assistant"}}]}
data: {"id":"chatcmpl-xxx","choices":[{"delta":{"content":"你"}}]}
data: {"id":"chatcmpl-xxx","choices":[{"delta":{"content":"好"}}]}
data: {"id":"chatcmpl-xxx","choices":[{"delta":{},"finish_reason":"stop"}]}
data: [DONE]
```

## 模型默认参数

| 参数 | 值 |
|------|-----|
| temperature | 0.6 |
| top_p | 0.95 |
| top_k | 20 |
| max_model_len | 40960 |
| dtype | bfloat16 |

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

## 故障排查

**显存不足：** 编辑 `start_vllm.sh`，降低 `--gpu-memory-utilization` 值

**端口占用：** `lsof -i :8000` 查看并 `kill -9 <PID>` 终止进程

**查看日志：** `docker-compose logs -f`（Docker 部署）
