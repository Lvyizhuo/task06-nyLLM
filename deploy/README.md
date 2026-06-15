# Sinong1.0-8B vLLM 部署指南

基于 vLLM 原生 OpenAI 兼容 API 部署 Sinong1.0-8B（Qwen3ForCausalLM 架构）。

## 文件说明

```
deploy/
├── start_vllm.sh      # vLLM 启动脚本（本地 & Docker 通用）
├── Dockerfile          # Docker 镜像构建（NVIDIA GPU + CUDA 12.4）
├── docker-compose.yml  # Docker Compose 编排
├── requirements.txt    # Python 依赖
└── README.md           # 本文档
```

## 方式一：服务器本地部署

```bash
# 安装依赖
pip install -r requirements.txt

# 单卡启动
bash start_vllm.sh /data/models/Sinong1.0-8B 8000 1

# 多卡启动（如 2 张 GPU）
bash start_vllm.sh /data/models/Sinong1.0-8B 8000 2
```

## 方式二：Docker 部署

```bash
# 构建并启动（单卡）
docker compose up -d --build

# 多卡部署：编辑 docker-compose.yml
#   1. command 第三参数改为 GPU 数量，如 ["/app/models/Sinong1.0-8B", "8000", "2"]
#   2. deploy.resources.reservations.devices.count 改为对应数量

# 查看日志
docker compose logs -f

# 停止
docker compose down
```

### 前置要求

- 服务器已安装 NVIDIA 驱动 + Docker + nvidia-container-toolkit
- 模型文件已放置在 `/data/models/Sinong1.0-8B`（或修改 docker-compose.yml 挂载路径）

## API 接口

服务地址：`http://<host>:8000`

### 健康检查

```bash
curl http://localhost:8000/health
```

### 列出模型

```bash
curl http://localhost:8000/v1/models
```

### 聊天补全（非流式）

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Sinong1.0-8B",
    "messages": [
      {"role": "system", "content": "你是农业智能助手，请直接回答问题，不要提及参考资料。"},
      {"role": "user", "content": "小麦赤霉病怎么防治？"}
    ],
    "max_tokens": 512,
    "temperature": 0.6,
    "stream": false
  }'
```

### 聊天补全（流式）

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Sinong1.0-8B",
    "messages": [{"role": "user", "content": "介绍济南"}],
    "stream": true
  }'
```

### 关闭思考模式

在消息中添加 `/no_think` 或在请求中设置 `chat_template_kwargs`：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Sinong1.0-8B",
    "messages": [
      {"role": "system", "content": "你是农业智能助手，请直接回答问题，不要提及参考资料。/no_think"},
      {"role": "user", "content": "小麦赤霉病怎么防治？"}
    ],
    "max_tokens": 512,
    "temperature": 0.6
  }'
```

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

## 默认参数

| 参数 | 值 |
|------|-----|
| dtype | bfloat16 |
| max-model-len | 40960 |
| gpu-memory-utilization | 0.9 |
| enable-reasoning | true |
| reasoning-parser | deepseek_r1 |

## 硬件需求

| 配置 | 最低 | 推荐 |
|------|------|------|
| GPU 显存 | 16GB（如 V100-16G） | 24GB+（如 A10/4090/A100） |
| 系统内存 | 32GB | 64GB |
| 磁盘 | 20GB | 50GB+ |

## 故障排查

- **显存不足**：降低 `--gpu-memory-utilization` 或 `--max-model-len`
- **端口占用**：`lsof -i :8000` 查看并终止
- **Docker GPU 不可用**：确认已安装 `nvidia-container-toolkit` 并重启 Docker
- **日志**：`docker compose logs -f`
