# Sinong1.0-32B vLLM 部署指南

基于 vLLM 原生 OpenAI 兼容 API 部署 Sinong1.0-32B（Qwen3ForCausalLM 架构）。

## 文件说明

```
deploy/
├── start_vllm.sh      # vLLM 启动脚本（本地 & Docker 通用）
├── Dockerfile          # Docker 镜像构建（NVIDIA GPU + CUDA 12.4）
├── docker-compose.yml  # Docker Compose 编排
├── requirements.txt    # Python 依赖
└── README.md           # 本文档
```

## 服务器环境

| 项目 | 配置 |
|------|------|
| GPU | 4 × NVIDIA A100-PCIE-40GB |
| CUDA | 12.6（驱动 560.28.03） |
| 模型路径 | `/root/ntt/lvyizhuo/nyLLM/models/` |
| 项目路径 | `/root/ntt/lvyizhuo/nyLLM/` |

## 模型信息

| 项目 | 值 |
|------|-----|
| 架构 | Qwen3ForCausalLM |
| 参数量 | 32B |
| 精度 | bfloat16（权重约 64GB） |
| 层数 | 64 |
| hidden_size | 5120 |
| max_position_embeddings | 40960 |
| 分片数 | 14 个 safetensors |

## GPU 并行策略

| TP | 使用卡数 | 显存占用 | 适用场景 |
|----|---------|---------|---------|
| 2 | 2 × A100 40GB | ~32GB/卡 | **最低推荐**，留 8GB 余量给 KV cache |
| 4 | 4 × A100 40GB | ~16GB/卡 | **推荐**，更大 batch/更长上下文 |

> ⚠️ **TP=1 不可用**：32B BF16 约 64GB，单卡 40GB 放不下。

---

## 方式一：服务器本地部署

```bash
cd /root/ntt/lvyizhuo/nyLLM

# 安装依赖
pip install -r deploy/requirements.txt

# TP=2 启动（2卡并行，最低推荐）
bash deploy/start_vllm.sh /root/ntt/lvyizhuo/nyLLM/models 8000 2

# TP=4 启动（4卡并行，推荐）
bash deploy/start_vllm.sh /root/ntt/lvyizhuo/nyLLM/models 8000 4
```

## 方式二：Docker 部署

### 前置条件

```bash
# 确认 nvidia-container-toolkit 已安装
nvidia-ctk --version

# 若未安装：
# curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
#   gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
# curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
#   sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
#   tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
# apt-get update && apt-get install -y nvidia-container-toolkit
# systemctl restart docker
```

### 部署步骤

```bash
cd /root/ntt/lvyizhuo/nyLLM

# 1. 把 deploy/ 目录上传到服务器（scp/rsync）
# 2. 确保模型文件在 /root/ntt/lvyizhuo/nyLLM/models/ 下
# 3. 构建并启动
cd deploy && docker compose up -d --build

# 4. 查看启动日志（首次加载模型约需 2-3 分钟）
docker compose logs -f

# 5. 验证
curl http://localhost:8000/v1/models
```

### 调整并行度

编辑 `docker-compose.yml` 中的 `command` 第三参数：

```yaml
# TP=2（2卡并行）
command: ["/app/models/Sinong1.0-32B", "8000", "2"]

# TP=4（4卡并行，推荐）
command: ["/app/models/Sinong1.0-32B", "8000", "4"]
```

同时调整 `CUDA_VISIBLE_DEVICES` 只使用需要的卡：

```yaml
# TP=2 只用 GPU 0,1
environment:
  - CUDA_VISIBLE_DEVICES=0,1

# TP=4 全部使用
environment:
  - CUDA_VISIBLE_DEVICES=0,1,2,3
```

### 停止 / 重启

```bash
docker compose down        # 停止
docker compose restart     # 重启
docker compose logs -f     # 查看日志
```

---

## API 接口

服务地址：`http://<服务器IP>:8000`

### 健康检查

```bash
curl http://localhost:8000/health
```

### 聊天补全（非流式）

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Sinong1.0-32B",
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
    "model": "Sinong1.0-32B",
    "messages": [{"role": "user", "content": "介绍济南"}],
    "stream": true
  }'
```

### 关闭思考模式

在系统提示词末尾加 `/no_think`：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Sinong1.0-32B",
    "messages": [
      {"role": "system", "content": "你是农业智能助手，请直接回答问题。/no_think"},
      {"role": "user", "content": "小麦赤霉病怎么防治？"}
    ],
    "max_tokens": 512
  }'
```

---

## LangGraph 智能体集成

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="Sinong1.0-32B",
    base_url="http://<服务器IP>:8000/v1",
    api_key="not-needed",
    temperature=0.6,
    max_tokens=2048,
    streaming=True,
)

# 在 LangGraph 中使用
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(llm, tools=[...])
result = agent.invoke({"messages": [("user", "小麦赤霉病怎么防治？")]})
```

流式调用：

```python
for chunk in llm.stream([("user", "你好")]):
    print(chunk.content, end="", flush=True)
```

---

## 默认参数

| 参数 | 值 |
|------|-----|
| dtype | bfloat16 |
| max-model-len | 40960 |
| gpu-memory-utilization | 0.9 |
| enable-reasoning | true |
| reasoning-parser | deepseek_r1 |
| tensor-parallel-size | 2（默认）/ 4（推荐） |

## 故障排查

| 问题 | 解决方案 |
|------|---------|
| 显存不足 OOM | 降低 `--max-model-len`（如 16384）或 `--gpu-memory-utilization`（如 0.8） |
| 端口占用 | `lsof -i :8000` 查看并 `kill -9 <PID>` |
| Docker GPU 不可用 | 确认 `nvidia-container-toolkit` 已安装并重启 Docker |
| 模型加载慢 | 首次加载约 2-3 分钟，属正常现象 |
| 日志 | `docker compose logs -f` 或本地启动看终端输出 |
