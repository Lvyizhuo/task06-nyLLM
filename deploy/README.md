# Sinong1.0-8B vllm OpenAI 服务部署指南

本指南介绍如何使用 vllm 原生 OpenAI 服务部署 Sinong1.0-8B 模型，并与 LangGraph 智能体集成。

## 架构说明

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   前端界面       │────▶│  LangGraph      │────▶│  vllm OpenAI    │
│  (流式输出)      │◀────│  智能体          │◀────│  服务           │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

**优势**：
- 使用 vllm 原生 OpenAI 服务，性能更好，更稳定
- 标准 OpenAI API 兼容，可直接被 LangGraph 调用
- 支持真正的流式输出
- 无需维护自定义 FastAPI 代码

## 前置要求

### 硬件要求
- NVIDIA GPU（推荐 A100/H100 或同等性能显卡）
- 显存：至少 16GB（8B 模型）

### 软件要求
- Python 3.9+
- CUDA 11.8+
- Docker（可选，用于容器化部署）

## 快速开始

### 方案一：直接启动（推荐开发环境）

1. **安装依赖**
```bash
pip install -r requirements.txt
```

2. **启动 vllm 服务**
```bash
# 使用 Python 脚本启动（推荐）
python start_vllm.py --model-path ../models/Sinong1.0-8B

# 或使用 shell 脚本启动
./start_vllm.sh

# 自定义配置
python start_vllm.py \
    --model-path ../models/Sinong1.0-8B \
    --port 8001 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 32768
```

3. **验证服务**
```bash
# 健康检查
curl http://localhost:8000/health

# 列出模型
curl http://localhost:8000/v1/models

# 测试对话
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Sinong1.0-8B",
        "messages": [{"role": "user", "content": "你好"}],
        "max_tokens": 100
    }'
```

### 方案二：Docker 部署（推荐生产环境）

1. **使用 Docker Compose**
```bash
cd deploy
docker-compose up -d
```

2. **查看日志**
```bash
docker-compose logs -f sinong-vllm
```

3. **停止服务**
```bash
docker-compose down
```

## 配置说明

### 启动参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model-path` | 必需 | 模型本地路径 |
| `--host` | 0.0.0.0 | 监听地址 |
| `--port` | 8000 | 监听端口 |
| `--gpu-memory-utilization` | 0.9 | GPU 显存利用率 |
| `--max-model-len` | 40960 | 最大上下文长度 |
| `--tensor-parallel-size` | 1 | Tensor 并行数（多 GPU） |
| `--dtype` | bfloat16 | 数据类型 |
| `--enforce-eager` | False | 强制 eager 模式（调试用） |
| `--served-model-name` | 目录名 | 对外模型名称 |
| `--api-key` | None | API Key 认证 |

### 环境变量

```bash
# 模型路径
export MODEL_PATH=./models/Sinong1.0-8B

# 服务配置
export HOST=0.0.0.0
export PORT=8000

# GPU 配置
export GPU_MEMORY_UTILIZATION=0.9
export MAX_MODEL_LEN=40960
export TENSOR_PARALLEL_SIZE=1
```

## LangGraph 集成

### 基本使用

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# 连接 vllm 服务
llm = ChatOpenAI(
    model="Sinong1.0-8B",
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
    temperature=0.6,
    max_tokens=2048,
    streaming=True,  # 启用流式
)

# 流式调用
for chunk in llm.stream([HumanMessage(content="你好")]):
    print(chunk.content, end="", flush=True)
```

### 完整智能体示例

参考 `langgraph_agent.py` 文件，包含：
- 非流式 Agent 调用
- 流式 Agent 调用
- 多轮对话支持
- 工具调用示例

## 多 GPU 部署

如果有多张 GPU，可以使用 Tensor Parallel：

```bash
# 2 GPU 并行
python start_vllm.py \
    --model-path ../models/Sinong1.0-8B \
    --tensor-parallel-size 2

# 4 GPU 并行
python start_vllm.py \
    --model-path ../models/Sinong1.0-8B \
    --tensor-parallel-size 4
```

## API Key 认证

如果需要 API Key 认证：

```bash
# 启动时设置 API Key
python start_vllm.py \
    --model-path ../models/Sinong1.0-8B \
    --api-key your-secret-key

# 调用时传入 API Key
llm = ChatOpenAI(
    model="Sinong1.0-8B",
    base_url="http://localhost:8000/v1",
    api_key="your-secret-key",
)
```

## 性能优化建议

1. **GPU 显存**：根据显存大小调整 `--gpu-memory-utilization`
   - 16GB 显存：0.8-0.85
   - 24GB 显存：0.85-0.9
   - 40GB+ 显存：0.9-0.95

2. **上下文长度**：根据实际需求调整 `--max-model-len`
   - 短对话为主：16384-32768
   - 长文档处理：40960 或更大

3. **Tensor Parallel**：多 GPU 时使用，可显著提升吞吐量

4. **Batch Size**：vllm 自动管理，无需手动设置

## 故障排查

### 1. 服务启动失败

```bash
# 检查 GPU 状态
nvidia-smi

# 检查 CUDA 版本
nvcc --version

# 检查 vllm 安装
python -c "import vllm; print(vllm.__version__)"
```

### 2. 显存不足

```bash
# 降低显存利用率
python start_vllm.py --gpu-memory-utilization 0.8

# 减少上下文长度
python start_vllm.py --max-model-len 16384
```

### 3. 流式输出问题

确保：
- vllm 服务正常运行
- LangGraph 配置了 `streaming=True`
- 前端正确处理 SSE 事件

### 4. 模型加载慢

首次启动需要加载模型，可能需要 1-3 分钟。后续启动会使用缓存。

## 与旧版 FastAPI 服务的对比

| 特性 | 旧版 FastAPI | 新版 vllm 原生 |
|------|-------------|---------------|
| 性能 | 一般 | 优秀 |
| 稳定性 | 一般 | 优秀 |
| 流式支持 | 模拟 | 真正流式 |
| 维护成本 | 高 | 低 |
| API 兼容性 | 自定义 | 标准 OpenAI |
| 多 GPU 支持 | 需手动实现 | 原生支持 |

## 迁移指南

如果你正在从旧版 FastAPI 服务迁移：

1. **停止旧服务**
```bash
# 停止旧的 FastAPI 服务
pkill -f "python server.py"
```

2. **启动新服务**
```bash
python start_vllm.py --model-path ../models/Sinong1.0-8B
```

3. **更新客户端代码**
   - 将 `base_url` 指向新服务（默认相同端口）
   - 无需修改其他代码，API 完全兼容

4. **验证功能**
```bash
curl http://localhost:8000/v1/models
```

## 相关文档

- [vllm 官方文档](https://docs.vllm.ai/)
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)
- [OpenAI API 规范](https://platform.openai.com/docs/api-reference)

## 支持

如有问题，请检查：
1. GPU 驱动和 CUDA 版本
2. vllm 版本兼容性
3. 模型文件完整性
4. 端口是否被占用
