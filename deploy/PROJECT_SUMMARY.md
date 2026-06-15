# 项目文件说明

本项目已从自定义 FastAPI 服务迁移到 vllm 原生 OpenAI 服务。以下是所有文件的说明。

## 文件结构

```
deploy/
├── README.md                 # 完整部署文档
├── MIGRATION.md             # 从 FastAPI 迁移指南
├── PROJECT_SUMMARY.md       # 本文件 - 项目文件说明
├── requirements.txt         # Python 依赖
├── .env.example            # 环境变量示例
│
├── start_vllm.py           # ✅ 新版启动脚本（Python，推荐）
├── start_vllm.sh           # ✅ 新版启动脚本（Shell）
├── start.sh                # ✅ 智能启动脚本（自动选择后端）
│
├── langgraph_agent.py      # ✅ LangGraph 智能体代码
├── example_streaming.py    # ✅ Python 流式输出示例
├── example_frontend.html   # ✅ 前端流式输出示例
├── test_service.py         # ✅ 服务测试脚本
│
├── server.py               # ⚠️ 旧版 FastAPI 服务（已弃用）
├── Dockerfile              # 旧版 Docker 配置
├── docker-compose.yml      # ✅ 新版 Docker Compose 配置
└── deploy_server.sh        # 旧版部署脚本
```

## 核心文件

### 1. 启动脚本

#### `start_vllm.py` - 推荐使用
Python 版启动脚本，功能最完整：
```bash
python start_vllm.py --model-path ../models/Sinong1.0-8B
```

**特性**：
- 自动检查依赖
- 显示 GPU 信息
- 灵活的参数配置
- 详细的启动日志

#### `start_vllm.sh` - Shell 版
Shell 版启动脚本，更轻量：
```bash
./start_vllm.sh
```

#### `start.sh` - 智能启动
自动检测环境，选择最佳启动方式：
```bash
./start.sh
```

- 如果 vllm 已安装：使用 vllm 原生服务
- 如果 vllm 未安装：回退到 FastAPI 服务

### 2. LangGraph 智能体

#### `langgraph_agent.py`
完整的 LangGraph 智能体实现，支持：
- 非流式调用
- 流式调用
- 多轮对话
- 工具调用

```python
from langgraph_agent import agent, streaming_agent

# 非流式
result = agent.invoke({"messages": [...]})

# 流式
for event in streaming_agent.stream({"messages": [...]}):
    print(event)
```

### 3. 示例代码

#### `example_streaming.py` - Python 流式示例
展示如何在 Python 中实现流式调用：
```bash
python example_streaming.py
```

#### `example_frontend.html` - 前端流式示例
完整的前端实现，包含：
- 流式输出
- 打字机效果
- 错误处理
- 响应式设计

直接在浏览器中打开即可使用。

### 4. 测试脚本

#### `test_service.py`
自动化测试脚本，验证服务是否正常：
```bash
# 运行所有测试
python test_service.py

# 测试特定功能
python test_service.py --test health
python test_service.py --test models
python test_service.py --test chat
python test_service.py --test stream
```

### 5. 配置文件

#### `requirements.txt`
Python 依赖列表：
```bash
pip install -r requirements.txt
```

#### `.env.example`
环境变量配置示例：
```bash
cp .env.example .env
# 编辑 .env 文件
```

#### `docker-compose.yml`
Docker Compose 配置，用于容器化部署：
```bash
docker-compose up -d
```

## 快速开始

### 1. 安装依赖

```bash
cd deploy
pip install -r requirements.txt
```

### 2. 启动服务

```bash
# 方式一：使用 Python 脚本（推荐）
python start_vllm.py --model-path ../models/Sinong1.0-8B

# 方式二：使用 Shell 脚本
./start_vllm.sh

# 方式三：智能启动
./start.sh
```

### 3. 验证服务

```bash
# 运行测试
python test_service.py

# 或手动测试
curl http://localhost:8000/health
curl http://localhost:8000/v1/models
```

### 4. 使用智能体

```python
from langgraph_agent import agent
from langchain_core.messages import HumanMessage

result = agent.invoke({
    "messages": [HumanMessage(content="小麦赤霉病怎么防治？")]
})
print(result["messages"][-1].content)
```

### 5. 前端集成

直接在浏览器中打开 `example_frontend.html`，或参考其实现集成到你的前端项目。

## API 接口

### 健康检查
```bash
GET /health
```

### 列出模型
```bash
GET /v1/models
```

### 聊天补全（非流式）
```bash
POST /v1/chat/completions
Content-Type: application/json

{
    "model": "Sinong1.0-8B",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 100,
    "temperature": 0.6,
    "stream": false
}
```

### 聊天补全（流式）
```bash
POST /v1/chat/completions
Content-Type: application/json

{
    "model": "Sinong1.0-8B",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 100,
    "temperature": 0.6,
    "stream": true
}
```

## 旧版文件（已弃用）

以下文件保留供参考，不建议使用：

- `server.py` - 旧版 FastAPI 服务
- `Dockerfile` - 旧版 Docker 配置
- `deploy_server.sh` - 旧版部署脚本

如需使用旧版，请参考 `server.py` 中的说明。

## 常见问题

### 1. 端口被占用
```bash
# 查找占用端口的进程
lsof -i :8000

# 终止进程
kill -9 <PID>

# 或使用不同端口
python start_vllm.py --port 8001
```

### 2. 显存不足
```bash
# 降低显存利用率
python start_vllm.py --gpu-memory-utilization 0.8

# 减少上下文长度
python start_vllm.py --max-model-len 16384
```

### 3. 模型加载慢
首次启动需要 1-3 分钟加载模型，这是正常现象。

### 4. 流式输出卡顿
确保：
- vllm 服务正常运行
- 客户端配置了 `streaming=True`
- 网络连接稳定

## 性能优化

1. **使用 Tensor Parallel**：多 GPU 时显著提升性能
2. **调整显存利用率**：根据 GPU 显存大小调整
3. **优化上下文长度**：根据实际需求设置
4. **使用异步引擎**：vllm 默认使用异步引擎

## 监控和日志

### 查看服务状态
```bash
curl http://localhost:8000/health
```

### 查看 GPU 使用
```bash
nvidia-smi
```

### 查看进程
```bash
ps aux | grep vllm
```

## 下一步

1. 阅读 `README.md` 了解完整部署流程
2. 如果从旧版迁移，参考 `MIGRATION.md`
3. 运行 `test_service.py` 验证服务
4. 参考 `example_frontend.html` 实现前端
5. 根据需要调整配置参数

## 支持

如有问题，请检查：
1. GPU 驱动和 CUDA 版本
2. vllm 版本兼容性
3. 模型文件完整性
4. 端口是否被占用
5. 阅读 `README.md` 中的故障排查章节
