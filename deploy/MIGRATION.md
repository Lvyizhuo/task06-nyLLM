# 从 FastAPI 迁移到 vllm 原生 OpenAI 服务

本文档说明如何从旧版 FastAPI 服务迁移到 vllm 原生 OpenAI 服务。

## 迁移步骤

### 1. 停止旧服务

```bash
# 如果旧服务在运行，先停止
pkill -f "python server.py"

# 或者如果使用 systemd
sudo systemctl stop sinong-api
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 启动新服务

```bash
# 使用 Python 脚本启动（推荐）
python start_vllm.py --model-path ../models/Sinong1.0-8B

# 或使用 shell 脚本
./start_vllm.sh
```

### 4. 验证服务

```bash
# 运行测试脚本
python test_service.py

# 或手动测试
curl http://localhost:8000/health
curl http://localhost:8000/v1/models
```

### 5. 更新客户端代码

由于新服务使用标准 OpenAI API，客户端代码通常无需修改。只需确保：

```python
# 确保 base_url 指向新服务
llm = ChatOpenAI(
    model="Sinong1.0-8B",
    base_url="http://localhost:8000/v1",  # 端口相同
    api_key="not-needed",
    # 其他参数保持不变
)
```

## 主要变化

### 1. 启动方式

**旧版（FastAPI）**:
```bash
python server.py --model-path ./models/Sinong1.0-8B --backend vllm --vllm-async
```

**新版（vllm 原生）**:
```bash
python start_vllm.py --model-path ./models/Sinong1.0-8B
```

### 2. 配置参数

**旧版参数**:
- `--backend transformers|vllm`
- `--device auto|cuda|cpu|mps`
- `--vllm-async`

**新版参数**:
- `--gpu-memory-utilization` (0.0-1.0)
- `--max-model-len`
- `--tensor-parallel-size`
- `--dtype bfloat16|float16|auto`
- `--enforce-eager`

### 3. API 端点

**相同点**:
- `GET /health` - 健康检查
- `GET /v1/models` - 列出模型
- `POST /v1/chat/completions` - 聊天补全

**不同点**:
- 旧版：自定义实现，可能有细微差异
- 新版：标准 OpenAI API，完全兼容

### 4. 流式输出

**旧版**:
- 模拟流式（同步引擎）
- 真正流式（异步引擎）

**新版**:
- 所有情况都是真正流式
- 性能更好，延迟更低

## 功能对比

| 功能 | 旧版 FastAPI | 新版 vllm 原生 |
|------|-------------|---------------|
| 非流式对话 | ✓ | ✓ |
| 流式对话 | ✓（模拟/真正） | ✓（真正流式） |
| 多 GPU 支持 | 需手动实现 | 原生支持 |
| 性能优化 | 一般 | 优秀 |
| API 兼容性 | 自定义 | 标准 OpenAI |
| 维护成本 | 高 | 低 |
| 思考模式支持 | ✓ | ✓（需要配置） |

## 保留的功能

以下功能在新版中仍然支持：

1. **系统提示词**：通过 LangGraph 的 `SystemMessage` 设置
2. **温度控制**：通过 API 参数 `temperature`
3. **最大 token 数**：通过 API 参数 `max_tokens`
4. **Top-p/Top-k**：通过 API 参数 `top_p`, `top_k`

## 注意事项

### 1. 思考模式

旧版支持 `enable_thinking` 参数。新版 vllm 默认不支持此参数，需要：

- 在模型的 chat template 中配置
- 或使用 vllm 的 `--chat-template` 参数指定自定义模板

### 2. 自定义系统提示词

旧版支持 `--system-prompt` 参数。新版需要在客户端代码中设置：

```python
from langchain_core.messages import SystemMessage

llm.invoke([
    SystemMessage(content="你的自定义系统提示词"),
    HumanMessage(content="用户问题")
])
```

### 3. 模型名称

确保 `--served-model-name` 与客户端使用的模型名称一致：

```bash
# 启动时指定模型名称
python start_vllm.py \
    --model-path ../models/Sinong1.0-8B \
    --served-model-name Sinong1.0-8B
```

## 回滚方案

如果需要回滚到旧版：

```bash
# 停止新服务
pkill -f "vllm"

# 启动旧服务
python server.py --model-path ../models/Sinong1.0-8B --backend vllm --vllm-async
```

## 常见问题

### Q1: 端口冲突

如果旧服务仍在占用 8000 端口：

```bash
# 查找占用端口的进程
lsof -i :8000

# 终止进程
kill -9 <PID>

# 或使用不同端口
python start_vllm.py --port 8001
```

### Q2: 显存不足

```bash
# 降低显存利用率
python start_vllm.py --gpu-memory-utilization 0.8

# 减少上下文长度
python start_vllm.py --max-model-len 16384
```

### Q3: 模型加载慢

首次启动需要加载模型，可能需要 1-3 分钟。这是正常现象。

### Q4: 流式输出卡顿

确保：
- vllm 服务正常运行
- 客户端配置了 `streaming=True`
- 网络连接稳定

## 性能优化建议

1. **使用异步引擎**：vllm 默认使用异步引擎，性能更好
2. **调整批处理大小**：vllm 自动管理，通常无需手动设置
3. **使用 Tensor Parallel**：多 GPU 时使用
4. **优化显存使用**：根据 GPU 显存调整参数

## 监控和日志

### 查看服务状态

```bash
# 健康检查
curl http://localhost:8000/health

# 查看 GPU 使用
nvidia-smi

# 查看进程
ps aux | grep vllm
```

### 查看日志

vllm 默认输出日志到标准输出。如需持久化日志：

```bash
# 重定向日志到文件
python start_vllm.py --model-path ../models/Sinong1.0-8B 2>&1 | tee vllm.log
```

## 下一步

迁移完成后，建议：

1. 运行完整测试：`python test_service.py`
2. 更新文档和部署脚本
3. 通知团队成员新的启动方式
4. 监控服务性能和稳定性
