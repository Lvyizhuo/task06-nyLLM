# Sinong1.0-32B vLLM 部署指南

使用 vLLM 官方镜像部署 Sinong1.0-32B 模型，提供 OpenAI 兼容 API。

## 文件说明

```
deploy/
├── start_vllm.sh      # vLLM 启动脚本（本地部署用）
├── requirements.txt    # Python 依赖（本地部署用）
└── README.md           # 本文档
```

## 服务器环境

| 项目 | 配置 |
|------|------|
| GPU | 4 × NVIDIA A100-PCIE-40GB |
| CUDA | 12.6（驱动 560.28.03） |
| 模型路径 | `/root/ntt/lvyizhuo/nyLLM/models/` |

---

## 模型下载

```bash
modelscope download --model NAULLM/Sinong1.0-32B --local_dir ./models
```

---

## Docker 部署（推荐）

### 前置条件

```bash
# 确认 nvidia-container-toolkit 已安装
nvidia-ctk --version

# 若未安装：
apt-get update && apt-get install -y nvidia-container-toolkit
systemctl restart docker
```

### 拉取镜像

```bash
# 使用华为云镜像源（国内加速）
docker pull swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/vllm/vllm-openai:v0.8.5
```

### 启动服务

```bash
docker run -d \
  --gpus all \
  --name sinongLLM \
  --shm-size=10g \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -p 8000:8000 \
  -v /root/ntt/lvyizhuo/nyLLM/models:/app/models:ro \
  swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/vllm/vllm-openai:v0.8.5 \
  --model /app/models \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --distributed-executor-backend mp \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192 \
  --dtype bfloat16 \
  --trust-remote-code
```

### 查看日志

```bash
docker logs -f sinongLLM
```

等待出现 `Application startup complete` 表示启动成功。

### 停止/重启

```bash
docker stop sinongLLM      # 停止
docker start sinongLLM     # 启动
docker rm -f sinongLLM     # 删除容器
```

---

## 测试请求

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/app/models",
    "messages": [
      {
        "role": "system",
        "content": "你是专业农业智能助手司农Sinong，精通粮食种植、果蔬栽培、病虫害防治、畜禽养殖、农资化肥、农田管理、农业政策，回答通俗易懂，给出实操步骤，数据贴合国内北方大田种植场景。"
      },
      {
        "role": "user",
        "content": "山东潍坊地区夏玉米苗期出现叶片发黄、根部腐烂，地里积水多，是什么病害？该怎么防治，后续田间排水和施肥方案是什么？"
      }
    ],
    "stream": true,
    "temperature": 0.4,
    "max_tokens": 1500
  }'
```

---

## 启动参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `--gpus all` | - | 让容器访问所有 GPU |
| `--shm-size=10g` | - | 共享内存，避免 Ray/多进程问题 |
| `-e CUDA_VISIBLE_DEVICES=0,1` | - | 指定使用 GPU 0 和 1 |
| `--tensor-parallel-size` | 2 | 2 卡张量并行（30B 模型推荐） |
| `--distributed-executor-backend` | mp | 使用 multiprocessing（比 Ray 稳定） |
| `--gpu-memory-utilization` | 0.9 | GPU 显存使用率 |
| `--max-model-len` | 8192 | 最大序列长度 |
| `--dtype` | bfloat16 | 数据类型 |

---

## 故障排查

| 问题 | 解决方案 |
|------|---------|
| CUDA 版本不兼容 | 使用 vllm v0.8.5（兼容 CUDA 12.4-12.6） |
| transformers 不识别 qwen3 | 使用 vllm v0.8.5+（内置 transformers 4.51+） |
| Ray 启动失败 | 添加 `--distributed-executor-backend mp` |
| 共享内存不足 | 添加 `--shm-size=10g` |
| GPU 不可见 | 使用 `--gpus all` + `CUDA_VISIBLE_DEVICES` |
| 显存不足 OOM | 降低 `--max-model-len` 或 `--gpu-memory-utilization` |
