FROM vllm/vllm-openai:latest

# 挂载模型目录
RUN mkdir -p /app/models

EXPOSE 8000

ENTRYPOINT ["vllm", "serve"]
