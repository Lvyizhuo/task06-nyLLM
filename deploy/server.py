"""
Sinong1.0-8B OpenAI-Compatible API Server
基于 FastAPI + Transformers 的本地推理服务
提供 OpenAI 兼容的 /v1/chat/completions 接口，可直接被 LangGraph 调用
"""

import os
import json
import time
import uuid
import argparse
from typing import Optional, List, Dict, Any, AsyncGenerator

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============ 全局变量 ============
MODEL = None
TOKENIZER = None
MODEL_PATH = None

# ============ 请求/响应模型 ============

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "Sinong1.0-8B"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.6
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 20
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False
    enable_thinking: Optional[bool] = True  # Qwen3 思考模式

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo

# ============ 应用 ============
app = FastAPI(title="Sinong1.0-8B API Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_model(model_path: str, device: str = "auto"):
    """加载模型和分词器"""
    global MODEL, TOKENIZER, MODEL_PATH
    MODEL_PATH = model_path

    print(f"[INFO] 正在加载模型: {model_path}")
    print(f"[INFO] 设备: {device}")

    TOKENIZER = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    MODEL = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device,
        trust_remote_code=True,
    )
    MODEL.eval()

    print(f"[INFO] 模型加载完成，设备: {MODEL.device}")


def generate_response(
    messages: List[Dict],
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    max_tokens: int = 2048,
    enable_thinking: bool = True,
) -> tuple:
    """生成回复，返回 (text, prompt_tokens, completion_tokens)"""
    # 构建对话文本
    text = TOKENIZER.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )

    inputs = TOKENIZER([text], return_tensors="pt").to(MODEL.device)
    prompt_tokens = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = MODEL.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=top_p,
            top_k=top_k,
            do_sample=temperature > 0,
            pad_token_id=TOKENIZER.eos_token_id[0] if isinstance(TOKENIZER.eos_token_id, list) else TOKENIZER.eos_token_id,
        )

    # 截取生成部分
    generated_ids = outputs[0][prompt_tokens:]
    completion_tokens = len(generated_ids)

    # 解码 - 处理 Qwen3 的思考内容
    response_text = TOKENIZER.decode(generated_ids, skip_special_tokens=True)

    # 分离思考内容和最终回答
    think_content = ""
    answer_content = response_text

    if enable_thinking and "<think!" in response_text:
        # Qwen3 思考模式: <think!>思考内容</think!>回答内容
        parts = response_text.split("</think!>")
        if len(parts) >= 2:
            think_content = parts[0].replace("<think!", "").strip()
            answer_content = parts[1].strip()
    elif enable_thinking:
        # 尝试其他格式
        for tag in ["</think}", "</think"]:
            if tag in response_text:
                parts = response_text.split(tag)
                if len(parts) >= 2:
                    think_content = parts[0].strip()
                    answer_content = parts[1].strip()
                break

    return answer_content, think_content, prompt_tokens, completion_tokens


# ============ API 端点 ============

@app.get("/v1/models")
async def list_models():
    """列出可用模型"""
    return {
        "object": "list",
        "data": [
            {
                "id": "Sinong1.0-8B",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "NAULLM",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI 兼容的聊天补全接口"""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="模型尚未加载完成")

    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    try:
        answer, think, prompt_tokens, completion_tokens = generate_response(
            messages=messages,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            max_tokens=request.max_tokens,
            enable_thinking=request.enable_thinking,
        )

        # 构建返回消息
        msg_content = answer
        if think:
            msg_content = f"<think!>{think}</think!>\n{answer}"

        response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=msg_content),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}")


@app.get("/health")
async def health():
    return {"status": "ok", "model": "Sinong1.0-8B", "loaded": MODEL is not None}


# ============ 启动 ============

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sinong1.0-8B API Server")
    parser.add_argument("--model-path", type=str, required=True, help="模型本地路径")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    parser.add_argument("--device", type=str, default="auto", help="设备 (auto/cuda/cpu/mps)")
    args = parser.parse_args()

    load_model(args.model_path, device=args.device)

    uvicorn.run(app, host=args.host, port=args.port)
