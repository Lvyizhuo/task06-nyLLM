"""
Sinong1.0-8B OpenAI-Compatible API Server
支持两种推理后端：Transformers（本地调试）和 vLLM（生产加速）
提供 OpenAI 兼容的 /v1/chat/completions 接口，支持流式/非流式，可直接被 LangGraph 调用

关键说明:
- 模型基于 Qwen3-8B 微调，使用 Qwen3 的 chat_template 和思考模式标签
- 思考标签: <think!>...</think!>  (token 151657/151658)
- 回答标签: <answer>...</answer>  (token 151665/151666)
- 默认关闭思考模式 (enable_thinking=false)，避免"根据参考资料"幻觉
- 通过 --backend transformers|vllm 切换推理后端

使用示例:
  # Transformers 后端（本地调试，Mac/服务器均可）
  python server.py --model-path ./models/Sinong1.0-8B --backend transformers --device mps

  # vLLM 后端 - 同步引擎（生产加速，需 GPU）
  python server.py --model-path ./models/Sinong1.0-8B --backend vllm --gpu-memory-utilization 0.9

  # vLLM 后端 - 异步引擎（推荐，流式性能更好）
  python server.py --model-path ./models/Sinong1.0-8B --backend vllm --vllm-async
"""

import os
import sys
import json
import time
import uuid
import argparse
import re
from typing import Optional, List, Dict, Any, AsyncGenerator

from loguru import logger
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

# ============ 日志配置 ============
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True,
)
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logger.add(
    os.path.join(LOG_DIR, "server_{time:YYYY-MM-DD}.log"),
    rotation="1 day",
    retention="30 days",
    encoding="utf-8",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG",
)
logger.info("日志系统初始化完成")

# ============ 全局变量 ============
BACKEND: Optional[str] = None  # "transformers" 或 "vllm"

# Transformers 后端全局
MODEL = None
TOKENIZER = None

# vLLM 后端全局
LLM_ENGINE = None       # 同步引擎
ASYNC_LLM_ENGINE = None  # 异步引擎

MODEL_PATH: Optional[str] = None

# 默认系统提示词 - 防止"根据参考资料"幻觉
DEFAULT_SYSTEM_PROMPT = (
    "你是司农(Sinong)，一个专业的农业领域智能助手。"
    "请直接回答用户的问题，不要提及或引用任何参考资料。"
    "如果问题涉及农业领域，请基于你的专业知识详细回答。"
    "如果问题不涉及农业，也请尽力给出有用的回答。"
)


# ============================================================
# 请求/响应模型
# ============================================================

class ChatMessage(BaseModel):
    role: str
    content: str
    reasoning_content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = "Sinong1.0-8B"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.6
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 20
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False
    enable_thinking: Optional[bool] = False
    include_think: Optional[bool] = False


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


# ============================================================
# 通用工具
# ============================================================

def _inject_system_prompt(messages: List[Dict]) -> List[Dict]:
    """确保消息列表包含系统提示词"""
    if not messages:
        return [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
    if messages[0].get("role") == "system":
        return messages
    return [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}] + messages


def _parse_think_answer(raw_text: str) -> tuple:
    """解析 Qwen3 的思考/回答内容"""
    think_content = ""
    answer_content = raw_text.strip()

    # <think!>...</think!>
    think_match = re.search(r'<think!>(.*?)</think!>', raw_text, re.DOTALL)
    if think_match:
        think_content = think_match.group(1).strip()
        remaining = raw_text[think_match.end():].strip()
        if remaining:
            answer_content = remaining

    # <answer>...</answer>
    answer_match = re.search(r'<answer>(.*?)</answer>', answer_content, re.DOTALL)
    if answer_match:
        answer_content = answer_match.group(1).strip()

    # 兼容 <think</think
    if not think_content:
        think_match2 = re.search(r'<think(.*?)</think', raw_text, re.DOTALL)
        if think_match2:
            think_content = think_match2.group(1).strip()
            remaining = raw_text[think_match2.end():].strip()
            if remaining:
                answer_content = remaining.lstrip('>').strip()

    return answer_content, think_content


def _sse_chunk(chat_id: str, model: str, delta: Dict, finish_reason: Optional[str] = None) -> str:
    """构造 OpenAI 兼容的 SSE chunk"""
    chunk = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


def _model_loaded() -> bool:
    """检查模型是否已加载"""
    if BACKEND == "transformers":
        return MODEL is not None
    elif BACKEND == "vllm":
        return LLM_ENGINE is not None or ASYNC_LLM_ENGINE is not None
    return False


# ============================================================
# Transformers 后端
# ============================================================

def load_model_transformers(model_path: str, device: str = "auto"):
    """加载模型 - Transformers 后端"""
    global MODEL, TOKENIZER, BACKEND
    BACKEND = "transformers"

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"[Transformers] 正在加载模型: {model_path}")
    logger.info(f"[Transformers] 设备: {device}")

    TOKENIZER = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    MODEL = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device,
        trust_remote_code=True,
    )
    MODEL.eval()
    logger.info(f"[Transformers] 模型加载完成，设备: {MODEL.device}")


def generate_transformers(
    messages: List[Dict],
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    max_tokens: int = 2048,
    enable_thinking: bool = False,
) -> tuple:
    """Transformers 后端 - 非流式生成"""
    import torch

    messages = _inject_system_prompt(messages)
    text = TOKENIZER.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking,
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

    generated_ids = outputs[0][prompt_tokens:]
    completion_tokens = len(generated_ids)
    response_text = TOKENIZER.decode(generated_ids, skip_special_tokens=True)
    answer_content, think_content = _parse_think_answer(response_text)

    return answer_content, think_content, prompt_tokens, completion_tokens


def generate_stream_transformers(
    messages: List[Dict],
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    max_tokens: int = 2048,
    enable_thinking: bool = False,
):
    """Transformers 后端 - 流式生成，返回 (streamer, thread)"""
    import torch
    from transformers import TextIteratorStreamer
    from threading import Thread

    messages = _inject_system_prompt(messages)
    text = TOKENIZER.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking,
    )
    inputs = TOKENIZER([text], return_tensors="pt").to(MODEL.device)

    streamer = TextIteratorStreamer(TOKENIZER, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = {
        **inputs,
        "max_new_tokens": max_tokens,
        "temperature": temperature if temperature > 0 else 1.0,
        "top_p": top_p,
        "top_k": top_k,
        "do_sample": temperature > 0,
        "streamer": streamer,
        "pad_token_id": TOKENIZER.eos_token_id[0] if isinstance(TOKENIZER.eos_token_id, list) else TOKENIZER.eos_token_id,
    }

    thread = Thread(target=MODEL.generate, kwargs=generation_kwargs)
    thread.start()
    return streamer, thread


# ============================================================
# vLLM 后端
# ============================================================

def load_model_vllm(
    model_path: str,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 40960,
    use_async: bool = False,
):
    """加载模型 - vLLM 后端"""
    global LLM_ENGINE, ASYNC_LLM_ENGINE, BACKEND
    BACKEND = "vllm"

    if use_async:
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        from vllm.engine.arg_utils import AsyncEngineArgs

        logger.info(f"[vLLM Async] 正在加载模型: {model_path}")
        logger.info(f"[vLLM Async] gpu_memory_utilization={gpu_memory_utilization}, max_model_len={max_model_len}")

        engine_args = AsyncEngineArgs(
            model=model_path,
            trust_remote_code=True,
            dtype="bfloat16",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enable_reasoning=True,
        )
        ASYNC_LLM_ENGINE = AsyncLLMEngine.from_engine_args(engine_args)
        logger.info("[vLLM Async] 异步引擎加载完成")
    else:
        from vllm import LLM

        logger.info(f"[vLLM] 正在加载模型: {model_path}")
        logger.info(f"[vLLM] gpu_memory_utilization={gpu_memory_utilization}, max_model_len={max_model_len}")

        LLM_ENGINE = LLM(
            model=model_path,
            trust_remote_code=True,
            dtype="bfloat16",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enable_reasoning=True,
        )
        logger.info("[vLLM] 同步引擎加载完成")


def generate_vllm(
    messages: List[Dict],
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    max_tokens: int = 2048,
    enable_thinking: bool = False,
) -> tuple:
    """vLLM 同步引擎 - 非流式生成"""
    from vllm import SamplingParams

    messages = _inject_system_prompt(messages)

    sampling_params = SamplingParams(
        temperature=temperature if temperature > 0 else 1.0,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        stop=["<|im_end|>"],
    )

    outputs = LLM_ENGINE.chat(messages=messages, sampling_params=sampling_params)
    output = outputs[0]

    prompt_tokens = len(output.prompt_token_ids)
    completion_tokens = len(output.outputs[0].token_ids)
    raw_text = output.outputs[0].text

    answer_content, think_content = _parse_think_answer(raw_text)
    return answer_content, think_content, prompt_tokens, completion_tokens


async def generate_stream_vllm_async(
    messages: List[Dict],
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    max_tokens: int = 2048,
    enable_thinking: bool = False,
    chat_id: str = "",
    model_name: str = "Sinong1.0-8B",
) -> AsyncGenerator[str, None]:
    """vLLM AsyncEngine 后端 - 异步流式生成"""
    from vllm import SamplingParams

    messages = _inject_system_prompt(messages)

    sampling_params = SamplingParams(
        temperature=temperature if temperature > 0 else 1.0,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        stop=["<|im_end|>"],
    )

    # 通过 chat 接口发起异步生成
    results_generator = ASYNC_LLM_ENGINE.chat(
        messages=messages,
        sampling_params=sampling_params,
        request_id=chat_id,
    )

    yield _sse_chunk(chat_id, model_name, {"role": "assistant"})

    collected_text = ""
    try:
        async for request_output in results_generator:
            for output in request_output.outputs:
                new_text = output.text[len(collected_text):]
                if new_text:
                    collected_text += new_text
                    yield _sse_chunk(chat_id, model_name, {"content": new_text})
    except Exception as e:
        logger.error(f"[vLLM Async] 流式生成异常: {e}")
        yield _sse_chunk(chat_id, model_name, {"content": f"\n[Error: {str(e)}]"})

    yield _sse_chunk(chat_id, model_name, {}, finish_reason="stop")
    yield "data: [DONE]\n\n"


async def generate_stream_vllm_sync(
    messages: List[Dict],
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    max_tokens: int = 2048,
    enable_thinking: bool = False,
    chat_id: str = "",
    model_name: str = "Sinong1.0-8B",
) -> AsyncGenerator[str, None]:
    """vLLM 同步引擎 - 模拟流式输出（逐 token 从完整结果中切片）"""
    logger.warning("[vLLM] 使用同步引擎模拟流式输出，建议加 --vllm-async 使用异步引擎")

    # 同步生成完整结果
    answer, think, prompt_tokens, completion_tokens = generate_vllm(
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        enable_thinking=enable_thinking,
    )

    yield _sse_chunk(chat_id, model_name, {"role": "assistant"})

    # 逐字符模拟流式（按中文字符/英文单词粒度）
    chunk_size = 4  # 每 4 个字符一个 chunk
    for i in range(0, len(answer), chunk_size):
        text_chunk = answer[i:i + chunk_size]
        yield _sse_chunk(chat_id, model_name, {"content": text_chunk})
        # 模拟逐 token 延迟
        import asyncio
        await asyncio.sleep(0.02)

    yield _sse_chunk(chat_id, model_name, {}, finish_reason="stop")
    yield "data: [DONE]\n\n"


# ============================================================
# FastAPI 应用
# ============================================================

app = FastAPI(title="Sinong1.0-8B API Server", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    logger.info(f"FastAPI 应用启动 | 后端: {BACKEND}")


# ============================================================
# API 端点
# ============================================================

@app.get("/v1/models")
async def list_models():
    """列出可用模型"""
    return {
        "object": "list",
        "data": [{
            "id": "Sinong1.0-8B",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "NAULLM",
        }],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI 兼容的聊天补全接口（支持流式/非流式，双后端）"""
    if not _model_loaded():
        raise HTTPException(status_code=503, detail=f"模型尚未加载完成 (backend={BACKEND})")

    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    logger.info(
        f"请求 | backend={BACKEND}, stream={request.stream}, "
        f"messages={len(messages)}条, thinking={request.enable_thinking}, "
        f"max_tokens={request.max_tokens}"
    )

    # ============ Transformers 后端 ============
    if BACKEND == "transformers":
        if request.stream:
            chat_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

            async def stream_tf():
                streamer, thread = generate_stream_transformers(
                    messages=messages,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    max_tokens=request.max_tokens,
                    enable_thinking=request.enable_thinking,
                )
                yield _sse_chunk(chat_id, request.model, {"role": "assistant"})
                try:
                    for text in streamer:
                        if text:
                            yield _sse_chunk(chat_id, request.model, {"content": text})
                except Exception as e:
                    logger.error(f"[Transformers] 流式异常: {e}")
                    yield _sse_chunk(chat_id, request.model, {"content": f"\n[Error: {str(e)}]"})
                yield _sse_chunk(chat_id, request.model, {}, finish_reason="stop")
                yield "data: [DONE]\n\n"
                thread.join()

            return StreamingResponse(
                stream_tf(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
            )

        # 非流式
        try:
            answer, think, prompt_tokens, completion_tokens = generate_transformers(
                messages=messages,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                max_tokens=request.max_tokens,
                enable_thinking=request.enable_thinking,
            )
        except Exception as e:
            logger.error(f"[Transformers] 生成异常: {e}")
            raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}")

    # ============ vLLM 后端 ============
    elif BACKEND == "vllm":
        if request.stream:
            chat_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

            if ASYNC_LLM_ENGINE is not None:
                # 异步引擎 - 真正的流式
                return StreamingResponse(
                    generate_stream_vllm_async(
                        messages=messages,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        top_k=request.top_k,
                        max_tokens=request.max_tokens,
                        enable_thinking=request.enable_thinking,
                        chat_id=chat_id,
                        model_name=request.model,
                    ),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
                )
            else:
                # 同步引擎 - 模拟流式
                return StreamingResponse(
                    generate_stream_vllm_sync(
                        messages=messages,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        top_k=request.top_k,
                        max_tokens=request.max_tokens,
                        enable_thinking=request.enable_thinking,
                        chat_id=chat_id,
                        model_name=request.model,
                    ),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
                )

        # 非流式
        try:
            answer, think, prompt_tokens, completion_tokens = generate_vllm(
                messages=messages,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                max_tokens=request.max_tokens,
                enable_thinking=request.enable_thinking,
            )
        except Exception as e:
            logger.error(f"[vLLM] 生成异常: {e}")
            raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}")

    else:
        raise HTTPException(status_code=500, detail=f"未知后端: {BACKEND}")

    # 构建非流式响应（两种后端共用）
    msg = ChatMessage(role="assistant", content=answer)
    if think and request.include_think:
        msg.reasoning_content = think

    response = ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model=request.model,
        choices=[ChatCompletionChoice(index=0, message=msg, finish_reason="stop")],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )

    logger.info(f"响应 | tokens: prompt={prompt_tokens}, completion={completion_tokens}")
    return response


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "Sinong1.0-8B",
        "backend": BACKEND,
        "loaded": _model_loaded(),
    }


# ============================================================
# 启动入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sinong1.0-8B API Server")

    # 通用参数
    parser.add_argument("--model-path", type=str, required=True, help="模型本地路径")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    parser.add_argument("--backend", type=str, default="transformers",
                        choices=["transformers", "vllm"], help="推理后端 (transformers/vllm)")
    parser.add_argument("--system-prompt", type=str, default=None, help="自定义系统提示词")

    # Transformers 专用参数
    parser.add_argument("--device", type=str, default="auto",
                        help="设备 (auto/cuda/cpu/mps)，仅 transformers 后端")

    # vLLM 专用参数
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                        help="GPU 显存利用率，仅 vllm 后端")
    parser.add_argument("--max-model-len", type=int, default=40960,
                        help="最大上下文长度，仅 vllm 后端")
    parser.add_argument("--vllm-async", action="store_true",
                        help="使用 vLLM 异步引擎（推荐流式场景），仅 vllm 后端")

    args = parser.parse_args()

    if args.system_prompt:
        DEFAULT_SYSTEM_PROMPT = args.system_prompt

    # 加载模型
    if args.backend == "transformers":
        load_model_transformers(args.model_path, device=args.device)
    elif args.backend == "vllm":
        load_model_vllm(
            args.model_path,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            use_async=args.vllm_async,
        )

    logger.info(f"服务启动: backend={args.backend}, host={args.host}, port={args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
