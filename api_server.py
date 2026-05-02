# api_server.py - 部署 LoRA 为 API 服务
import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
from peft import PeftModel
from threading import Thread
import uvicorn
import json

# ============================================================
# 配置
# ============================================================
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"      # 基础模型
ADAPTER_PATH = "output/qwen2.5-7b-lora"        # LoRA adapter 路径
USE_QLORA = True                               # 是否使用 4-bit 量化
PORT = 8000                                    # API 端口

# ============================================================

print("🚀 正在加载模型...")

# 加载 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    local_files_only=True,
)

# 加载基础模型
if USE_QLORA:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        local_files_only=True,
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        local_files_only=True,
    )

# 加载 LoRA adapter
model = PeftModel.from_pretrained(model, ADAPTER_PATH, local_files_only=True)
model.eval()

print(f"✅ 模型加载完成，准备就绪！")
print(f"📁 Adapter: {ADAPTER_PATH}")
print(f"🔌 设备: {model.device}")

# 创建 FastAPI 应用
app = FastAPI(title="LoRA Model API", version="1.0")

# 添加 CORS 中间件允许前端跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头部
)


class ChatRequest(BaseModel):
    message: str
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9


class ChatResponse(BaseModel):
    response: str


@app.get("/")
def root():
    return {"status": "ok", "model": "LoRA Fine-tuned Qwen2.5-7B"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """单轮对话"""
    messages = [{"role": "user", "content": request.message}]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_new_tokens,
            do_sample=True,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=1.1,
        )

    response_text = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # 提取助手回复
    if "<|im_start|>assistant" in response_text:
        response = response_text.split("<|im_start|>assistant")[-1].replace("左", "").replace("<|endoftext|>", "").strip()
    else:
        response = response_text[len(prompt):].strip()

    return ChatResponse(response=response)


@app.post("/chat/history")
def chat_with_history(history: list[dict]):
    """多轮对话（支持历史记录）

    history 格式: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
    """
    prompt = tokenizer.apply_chat_template(
        history,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
        )

    response_text = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # 提取最新助手回复
    if "<|im_start|>assistant" in response_text:
        parts = response_text.split("<|im_start|>assistant")
        response = parts[-1].replace("左", "").replace("<|endoftext|>", "").strip()
    else:
        response = response_text[len(prompt):].strip()

    return {"response": response, "history": history + [{"role": "assistant", "content": response}]}


@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    """单轮对话 - 流式输出"""
    messages = [{"role": "user", "content": request.message}]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=request.max_new_tokens,
        do_sample=True,
        temperature=request.temperature,
        top_p=request.top_p,
        repetition_penalty=1.1,
        streamer=streamer,
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    def generate():
        for text in streamer:
            if text:
                yield f"data: {json.dumps({'text': text}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'text': '', 'done': True}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/chat/history/stream")
def chat_history_stream(history: list[dict]):
    """多轮对话 - 流式输出

    history 格式: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
    """
    prompt = tokenizer.apply_chat_template(
        history,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        streamer=streamer,
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    def generate():
        full_response = ""
        for text in streamer:
            if text:
                full_response += text
                yield f"data: {json.dumps({'text': text}, ensure_ascii=False)}\n\n"
        # 构建历史记录
        assistant_msg = {"role": "assistant", "content": full_response}
        new_history = history + [assistant_msg]
        data = json.dumps({'text': '', 'done': True, 'history': new_history}, ensure_ascii=False)
        yield f"data: {data}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


if __name__ == "__main__":
    print(f"\n🚀 启动 API 服务: http://localhost:{PORT}")
    print(f"   API 文档: http://localhost:{PORT}/docs\n")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
