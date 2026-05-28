"""Minimal OpenAI-compatible server wrapping the trained qwen-pii-v6
LoRA adapter, so the BrewSLM Playground can call the real model live.

Loads Qwen2.5-1.5B-Instruct + the project-3/experiment-12 LoRA adapter
once at startup. Exposes ``/v1/chat/completions`` in the shape the
playground's ``openai_compatible`` provider expects. Wraps the user's
message in the exact format the model was trained on
(``Input: <text>\\nStructured Output:``) and returns the generated
span-JSON array as the assistant message.

Run:
  python tts/pii_model_server.py            # serves on 127.0.0.1:5009
"""

from __future__ import annotations

import time

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


BASE = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER = "data/projects/3/experiments/12/model"
HOST = "127.0.0.1"
PORT = 5009

print("[pii-server] loading base + LoRA…", flush=True)
_t0 = time.time()
_device = "cuda" if torch.cuda.is_available() else "cpu"
_tok = AutoTokenizer.from_pretrained(BASE)
_model = AutoModelForCausalLM.from_pretrained(
    BASE,
    dtype=torch.bfloat16 if _device == "cuda" else torch.float32,
    device_map=_device,
)
_model = PeftModel.from_pretrained(_model, ADAPTER).eval()
print(f"[pii-server] loaded in {time.time() - _t0:.1f}s on {_device}", flush=True)


app = FastAPI()


class _Msg(BaseModel):
    role: str
    content: str


class _ChatRequest(BaseModel):
    model: str | None = None
    messages: list[_Msg]
    temperature: float | None = 0.0
    max_tokens: int | None = 128
    stream: bool | None = False


def _last_user_text(messages: list[_Msg]) -> str:
    for m in reversed(messages):
        if m.role == "user":
            return m.content
    return messages[-1].content if messages else ""


def _generate(text: str, max_tokens: int) -> str:
    # Exact trained format: the model continues after "Structured Output:".
    prompt = f"Input: {text}\nStructured Output:"
    inputs = _tok(prompt, return_tensors="pt").to(_device)
    with torch.inference_mode():
        out = _model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
        )
    gen = _tok.decode(
        out[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return gen.strip()


@app.get("/v1/models")
def list_models() -> dict:
    return {
        "object": "list",
        "data": [{"id": "qwen-pii-v6", "object": "model", "owned_by": "brewslm"}],
    }


@app.post("/v1/chat/completions")
def chat_completions(req: _ChatRequest) -> dict:
    user_text = _last_user_text(req.messages)
    t0 = time.time()
    reply = _generate(user_text, int(req.max_tokens or 128))
    elapsed = time.time() - t0
    return {
        "id": f"pii-{int(time.time())}",
        "object": "chat.completion",
        "model": req.model or "qwen-pii-v6",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": reply},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "latency_ms": round(elapsed * 1000, 1),
        },
    }


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
