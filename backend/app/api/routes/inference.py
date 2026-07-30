import json

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.services import inference_service

router = APIRouter()


class LoadModelRequest(BaseModel):
    repo_id: str
    adapter_path: str | None = None
    load_in_4bit: bool = False


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    system_prompt: str | None = None
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    min_p: float = 0.0
    repetition_penalty: float = 1.1


class BenchmarkRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 256


@router.post("/load")
async def load_model(payload: LoadModelRequest):
    try:
        await run_in_threadpool(
            inference_service.load_model, payload.repo_id, payload.adapter_path, payload.load_in_4bit
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"Failed to load model: {e}")
    return {"status": "loaded", "repo_id": payload.repo_id}


@router.post("/unload")
async def unload_model():
    await run_in_threadpool(inference_service.unload_model)
    return {"status": "unloaded"}


@router.get("/current")
async def current_model():
    return {"repo_id": inference_service.get_current_model_id()}


@router.post("/chat/stream")
async def chat_stream(payload: ChatRequest):
    if inference_service.get_current_model_id() is None:
        raise HTTPException(400, "No model loaded. Call /inference/load first.")

    messages = [m.model_dump() for m in payload.messages]

    def sse_generator():
        gen = inference_service.stream_generate(
            messages,
            system_prompt=payload.system_prompt,
            max_new_tokens=payload.max_new_tokens,
            temperature=payload.temperature,
            top_p=payload.top_p,
            top_k=payload.top_k,
            min_p=payload.min_p,
            repetition_penalty=payload.repetition_penalty,
        )
        for text, is_final, stats in gen:
            if is_final:
                yield f"data: {json.dumps({'done': True, 'stats': stats})}\n\n"
            else:
                yield f"data: {json.dumps({'token': text})}\n\n"

    return StreamingResponse(sse_generator(), media_type="text/event-stream")


@router.post("/benchmark")
async def benchmark(payload: BenchmarkRequest):
    if inference_service.get_current_model_id() is None:
        raise HTTPException(400, "No model loaded. Call /inference/load first.")
    result = await run_in_threadpool(inference_service.benchmark_generation, payload.prompt, payload.max_new_tokens)
    return result
