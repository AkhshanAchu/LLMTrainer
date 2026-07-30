from functools import lru_cache

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

router = APIRouter()


@lru_cache(maxsize=8)
def _load_tokenizer(repo_id: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)


class EncodeRequest(BaseModel):
    repo_id: str
    text: str


class ChatTemplateRequest(BaseModel):
    repo_id: str
    messages: list[dict]
    add_generation_prompt: bool = True


@router.post("/encode")
async def encode(payload: EncodeRequest):
    try:
        tokenizer = await run_in_threadpool(_load_tokenizer, payload.repo_id)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Failed to load tokenizer for {payload.repo_id}: {e}")

    ids = tokenizer.encode(payload.text)
    tokens = [tokenizer.decode([i]) for i in ids]
    special_ids = set(tokenizer.all_special_ids)

    return {
        "token_count": len(ids),
        "tokens": [{"id": i, "text": t, "is_special": i in special_ids} for i, t in zip(ids, tokens)],
        "special_tokens": {
            "bos": tokenizer.bos_token,
            "eos": tokenizer.eos_token,
            "pad": tokenizer.pad_token,
            "unk": tokenizer.unk_token,
        },
        "vocab_size": tokenizer.vocab_size,
    }


@router.post("/chat-template")
async def chat_template(payload: ChatTemplateRequest):
    try:
        tokenizer = await run_in_threadpool(_load_tokenizer, payload.repo_id)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Failed to load tokenizer for {payload.repo_id}: {e}")

    if tokenizer.chat_template is None:
        raise HTTPException(400, f"{payload.repo_id} has no chat template defined")

    formatted = tokenizer.apply_chat_template(
        payload.messages, tokenize=False, add_generation_prompt=payload.add_generation_prompt
    )
    token_ids = tokenizer.apply_chat_template(payload.messages, add_generation_prompt=payload.add_generation_prompt)

    return {
        "formatted_prompt": formatted,
        "token_count": len(token_ids),
        "chat_template_raw": tokenizer.chat_template,
    }


@router.get("/vocab-lookup")
async def vocab_lookup(repo_id: str, token: str | None = None, token_id: int | None = None):
    try:
        tokenizer = await run_in_threadpool(_load_tokenizer, repo_id)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Failed to load tokenizer for {repo_id}: {e}")

    if token_id is not None:
        return {"id": token_id, "text": tokenizer.decode([token_id])}
    if token is not None:
        ids = tokenizer.encode(token, add_special_tokens=False)
        return {"token": token, "ids": ids}
    raise HTTPException(400, "Provide either token or token_id")
