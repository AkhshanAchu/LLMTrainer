from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from huggingface_hub.errors import HfHubHTTPError
from pydantic import BaseModel
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.db import SQLModelAsyncSession, engine, get_session
from app.models.db import ModelEntry
from app.schemas.common import VRAMEstimate
from app.services import model_service

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Request/response bodies
# ---------------------------------------------------------------------------


class VRAMEstimateRequest(BaseModel):
    repo_id: str | None = None
    param_count: int | None = None
    method: Literal["full", "lora", "qlora"] = "qlora"
    dtype: Literal["fp32", "fp16", "bf16", "int8", "nf4"] = "bf16"


class DownloadRequest(BaseModel):
    repo_id: str
    display_name: str | None = None


# ---------------------------------------------------------------------------
# Browse / search / info
# ---------------------------------------------------------------------------


@router.get("/curated")
async def get_curated_models():
    return model_service.CURATED_MODELS


@router.get("/search")
async def search_models(q: str = Query(..., min_length=1), limit: int = 20):
    try:
        return model_service.search_hub_models(query=q, limit=limit)
    except HfHubHTTPError as e:
        raise HTTPException(status_code=502, detail=f"Hugging Face Hub error: {e}") from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Failed to search Hugging Face Hub: {e}") from e


@router.get("/info/{repo_id:path}")
async def get_model_info(repo_id: str):
    try:
        info = model_service.get_model_info(repo_id)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Failed to fetch model info: {e}") from e

    param_count = info.get("param_count")
    vram_estimates = None
    if param_count:
        vram_estimates = {
            "lora": model_service.estimate_vram(param_count, "lora", "bf16"),
            "qlora": model_service.estimate_vram(param_count, "qlora", "nf4"),
            "full": model_service.estimate_vram(param_count, "full", "bf16"),
        }

    return {"info": info, "vram_estimates": vram_estimates}


@router.post("/vram-estimate", response_model=VRAMEstimate)
async def vram_estimate(body: VRAMEstimateRequest):
    param_count = body.param_count
    if param_count is None and body.repo_id:
        try:
            info = model_service.get_model_info(body.repo_id)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"Failed to fetch model info: {e}") from e
        param_count = info.get("param_count")

    if param_count is None:
        raise HTTPException(
            status_code=400,
            detail="param_count could not be determined; provide param_count or a repo_id with a readable config.",
        )

    return model_service.estimate_vram(param_count, body.method, body.dtype)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


async def _download_and_record(repo_id: str, display_name: str | None) -> None:
    try:
        local_path = await model_service.download_model(repo_id)
        try:
            info = model_service.get_model_info(repo_id)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to fetch model info for %s after download", repo_id)
            info = {}

        async with SQLModelAsyncSession(engine, expire_on_commit=False) as session:
            await model_service.create_or_update_entry_after_download(
                session, repo_id=repo_id, local_path=local_path, display_name=display_name, info=info
            )
    except Exception:  # noqa: BLE001
        logger.exception("Download failed for %s", repo_id)


@router.post("/download")
async def download_model(body: DownloadRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(_download_and_record, body.repo_id, body.display_name)
    return {"status": "started", "repo_id": body.repo_id}


# ---------------------------------------------------------------------------
# Library CRUD
# ---------------------------------------------------------------------------


@router.get("/library", response_model=list[ModelEntry])
async def get_library(
    bookmarked_only: bool = False,
    is_vision: bool | None = None,
    session: AsyncSession = Depends(get_session),
):
    return await model_service.list_models(session, bookmarked_only=bookmarked_only, is_vision=is_vision)


@router.post("/{model_id}/bookmark", response_model=ModelEntry)
async def toggle_bookmark(model_id: int, session: AsyncSession = Depends(get_session)):
    entry = await model_service.toggle_bookmark(session, model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return entry


@router.delete("/{model_id}")
async def delete_model(model_id: int, session: AsyncSession = Depends(get_session)):
    deleted = await model_service.delete_model(session, model_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Model not found")
    return {"status": "deleted", "model_id": model_id}
