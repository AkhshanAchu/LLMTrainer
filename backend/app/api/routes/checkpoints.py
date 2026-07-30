from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.config import get_settings
from app.core.db import get_session
from app.models.db import TrainingRun
from app.services import checkpoint_service

router = APIRouter()
settings = get_settings()


class CheckpointOut(BaseModel):
    id: int
    step: int
    epoch: float
    path: str
    size_bytes: int
    eval_loss: float | None
    is_best: bool


class MergeRequest(BaseModel):
    checkpoint_path: str | None = None  # defaults to run's "final" dir


class ExportRequest(BaseModel):
    checkpoint_path: str | None = None
    format: str = "safetensors"  # safetensors | gguf
    quant_type: str = "q4_k_m"


@router.get("/{run_id}", response_model=list[CheckpointOut])
async def list_run_checkpoints(run_id: int, session: AsyncSession = Depends(get_session)):
    checkpoints = await checkpoint_service.list_checkpoints(session, run_id)
    return [
        CheckpointOut(
            id=c.id, step=c.step, epoch=c.epoch, path=c.path, size_bytes=c.size_bytes,
            eval_loss=c.eval_loss, is_best=c.is_best,
        )
        for c in checkpoints
    ]


@router.post("/{run_id}/best")
async def compute_best_checkpoint(run_id: int, session: AsyncSession = Depends(get_session)):
    best = await checkpoint_service.mark_best_checkpoint(session, run_id)
    if best is None:
        raise HTTPException(404, "No checkpoints with eval_loss found for this run")
    return {"best_checkpoint_id": best.id, "step": best.step, "eval_loss": best.eval_loss}


@router.post("/{run_id}/export")
async def export_checkpoint(run_id: int, payload: ExportRequest, session: AsyncSession = Depends(get_session)):
    run = await session.get(TrainingRun, run_id)
    if run is None:
        raise HTTPException(404, "Training run not found")

    adapter_path = payload.checkpoint_path or str(Path(run.output_dir) / "final")
    if not Path(adapter_path).exists():
        raise HTTPException(404, f"Checkpoint path not found: {adapter_path}")

    export_root = settings.exports_dir / f"run_{run_id}"
    export_root.mkdir(parents=True, exist_ok=True)

    if payload.format == "safetensors":
        merged_dir = export_root / "merged"
        result = await run_in_threadpool(
            checkpoint_service.export_merged_safetensors, run.base_model_repo_id, adapter_path, str(merged_dir)
        )
    elif payload.format == "gguf":
        merged_dir = export_root / "merged"
        if not merged_dir.exists():
            merge_result = await run_in_threadpool(
                checkpoint_service.export_merged_safetensors, run.base_model_repo_id, adapter_path, str(merged_dir)
            )
            if not merge_result.get("success"):
                raise HTTPException(500, merge_result.get("error", "merge failed"))
        gguf_path = export_root / f"model.{payload.quant_type}.gguf"
        result = await run_in_threadpool(
            checkpoint_service.export_to_gguf, str(merged_dir), str(gguf_path), payload.quant_type
        )
    else:
        raise HTTPException(400, f"Unsupported export format: {payload.format}")

    if not result.get("success"):
        raise HTTPException(500, result.get("error", "export failed"))
    return result


@router.post("/{run_id}/cleanup")
async def cleanup_checkpoints(run_id: int, keep_last_n: int = 3, session: AsyncSession = Depends(get_session)):
    run = await session.get(TrainingRun, run_id)
    if run is None:
        raise HTTPException(404, "Training run not found")
    removed = await run_in_threadpool(checkpoint_service.cleanup_old_checkpoints, run.output_dir, keep_last_n)
    return {"removed": removed}
