import asyncio

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.db import get_session
from app.core.events import event_bus
from app.models.db import Dataset, JobStatus, TrainingRun
from app.schemas.training import HyperparamSuggestion, MetricPointOut, TrainingRunCreate, TrainingRunOut
from app.services import training_service
from app.services.hparam_service import suggest_hyperparams
from app.services.model_service import get_model_info

router = APIRouter()


async def _estimate_params_from_repo(repo_id: str) -> int | None:
    info = await run_in_threadpool(get_model_info, repo_id)
    return info.get("param_count")


def _to_out(run: TrainingRun) -> TrainingRunOut:
    return TrainingRunOut(
        id=run.id,
        name=run.name,
        base_model_repo_id=run.base_model_repo_id,
        dataset_id=run.dataset_id,
        method=run.method,
        objective=run.objective,
        status=run.status,
        config=run.config,
        output_dir=run.output_dir,
        current_step=run.current_step,
        total_steps=run.total_steps,
        current_epoch=run.current_epoch,
        latest_metrics=run.latest_metrics,
        error_message=run.error_message,
    )


@router.post("", response_model=TrainingRunOut)
async def create_training_run(payload: TrainingRunCreate, session: AsyncSession = Depends(get_session)):
    cfg = payload.config
    dataset = await session.get(Dataset, cfg.dataset_id)
    if dataset is None:
        raise HTTPException(404, f"Dataset {cfg.dataset_id} not found")

    dataset_path = dataset.stored_path
    dataset_format = dataset.detected_format.value if hasattr(dataset.detected_format, "value") else dataset.detected_format

    run = TrainingRun(
        name=cfg.name,
        base_model_repo_id=cfg.base_model_repo_id,
        dataset_id=cfg.dataset_id,
        method=cfg.method,
        objective=cfg.objective,
        status=JobStatus.queued,
        config=cfg.model_dump(mode="json"),
    )
    session.add(run)
    await session.commit()

    worker_config = cfg.model_dump(mode="json")
    worker_config["dataset_path"] = dataset_path
    worker_config["dataset_format"] = dataset_format

    asyncio.create_task(training_service.start_training_run(run.id, worker_config))

    return _to_out(run)


@router.get("", response_model=list[TrainingRunOut])
async def list_training_runs(session: AsyncSession = Depends(get_session)):
    result = await session.exec(select(TrainingRun).order_by(TrainingRun.created_at.desc()))
    return [_to_out(r) for r in result.all()]


@router.get("/{run_id}", response_model=TrainingRunOut)
async def get_training_run(run_id: int, session: AsyncSession = Depends(get_session)):
    run = await session.get(TrainingRun, run_id)
    if run is None:
        raise HTTPException(404, "Training run not found")
    return _to_out(run)


@router.get("/{run_id}/metrics", response_model=list[MetricPointOut])
async def get_training_metrics(run_id: int, session: AsyncSession = Depends(get_session)):
    points = await training_service.get_metric_history(session, run_id)
    return [
        MetricPointOut(
            step=p.step,
            epoch=p.epoch,
            loss=p.loss,
            eval_loss=p.eval_loss,
            learning_rate=p.learning_rate,
            grad_norm=p.grad_norm,
            tokens_per_sec=p.tokens_per_sec,
            samples_per_sec=p.samples_per_sec,
            gpu_mem_used_gb=p.gpu_mem_used_gb,
        )
        for p in points
    ]


@router.post("/{run_id}/cancel")
async def cancel_training_run(run_id: int):
    cancelled = await training_service.cancel_training_run(run_id)
    if not cancelled:
        raise HTTPException(400, "Run is not active")
    return {"status": "cancelled"}


@router.websocket("/{run_id}/ws")
async def training_ws(websocket: WebSocket, run_id: int):
    await websocket.accept()
    topic = f"training_{run_id}"
    queue = event_bus.subscribe(topic)
    try:
        while True:
            payload = await queue.get()
            await websocket.send_json(payload)
    except WebSocketDisconnect:
        pass
    finally:
        event_bus.unsubscribe(topic, queue)


@router.get("/hparams/suggest", response_model=HyperparamSuggestion)
async def suggest_hparams(repo_id: str, available_vram_gb: float = 8.0):
    param_count = await _estimate_params_from_repo(repo_id)
    if param_count is None:
        raise HTTPException(400, f"Could not estimate parameter count for {repo_id}")
    return suggest_hyperparams(param_count, available_vram_gb)
