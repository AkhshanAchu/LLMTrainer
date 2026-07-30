from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.config import get_settings
from app.core.db import get_session
from app.models.db import Dataset, DatasetFormat
from app.services import dataset_service as svc

router = APIRouter()


# --------------------------------------------------------------------------
# Request/response schemas
# --------------------------------------------------------------------------


class DatasetSummary(BaseModel):
    id: int
    name: str
    detected_format: DatasetFormat
    num_examples: int
    created_at: str
    quality_score: float | None = None


class DatasetDetail(BaseModel):
    id: int
    name: str
    source_filename: str
    stored_path: str
    detected_format: DatasetFormat
    num_examples: int
    stats: dict
    validation_report: dict
    created_at: str
    updated_at: str


class TokenizePreviewRequest(BaseModel):
    tokenizer_repo_id: str | None = None
    max_seq_len: int = 2048
    n: int = 50


class RepairRequest(BaseModel):
    options: dict[str, Any] = {}
    new_name: str | None = None


class RuntimeEstimateRequest(BaseModel):
    seq_len: int = 2048
    batch_size: int = 1
    epochs: float = 1.0
    tokenizer_repo_id: str | None = None
    tokens_per_sec: float = svc.DEFAULT_THROUGHPUT_TOK_S
    target_tokens: int | None = None


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _load_records(stored_path: str) -> list[dict]:
    return svc.parse_file(Path(stored_path))


def _dataset_to_detail(ds: Dataset) -> DatasetDetail:
    return DatasetDetail(
        id=ds.id,
        name=ds.name,
        source_filename=ds.source_filename,
        stored_path=ds.stored_path,
        detected_format=ds.detected_format,
        num_examples=ds.num_examples,
        stats=ds.stats,
        validation_report=ds.validation_report,
        created_at=ds.created_at.isoformat(),
        updated_at=ds.updated_at.isoformat(),
    )


async def _get_dataset_or_404(dataset_id: int, session: AsyncSession) -> Dataset:
    ds = await session.get(Dataset, dataset_id)
    if ds is None:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    return ds


def _build_dataset_row(
    name: str, source_filename: str, stored_path: Path, records: list[dict]
) -> Dataset:
    detection = svc.detect_format(records)
    fmt = detection.format
    stats = svc.compute_stats(records, fmt)
    stats["format_detection"] = {
        "format": fmt.value,
        "confidence": detection.confidence,
        "signals": detection.signals,
    }
    validation_report = svc.validate_dataset(records, fmt)
    return Dataset(
        name=name,
        source_filename=source_filename,
        stored_path=str(stored_path),
        detected_format=fmt,
        num_examples=len(records),
        stats=stats,
        validation_report=validation_report,
    )


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------


@router.post("/upload", response_model=DatasetDetail)
async def upload_dataset(
    file: UploadFile,
    session: AsyncSession = Depends(get_session),
) -> DatasetDetail:
    settings = get_settings()
    content = await file.read()

    def _ingest() -> Dataset:
        stored_path = svc.save_upload(settings.uploads_dir, file.filename or "upload.dat", content)
        try:
            records = svc.parse_file(stored_path)
        except svc.UnsupportedFileTypeError as e:
            stored_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=str(e))
        if not records:
            stored_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="No parseable records found in file")
        return _build_dataset_row(
            name=Path(file.filename or "dataset").stem,
            source_filename=file.filename or "upload.dat",
            stored_path=stored_path,
            records=records,
        )

    dataset = await run_in_threadpool(_ingest)
    session.add(dataset)
    await session.commit()
    await session.refresh(dataset)
    return _dataset_to_detail(dataset)


@router.get("/", response_model=list[DatasetSummary])
async def list_datasets(session: AsyncSession = Depends(get_session)) -> list[DatasetSummary]:
    result = await session.exec(select(Dataset).order_by(Dataset.created_at.desc()))
    datasets = result.all()
    out = []
    for ds in datasets:
        out.append(
            DatasetSummary(
                id=ds.id,
                name=ds.name,
                detected_format=ds.detected_format,
                num_examples=ds.num_examples,
                created_at=ds.created_at.isoformat(),
                quality_score=(ds.stats or {}).get("quality_score"),
            )
        )
    return out


@router.get("/{dataset_id}", response_model=DatasetDetail)
async def get_dataset(dataset_id: int, session: AsyncSession = Depends(get_session)) -> DatasetDetail:
    ds = await _get_dataset_or_404(dataset_id, session)
    return _dataset_to_detail(ds)


@router.get("/{dataset_id}/preview")
async def preview_dataset(
    dataset_id: int,
    n: int = Query(20, ge=1, le=1000),
    format: str = Query("raw", pattern="^(raw|chat)$"),
    session: AsyncSession = Depends(get_session),
) -> dict:
    ds = await _get_dataset_or_404(dataset_id, session)

    def _preview() -> dict:
        records = _load_records(ds.stored_path)[:n]
        if format == "raw":
            return {"format": "raw", "records": records}
        chat_records = [svc.to_chat_messages(r, ds.detected_format) for r in records]
        return {"format": "chat", "records": chat_records}

    return await run_in_threadpool(_preview)


@router.post("/{dataset_id}/tokenize-preview")
async def tokenize_preview(
    dataset_id: int,
    body: TokenizePreviewRequest,
    session: AsyncSession = Depends(get_session),
) -> dict:
    ds = await _get_dataset_or_404(dataset_id, session)

    def _run() -> dict:
        records = _load_records(ds.stored_path)
        sample = records[: body.n]
        return svc.analyze_with_tokenizer(
            sample,
            ds.detected_format,
            tokenizer_repo_id=body.tokenizer_repo_id,
            max_seq_len=body.max_seq_len,
        )

    result = await run_in_threadpool(_run)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.post("/{dataset_id}/estimate-runtime")
async def estimate_runtime(
    dataset_id: int,
    body: RuntimeEstimateRequest,
    session: AsyncSession = Depends(get_session),
) -> dict:
    ds = await _get_dataset_or_404(dataset_id, session)

    def _run() -> dict:
        records = _load_records(ds.stored_path)
        return svc.estimate_dataset_runtime(
            records,
            ds.detected_format,
            seq_len=body.seq_len,
            batch_size=body.batch_size,
            epochs=body.epochs,
            tokenizer_repo_id=body.tokenizer_repo_id,
            tokens_per_sec=body.tokens_per_sec,
            target_tokens=body.target_tokens,
        )

    result = await run_in_threadpool(_run)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.post("/{dataset_id}/repair", response_model=DatasetDetail)
async def repair_dataset(
    dataset_id: int,
    body: RepairRequest,
    session: AsyncSession = Depends(get_session),
) -> DatasetDetail:
    ds = await _get_dataset_or_404(dataset_id, session)
    settings = get_settings()

    def _run() -> tuple[Dataset, dict]:
        records = _load_records(ds.stored_path)
        repaired, summary = svc.auto_repair(records, ds.detected_format, body.options)
        if not repaired:
            raise HTTPException(status_code=400, detail="Repair produced zero usable examples")

        import json as _json

        content = "\n".join(_json.dumps(r, ensure_ascii=False) for r in repaired).encode("utf-8")
        original_name = Path(ds.source_filename).stem
        new_filename = f"{original_name}_repaired.jsonl"
        stored_path = svc.save_upload(settings.uploads_dir, new_filename, content)

        new_records = repaired  # already {"messages": [...]} shape -> chatml-like
        new_ds = _build_dataset_row(
            name=f"{ds.name} (repaired)",
            source_filename=new_filename,
            stored_path=stored_path,
            records=new_records,
        )
        new_ds.stats["repair_summary"] = summary
        new_ds.stats["repaired_from_dataset_id"] = ds.id
        return new_ds, summary

    new_ds, _summary = await run_in_threadpool(_run)
    session.add(new_ds)
    await session.commit()
    await session.refresh(new_ds)
    return _dataset_to_detail(new_ds)


@router.delete("/{dataset_id}")
async def delete_dataset(dataset_id: int, session: AsyncSession = Depends(get_session)) -> dict:
    ds = await _get_dataset_or_404(dataset_id, session)
    stored_path = Path(ds.stored_path)
    if stored_path.exists():
        stored_path.unlink(missing_ok=True)
    await session.delete(ds)
    await session.commit()
    return {"deleted": True, "id": dataset_id}
