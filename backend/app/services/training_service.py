"""Orchestrates training jobs as isolated OS subprocesses.

Each run gets its own Python process (via `python -m app.workers.train_worker`)
so that CUDA memory and any crash/OOM in the training loop can't take down the
API server. Progress is communicated back through a per-run JSONL status file
that an asyncio tail-task streams into the DB and the event bus.
"""

import asyncio
import json
import sys
from pathlib import Path

from sqlmodel import select

from app.core.config import get_settings
from app.core.db import engine
from app.core.events import event_bus
from app.models.db import Checkpoint, JobStatus, MetricPoint, TrainingRun
from sqlmodel.ext.asyncio.session import AsyncSession

settings = get_settings()

# run_id -> asyncio.subprocess.Process
_active_processes: dict[int, asyncio.subprocess.Process] = {}
_tail_tasks: dict[int, asyncio.Task] = {}


def _run_dir(run_id: int) -> Path:
    d = settings.checkpoints_dir / f"run_{run_id}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _status_path(run_id: int) -> Path:
    return _run_dir(run_id) / "status.jsonl"


def _config_path(run_id: int) -> Path:
    return _run_dir(run_id) / "config.json"


async def start_training_run(run_id: int, worker_config: dict) -> None:
    run_dir = _run_dir(run_id)
    status_path = _status_path(run_id)
    if status_path.exists():
        status_path.unlink()
    config_path = _config_path(run_id)
    worker_config["output_dir"] = str(run_dir)
    config_path.write_text(json.dumps(worker_config), encoding="utf-8")

    async with AsyncSession(engine, expire_on_commit=False) as session:
        run = await session.get(TrainingRun, run_id)
        if run is None:
            return
        run.status = JobStatus.preparing
        run.output_dir = str(run_dir)
        session.add(run)
        await session.commit()

    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "app.workers.train_worker",
        str(run_id),
        str(config_path),
        str(status_path),
        cwd=str(Path(__file__).resolve().parents[2]),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    _active_processes[run_id] = process
    _tail_tasks[run_id] = asyncio.create_task(_tail_run(run_id, process))


async def _tail_run(run_id: int, process: asyncio.subprocess.Process) -> None:
    status_path = _status_path(run_id)
    offset = 0
    stdout_lines: list[str] = []

    async def read_stdout():
        if process.stdout is None:
            return
        async for line in process.stdout:
            stdout_lines.append(line.decode(errors="replace"))
            if len(stdout_lines) > 500:
                stdout_lines.pop(0)

    stdout_task = asyncio.create_task(read_stdout())

    async with AsyncSession(engine, expire_on_commit=False) as session:
        run = await session.get(TrainingRun, run_id)
        if run is not None:
            run.status = JobStatus.running
            session.add(run)
            await session.commit()

    try:
        while True:
            if status_path.exists():
                text = status_path.read_text(encoding="utf-8")
                new_text = text[offset:]
                offset = len(text)
                for line in new_text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    await _handle_event(run_id, record)

            if process.returncode is not None:
                break
            await asyncio.sleep(0.5)

        await process.wait()
        await stdout_task

        async with AsyncSession(engine, expire_on_commit=False) as session:
            run = await session.get(TrainingRun, run_id)
            if run is not None and run.status not in (JobStatus.cancelled,):
                if process.returncode == 0:
                    run.status = JobStatus.completed
                else:
                    run.status = JobStatus.failed
                    if not run.error_message:
                        run.error_message = "".join(stdout_lines[-50:]) or "Worker process exited with an error."
                session.add(run)
                await session.commit()
        await event_bus.publish(f"training_{run_id}", {"event": "process_exit", "returncode": process.returncode})
    finally:
        _active_processes.pop(run_id, None)


async def _handle_event(run_id: int, record: dict) -> None:
    event = record.get("event")
    await event_bus.publish(f"training_{run_id}", record)

    async with AsyncSession(engine, expire_on_commit=False) as session:
        run = await session.get(TrainingRun, run_id)
        if run is None:
            return

        if event == "metric":
            point = MetricPoint(
                run_id=run_id,
                step=record.get("step", 0),
                epoch=record.get("epoch", 0.0),
                loss=record.get("loss"),
                eval_loss=record.get("eval_loss"),
                learning_rate=record.get("learning_rate"),
                grad_norm=record.get("grad_norm"),
                gpu_mem_used_gb=record.get("gpu_mem_used_gb"),
            )
            session.add(point)
            run.current_step = record.get("step", run.current_step)
            run.current_epoch = record.get("epoch", run.current_epoch)
            if record.get("total_steps"):
                run.total_steps = record["total_steps"]
            run.latest_metrics = {
                "loss": record.get("loss"),
                "eval_loss": record.get("eval_loss"),
                "learning_rate": record.get("learning_rate"),
                "grad_norm": record.get("grad_norm"),
                "gpu_mem_used_gb": record.get("gpu_mem_used_gb"),
            }
            session.add(run)

        elif event == "checkpoint":
            step = record.get("step", 0)
            ckpt_path = _run_dir(run_id) / f"checkpoint-{step}"
            size = 0
            if ckpt_path.exists():
                size = sum(f.stat().st_size for f in ckpt_path.rglob("*") if f.is_file())
            checkpoint = Checkpoint(
                run_id=run_id,
                step=step,
                epoch=record.get("epoch", 0.0),
                path=str(ckpt_path),
                size_bytes=size,
            )
            session.add(checkpoint)

        elif event == "error":
            run.status = JobStatus.failed
            run.error_message = record.get("message", "Unknown training error")
            session.add(run)

        elif event == "done":
            run.status = JobStatus.completed
            session.add(run)

        await session.commit()


async def cancel_training_run(run_id: int) -> bool:
    process = _active_processes.get(run_id)
    if process is None or process.returncode is not None:
        return False
    process.terminate()
    async with AsyncSession(engine, expire_on_commit=False) as session:
        run = await session.get(TrainingRun, run_id)
        if run is not None:
            run.status = JobStatus.cancelled
            session.add(run)
            await session.commit()
    return True


def is_run_active(run_id: int) -> bool:
    process = _active_processes.get(run_id)
    return process is not None and process.returncode is None


async def get_metric_history(session, run_id: int) -> list[MetricPoint]:
    result = await session.exec(select(MetricPoint).where(MetricPoint.run_id == run_id).order_by(MetricPoint.step))
    return list(result.all())
