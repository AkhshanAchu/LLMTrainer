from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.events import event_bus
from app.schemas.common import SystemStats
from app.services.gpu_service import get_system_stats

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/stats", response_model=SystemStats)
async def stats() -> SystemStats:
    """One-shot snapshot of current CPU/RAM/disk/GPU stats."""
    return await asyncio.to_thread(get_system_stats)


@router.websocket("/ws")
async def gpu_stats_ws(websocket: WebSocket) -> None:
    """Stream live system/GPU stats as they're published to the event bus."""
    await websocket.accept()
    queue = event_bus.subscribe("gpu_stats")
    try:
        while True:
            payload = await queue.get()
            await websocket.send_json(payload)
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("gpu_stats_ws error")
    finally:
        event_bus.unsubscribe("gpu_stats", queue)


@router.get("/nvidia-smi")
async def nvidia_smi() -> dict:
    """Best-effort raw nvidia-smi output, for debugging/fallback."""
    query = (
        "index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,"
        "power.draw,power.limit,fan.speed"
    )
    try:
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
    except FileNotFoundError:
        return {"available": False, "error": "nvidia-smi not found on PATH"}
    except Exception as exc:
        return {"available": False, "error": str(exc)}

    if proc.returncode != 0:
        return {
            "available": False,
            "error": stderr.decode("utf-8", errors="replace").strip()
            or f"nvidia-smi exited with code {proc.returncode}",
        }

    return {"available": True, "output": stdout.decode("utf-8", errors="replace")}
