"""GPU + system monitoring service.

Uses `pynvml` (nvidia-ml-py) for real NVIDIA GPU telemetry and `psutil` for
CPU/RAM/disk stats. Designed to degrade gracefully on machines without an
NVIDIA driver/GPU: NVML init failures are caught and all GPU queries then
return an empty list instead of raising.
"""

from __future__ import annotations

import asyncio
import logging
import os

import psutil

from app.core.events import event_bus
from app.schemas.common import GPUStats, SystemStats

logger = logging.getLogger(__name__)

NVML_AVAILABLE = False

try:
    import pynvml

    try:
        pynvml.nvmlInit()
        NVML_AVAILABLE = True
    except Exception as exc:  # pragma: no cover - depends on host hardware/driver
        logger.warning("NVML init failed, GPU stats will be unavailable: %s", exc)
        NVML_AVAILABLE = False
except ImportError:  # pragma: no cover - pynvml not installed
    pynvml = None  # type: ignore[assignment]
    logger.warning("pynvml not installed, GPU stats will be unavailable")
    NVML_AVAILABLE = False


def _safe(fn, *args):
    """Call an NVML getter, returning None instead of raising (e.g. fan speed
    on laptop GPUs that don't expose the sensor)."""
    try:
        return fn(*args)
    except Exception:
        return None


def get_gpu_stats() -> list[GPUStats]:
    """Return live per-GPU telemetry, or an empty list if NVML is unavailable."""
    if not NVML_AVAILABLE:
        return []

    stats: list[GPUStats] = []
    try:
        device_count = pynvml.nvmlDeviceGetCount()
    except Exception as exc:
        logger.warning("nvmlDeviceGetCount failed: %s", exc)
        return []

    for index in range(device_count):
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        except Exception as exc:
            logger.warning("Could not get handle for GPU %d: %s", index, exc)
            continue

        name = _safe(pynvml.nvmlDeviceGetName, handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        if name is None:
            name = f"GPU {index}"

        temperature_c = _safe(
            pynvml.nvmlDeviceGetTemperature, handle, pynvml.NVML_TEMPERATURE_GPU
        )

        util = _safe(pynvml.nvmlDeviceGetUtilizationRates, handle)
        utilization_pct = util.gpu if util is not None else None

        mem = _safe(pynvml.nvmlDeviceGetMemoryInfo, handle)
        memory_used_mb = mem.used / (1024 * 1024) if mem is not None else None
        memory_total_mb = mem.total / (1024 * 1024) if mem is not None else None

        power_draw_mw = _safe(pynvml.nvmlDeviceGetPowerUsage, handle)
        power_draw_w = power_draw_mw / 1000 if power_draw_mw is not None else None

        power_limit_mw = _safe(pynvml.nvmlDeviceGetEnforcedPowerLimit, handle)
        power_limit_w = power_limit_mw / 1000 if power_limit_mw is not None else None

        fan_speed_pct = _safe(pynvml.nvmlDeviceGetFanSpeed, handle)

        stats.append(
            GPUStats(
                index=index,
                name=name,
                temperature_c=temperature_c,
                utilization_pct=utilization_pct,
                memory_used_mb=memory_used_mb,
                memory_total_mb=memory_total_mb,
                power_draw_w=power_draw_w,
                power_limit_w=power_limit_w,
                fan_speed_pct=fan_speed_pct,
            )
        )

    return stats


def _disk_root() -> str:
    """Best-effort robust disk root for psutil.disk_usage() on Windows."""
    drive = os.path.splitdrive(os.getcwd())[0]
    if drive:
        return drive + "\\"
    return "C:\\"


def get_system_stats() -> SystemStats:
    """Return a snapshot of CPU/RAM/disk + all GPU stats."""
    cpu_pct = psutil.cpu_percent(interval=None)

    vm = psutil.virtual_memory()
    ram_used_gb = (vm.total - vm.available) / (1024**3)
    ram_total_gb = vm.total / (1024**3)

    try:
        disk = psutil.disk_usage(_disk_root())
        disk_used_gb = disk.used / (1024**3)
        disk_total_gb = disk.total / (1024**3)
    except Exception as exc:
        logger.warning("disk_usage failed: %s", exc)
        disk_used_gb = 0.0
        disk_total_gb = 0.0

    return SystemStats(
        cpu_pct=cpu_pct,
        ram_used_gb=ram_used_gb,
        ram_total_gb=ram_total_gb,
        disk_used_gb=disk_used_gb,
        disk_total_gb=disk_total_gb,
        gpus=get_gpu_stats(),
    )


async def gpu_monitor_loop(interval_s: float = 1.5) -> None:
    """Background task: poll system/GPU stats and publish to the event bus."""
    while True:
        try:
            stats = await asyncio.to_thread(get_system_stats)
            await event_bus.publish("gpu_stats", stats.model_dump())
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("gpu_monitor_loop iteration failed")
        await asyncio.sleep(interval_s)
