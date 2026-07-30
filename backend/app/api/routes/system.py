from __future__ import annotations

import platform
import sys
from importlib import metadata as importlib_metadata
from importlib import import_module

import psutil
from fastapi import APIRouter

router = APIRouter()

_DEPENDENCIES = ["torch", "transformers", "peft", "trl", "accelerate", "bitsandbytes", "datasets"]


def _check_dependency(name: str) -> dict:
    try:
        import_module(name)
    except Exception:
        return {"installed": False, "version": None}

    try:
        version = importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        version = None
    return {"installed": True, "version": version}


@router.get("/info")
async def info() -> dict:
    """Environment diagnostics: python/torch/CUDA info, platform, and a
    dependency checklist for the core fine-tuning stack."""

    torch_info: dict = {"installed": False, "version": None, "cuda_available": False}
    try:
        import torch

        torch_info["installed"] = True
        torch_info["version"] = torch.__version__
        cuda_available = torch.cuda.is_available()
        torch_info["cuda_available"] = cuda_available
        torch_info["cuda_version"] = torch.version.cuda
        if cuda_available:
            try:
                torch_info["device_name"] = torch.cuda.get_device_name(0)
                torch_info["device_count"] = torch.cuda.device_count()
            except Exception:
                torch_info["device_name"] = None
        else:
            torch_info["device_name"] = None
    except Exception:
        pass

    dependencies = {name: _check_dependency(name) for name in _DEPENDENCIES}

    vm = psutil.virtual_memory()

    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "cpu_count": psutil.cpu_count(logical=True),
        "ram_total_gb": vm.total / (1024**3),
        "torch": torch_info,
        "dependencies": dependencies,
    }
