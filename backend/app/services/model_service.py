"""Model Manager service: search/browse the HF Hub, inspect configs, estimate
VRAM requirements, download weights, and track local models in the DB.

VRAM estimate assumptions (see estimate_vram docstring for details):
  - bytes-per-param depends on dtype (fp32=4, fp16/bf16=2, int8=1, nf4=0.5)
  - full fine-tuning: all params are trainable -> gradients same size as
    weights (in the training dtype) and AdamW optimizer states cost 2x the
    weights in fp32 (8 bytes/param), which is the standard rule of thumb.
  - LoRA/QLoRA: only a small adapter fraction of params is trainable
    (~1-2% for LoRA, ~0.5-1% for QLoRA with typical rank/alpha), so
    gradients + optimizer states scale down to that fraction of the base
    weight size instead of the full model.
  - activations are a flat heuristic for seq_len~2048, batch=1, scaled up
    for full fine-tuning where activation checkpointing is less commonly
    assumed and batch sizes tend to be larger.
This is inherently approximate - real usage varies with framework,
attention implementation, sequence length, batch size, and gradient
checkpointing settings.
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from typing import Any, Literal

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.utils import GatedRepoError
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.config import get_settings
from app.models.db import ModelEntry
from app.schemas.common import VRAMEstimate

logger = logging.getLogger(__name__)

MethodT = Literal["full", "lora", "qlora"]
DtypeT = Literal["fp32", "fp16", "bf16", "int8", "nf4"]

BYTES_PER_PARAM: dict[str, float] = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "int8": 1.0,
    "nf4": 0.5,
}

GB = 1024**3

# Fraction of total params considered "trainable" for adapter-based methods.
TRAINABLE_FRACTION = {
    "lora": 0.02,
    "qlora": 0.01,
}

# Flat activation-memory heuristics (GB) at seq_len~2048, batch=1.
ACTIVATIONS_BASE_GB = {
    "full": 4.0,
    "lora": 2.5,
    "qlora": 2.0,
}


# ---------------------------------------------------------------------------
# Curated seed list (offline / default browse fallback)
# ---------------------------------------------------------------------------

CURATED_MODELS: list[dict[str, Any]] = [
    {"repo_id": "meta-llama/Meta-Llama-3.1-8B-Instruct", "display_name": "Llama 3.1 8B Instruct", "family": "Llama", "param_count": 8_000_000_000, "is_vision": False},
    {"repo_id": "meta-llama/Meta-Llama-3.1-70B-Instruct", "display_name": "Llama 3.1 70B Instruct", "family": "Llama", "param_count": 70_000_000_000, "is_vision": False},
    {"repo_id": "meta-llama/Meta-Llama-3-8B-Instruct", "display_name": "Llama 3 8B Instruct", "family": "Llama", "param_count": 8_000_000_000, "is_vision": False},
    {"repo_id": "meta-llama/Llama-3.2-1B-Instruct", "display_name": "Llama 3.2 1B Instruct", "family": "Llama", "param_count": 1_000_000_000, "is_vision": False},
    {"repo_id": "meta-llama/Llama-3.2-3B-Instruct", "display_name": "Llama 3.2 3B Instruct", "family": "Llama", "param_count": 3_000_000_000, "is_vision": False},
    {"repo_id": "mistralai/Mistral-7B-Instruct-v0.3", "display_name": "Mistral 7B Instruct v0.3", "family": "Mistral", "param_count": 7_000_000_000, "is_vision": False},
    {"repo_id": "mistralai/Mixtral-8x7B-Instruct-v0.1", "display_name": "Mixtral 8x7B Instruct", "family": "Mistral", "param_count": 46_700_000_000, "is_vision": False},
    {"repo_id": "mistralai/Mistral-Nemo-Instruct-2407", "display_name": "Mistral Nemo 12B Instruct", "family": "Mistral", "param_count": 12_000_000_000, "is_vision": False},
    {"repo_id": "Qwen/Qwen2.5-7B-Instruct", "display_name": "Qwen2.5 7B Instruct", "family": "Qwen", "param_count": 7_000_000_000, "is_vision": False},
    {"repo_id": "Qwen/Qwen2.5-14B-Instruct", "display_name": "Qwen2.5 14B Instruct", "family": "Qwen", "param_count": 14_000_000_000, "is_vision": False},
    {"repo_id": "Qwen/Qwen2.5-0.5B-Instruct", "display_name": "Qwen2.5 0.5B Instruct", "family": "Qwen", "param_count": 500_000_000, "is_vision": False},
    {"repo_id": "Qwen/Qwen2-7B-Instruct", "display_name": "Qwen2 7B Instruct", "family": "Qwen", "param_count": 7_000_000_000, "is_vision": False},
    {"repo_id": "google/gemma-2-9b-it", "display_name": "Gemma 2 9B Instruct", "family": "Gemma", "param_count": 9_000_000_000, "is_vision": False},
    {"repo_id": "google/gemma-2-2b-it", "display_name": "Gemma 2 2B Instruct", "family": "Gemma", "param_count": 2_000_000_000, "is_vision": False},
    {"repo_id": "google/gemma-7b-it", "display_name": "Gemma 7B Instruct", "family": "Gemma", "param_count": 7_000_000_000, "is_vision": False},
    {"repo_id": "microsoft/Phi-3-mini-4k-instruct", "display_name": "Phi-3 Mini 4K Instruct", "family": "Phi", "param_count": 3_800_000_000, "is_vision": False},
    {"repo_id": "microsoft/Phi-3-medium-4k-instruct", "display_name": "Phi-3 Medium 4K Instruct", "family": "Phi", "param_count": 14_000_000_000, "is_vision": False},
    {"repo_id": "microsoft/Phi-4", "display_name": "Phi-4 14B", "family": "Phi", "param_count": 14_000_000_000, "is_vision": False},
    {"repo_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "display_name": "TinyLlama 1.1B Chat", "family": "TinyLlama", "param_count": 1_100_000_000, "is_vision": False},
    {"repo_id": "deepseek-ai/deepseek-llm-7b-chat", "display_name": "DeepSeek LLM 7B Chat", "family": "DeepSeek", "param_count": 7_000_000_000, "is_vision": False},
    {"repo_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "display_name": "DeepSeek R1 Distill Qwen 7B", "family": "DeepSeek", "param_count": 7_000_000_000, "is_vision": False},
    {"repo_id": "01-ai/Yi-1.5-9B-Chat", "display_name": "Yi 1.5 9B Chat", "family": "Yi", "param_count": 9_000_000_000, "is_vision": False},
    {"repo_id": "01-ai/Yi-1.5-6B-Chat", "display_name": "Yi 1.5 6B Chat", "family": "Yi", "param_count": 6_000_000_000, "is_vision": False},
    {"repo_id": "llava-hf/llava-1.5-7b-hf", "display_name": "LLaVA 1.5 7B", "family": "LLaVA", "param_count": 7_000_000_000, "is_vision": True},
    {"repo_id": "Qwen/Qwen2-VL-7B-Instruct", "display_name": "Qwen2-VL 7B Instruct", "family": "Qwen-VL", "param_count": 7_000_000_000, "is_vision": True},
]


# ---------------------------------------------------------------------------
# Hub search / inspection
# ---------------------------------------------------------------------------


def search_hub_models(query: str, limit: int = 20) -> list[dict[str, Any]]:
    """Search the HF Hub for text-generation models matching `query`."""
    settings = get_settings()
    api = HfApi(token=settings.hf_token)
    results = api.list_models(
        search=query,
        limit=limit,
        sort="downloads",
        pipeline_tag="text-generation",
    )
    out: list[dict[str, Any]] = []
    for m in results:
        out.append(
            {
                "repo_id": m.id,
                "downloads": getattr(m, "downloads", None),
                "likes": getattr(m, "likes", None),
                "tags": getattr(m, "tags", None) or [],
                "pipeline_tag": getattr(m, "pipeline_tag", None),
            }
        )
    return out


def get_model_info(repo_id: str) -> dict[str, Any]:
    """Fetch model metadata + config.json fields from the Hub.

    Best-effort: gated/missing/network errors are caught and returned as a
    partial dict with an "error" key rather than raised, so callers (and the
    API layer) can still show whatever info is available.
    """
    settings = get_settings()
    api = HfApi(token=settings.hf_token)
    info: dict[str, Any] = {"repo_id": repo_id}

    try:
        model_info = api.model_info(repo_id)
        info["license"] = (model_info.card_data or {}).get("license") if model_info.card_data else None
        info["tags"] = model_info.tags or []
        info["pipeline_tag"] = model_info.pipeline_tag
        info["gated"] = bool(model_info.gated) if model_info.gated is not None else False
        info["siblings"] = [s.rfilename for s in (model_info.siblings or [])]
    except GatedRepoError as e:
        info["error"] = f"gated repo, access required: {e}"
    except HfHubHTTPError as e:
        info["error"] = f"HF Hub HTTP error fetching model_info: {e}"
    except Exception as e:  # noqa: BLE001 - keep best-effort behavior
        info["error"] = f"failed to fetch model_info: {e}"

    config: dict[str, Any] | None = None
    try:
        config_path = hf_hub_download(repo_id, filename="config.json", token=settings.hf_token)
        import json

        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
    except Exception as e:  # noqa: BLE001
        info.setdefault("config_error", f"failed to fetch config.json: {e}")

    if config:
        info["architecture"] = (config.get("architectures") or [None])[0]
        info["model_type"] = config.get("model_type")
        info["hidden_size"] = config.get("hidden_size") or config.get("n_embd")
        info["num_hidden_layers"] = config.get("num_hidden_layers") or config.get("n_layer")
        info["num_attention_heads"] = config.get("num_attention_heads") or config.get("n_head")
        info["context_length"] = (
            config.get("max_position_embeddings")
            or config.get("max_sequence_length")
            or config.get("n_positions")
            or config.get("seq_length")
        )
        info["vocab_size"] = config.get("vocab_size")
        info["torch_dtype"] = config.get("torch_dtype")
        info["raw_config"] = config

        param_count = config.get("num_parameters")
        if not param_count:
            param_count = estimate_params_from_config(config)
        info["param_count"] = param_count
        info["param_count_is_estimate"] = not bool(config.get("num_parameters"))

    return info


def estimate_params_from_config(config: dict[str, Any]) -> int | None:
    """Rough parameter count estimate from architectural dims.

    Uses the standard decoder-only transformer approximation:
        params ~= 12 * L * H^2  (attention + MLP blocks, GPT-style MLP ratio)
                  + 2 * V * H   (input/output embeddings, tied or not - we
                                 count both since untied is the common case
                                 for models we care about)
    This is a coarse estimate (+/- 10-20%) meant only for VRAM planning when
    a repo doesn't publish an exact parameter count.
    """
    hidden_size = config.get("hidden_size") or config.get("n_embd")
    num_layers = config.get("num_hidden_layers") or config.get("n_layer")
    vocab_size = config.get("vocab_size")

    if not hidden_size or not num_layers:
        return None

    hidden_size = int(hidden_size)
    num_layers = int(num_layers)
    vocab_size = int(vocab_size) if vocab_size else 32000

    params = 12 * num_layers * hidden_size**2 + 2 * vocab_size * hidden_size
    return int(params)


# ---------------------------------------------------------------------------
# VRAM estimation
# ---------------------------------------------------------------------------


def estimate_vram(
    param_count: int,
    method: MethodT,
    dtype: DtypeT = "bf16",
) -> VRAMEstimate:
    """Estimate VRAM (GB) needed to fine-tune a model of `param_count` params.

    See module docstring for the assumptions behind these numbers. Not a
    precise prediction - use as a planning heuristic only.
    """
    bytes_per_param = BYTES_PER_PARAM[dtype]
    weights_gb = (param_count * bytes_per_param) / GB

    if method == "full":
        gradients_gb = weights_gb
        # AdamW: 2 states (momentum + variance) in fp32 = 8 bytes/param.
        optimizer_states_gb = (param_count * 8) / GB
        activations_gb = ACTIVATIONS_BASE_GB["full"] * max(1.0, weights_gb / 8.0)
    else:
        trainable_fraction = TRAINABLE_FRACTION[method]
        trainable_params = param_count * trainable_fraction
        gradients_gb = (trainable_params * bytes_per_param) / GB
        optimizer_states_gb = (trainable_params * 8) / GB
        activations_gb = ACTIVATIONS_BASE_GB[method] * max(1.0, weights_gb / 8.0)

    total = weights_gb + gradients_gb + optimizer_states_gb + activations_gb

    return VRAMEstimate(
        method=method,
        param_count=param_count,
        dtype=dtype,
        model_weights_gb=round(weights_gb, 2),
        gradients_gb=round(gradients_gb, 2),
        optimizer_states_gb=round(optimizer_states_gb, 2),
        activations_gb=round(activations_gb, 2),
        total_estimated_gb=round(total, 2),
    )


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def download_model_blocking(repo_id: str) -> str:
    """Blocking snapshot download. Call via run_in_threadpool/asyncio.to_thread."""
    settings = get_settings()
    local_path = snapshot_download(
        repo_id=repo_id,
        cache_dir=str(settings.models_cache_dir),
        token=settings.hf_token,
    )
    return local_path


async def download_model(repo_id: str) -> str:
    from starlette.concurrency import run_in_threadpool

    return await run_in_threadpool(download_model_blocking, repo_id)


# ---------------------------------------------------------------------------
# CRUD helpers (ModelEntry table)
# ---------------------------------------------------------------------------


async def list_models(
    session: AsyncSession,
    bookmarked_only: bool = False,
    is_vision: bool | None = None,
) -> list[ModelEntry]:
    stmt = select(ModelEntry)
    if bookmarked_only:
        stmt = stmt.where(ModelEntry.bookmarked == True)  # noqa: E712
    if is_vision is not None:
        stmt = stmt.where(ModelEntry.is_vision == is_vision)
    stmt = stmt.order_by(ModelEntry.created_at.desc())
    result = await session.exec(stmt)
    return list(result.all())


async def list_recent(session: AsyncSession, limit: int = 10) -> list[ModelEntry]:
    stmt = (
        select(ModelEntry)
        .where(ModelEntry.last_used_at.is_not(None))
        .order_by(ModelEntry.last_used_at.desc())
        .limit(limit)
    )
    result = await session.exec(stmt)
    return list(result.all())


async def get_model_by_id(session: AsyncSession, model_id: int) -> ModelEntry | None:
    return await session.get(ModelEntry, model_id)


async def get_model_by_repo_id(session: AsyncSession, repo_id: str) -> ModelEntry | None:
    stmt = select(ModelEntry).where(ModelEntry.repo_id == repo_id)
    result = await session.exec(stmt)
    return result.first()


async def toggle_bookmark(session: AsyncSession, model_id: int) -> ModelEntry | None:
    entry = await session.get(ModelEntry, model_id)
    if entry is None:
        return None
    entry.bookmarked = not entry.bookmarked
    session.add(entry)
    await session.commit()
    await session.refresh(entry)
    return entry


async def record_last_used(session: AsyncSession, model_id: int) -> ModelEntry | None:
    entry = await session.get(ModelEntry, model_id)
    if entry is None:
        return None
    entry.last_used_at = datetime.now(timezone.utc)
    session.add(entry)
    await session.commit()
    await session.refresh(entry)
    return entry


async def create_or_update_entry_after_download(
    session: AsyncSession,
    repo_id: str,
    local_path: str,
    display_name: str | None = None,
    info: dict[str, Any] | None = None,
) -> ModelEntry:
    entry = await get_model_by_repo_id(session, repo_id)
    info = info or {}

    is_vision = False
    tags = info.get("tags") or []
    if any("vision" in t or "vl" in t.lower() or "image-text" in t for t in tags):
        is_vision = True
    if "llava" in repo_id.lower() or "-vl" in repo_id.lower() or "vision" in repo_id.lower():
        is_vision = True

    if entry is None:
        entry = ModelEntry(
            repo_id=repo_id,
            display_name=display_name or repo_id,
            architecture=info.get("architecture"),
            param_count=info.get("param_count"),
            context_length=info.get("context_length"),
            license=info.get("license"),
            is_vision=is_vision,
            local_path=local_path,
            downloaded=True,
            metadata_json=info,
        )
    else:
        entry.local_path = local_path
        entry.downloaded = True
        entry.architecture = info.get("architecture") or entry.architecture
        entry.param_count = info.get("param_count") or entry.param_count
        entry.context_length = info.get("context_length") or entry.context_length
        entry.license = info.get("license") or entry.license
        entry.metadata_json = info or entry.metadata_json
        if display_name:
            entry.display_name = display_name

    session.add(entry)
    await session.commit()
    await session.refresh(entry)
    return entry


async def delete_model(session: AsyncSession, model_id: int) -> bool:
    entry = await session.get(ModelEntry, model_id)
    if entry is None:
        return False
    if entry.local_path:
        try:
            shutil.rmtree(entry.local_path, ignore_errors=True)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to remove local model files at %s", entry.local_path)
    await session.delete(entry)
    await session.commit()
    return True
