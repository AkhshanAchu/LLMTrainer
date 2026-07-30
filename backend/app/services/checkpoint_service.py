"""Checkpoint listing, LoRA merge, and export utilities.

Merge/export are heavyweight (load full model + adapter into memory) so they
run via `run_in_threadpool` from the route layer, not inline in async code.
"""

import shutil
from pathlib import Path

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.config import get_settings
from app.models.db import Checkpoint, TrainingRun

settings = get_settings()


async def list_checkpoints(session: AsyncSession, run_id: int) -> list[Checkpoint]:
    result = await session.exec(select(Checkpoint).where(Checkpoint.run_id == run_id).order_by(Checkpoint.step))
    return list(result.all())


async def mark_best_checkpoint(session: AsyncSession, run_id: int) -> Checkpoint | None:
    checkpoints = await list_checkpoints(session, run_id)
    with_eval = [c for c in checkpoints if c.eval_loss is not None]
    if not with_eval:
        return None
    best = min(with_eval, key=lambda c: c.eval_loss)
    for c in checkpoints:
        c.is_best = c.id == best.id
        session.add(c)
    await session.commit()
    return best


def merge_lora_adapter(base_model_repo_id: str, adapter_path: str, output_dir: str) -> str:
    """Merge a LoRA/DoRA adapter into the base model and save full-precision weights."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_repo_id, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    merged = model.merge_and_unload()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out)

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    tokenizer.save_pretrained(out)
    return str(out)


def export_to_gguf(merged_model_dir: str, output_path: str, quant_type: str = "q4_k_m") -> dict:
    """Convert a merged HF model directory to GGUF via llama.cpp's convert script, if available.

    Requires a local llama.cpp checkout with `convert_hf_to_gguf.py` and the
    `llama-quantize` binary on PATH. This is best-effort: if llama.cpp tooling
    isn't installed, return a clear error dict rather than raising, so the API
    can surface actionable guidance instead of a 500.
    """
    import subprocess

    convert_script = shutil.which("convert_hf_to_gguf.py") or shutil.which("convert-hf-to-gguf.py")
    if convert_script is None:
        return {
            "success": False,
            "error": (
                "llama.cpp's convert_hf_to_gguf.py was not found on PATH. Install llama.cpp "
                "(https://github.com/ggerganov/llama.cpp) and ensure its conversion script is "
                "available to enable GGUF export."
            ),
        }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fp16_path = out.with_suffix(".fp16.gguf")

    try:
        subprocess.run(
            ["python", convert_script, merged_model_dir, "--outfile", str(fp16_path), "--outtype", "f16"],
            check=True,
            capture_output=True,
            text=True,
        )
        quantize_bin = shutil.which("llama-quantize") or shutil.which("quantize")
        if quantize_bin is None:
            return {"success": True, "path": str(fp16_path), "note": "fp16 GGUF only; llama-quantize not found for further compression."}

        subprocess.run([quantize_bin, str(fp16_path), str(out), quant_type], check=True, capture_output=True, text=True)
        fp16_path.unlink(missing_ok=True)
        return {"success": True, "path": str(out), "quant_type": quant_type}
    except subprocess.CalledProcessError as e:
        return {"success": False, "error": f"conversion failed: {e.stderr or e.stdout}"}


def export_merged_safetensors(base_model_repo_id: str, adapter_path: str, output_dir: str) -> dict:
    try:
        path = merge_lora_adapter(base_model_repo_id, adapter_path, output_dir)
        return {"success": True, "path": path}
    except Exception as e:  # noqa: BLE001
        return {"success": False, "error": str(e)}


def cleanup_old_checkpoints(run_dir: str, keep_last_n: int = 3) -> list[str]:
    """Delete step-numbered checkpoint dirs beyond the most recent N (keeps 'final')."""
    base = Path(run_dir)
    if not base.exists():
        return []
    ckpt_dirs = sorted(
        (p for p in base.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    to_delete = ckpt_dirs[:-keep_last_n] if keep_last_n > 0 else ckpt_dirs
    removed = []
    for d in to_delete:
        shutil.rmtree(d, ignore_errors=True)
        removed.append(str(d))
    return removed
