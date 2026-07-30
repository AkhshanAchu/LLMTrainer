"""Heuristic hyperparameter + VRAM-fit suggestions.

Not a search procedure — a set of well-established rules of thumb used across
the QLoRA/LoRA fine-tuning community, tuned for consumer single-GPU setups.
"""

from app.models.db import TrainingMethod
from app.schemas.training import HyperparamSuggestion


def suggest_hyperparams(
    param_count: int,
    available_vram_gb: float,
    method: TrainingMethod | None = None,
) -> HyperparamSuggestion:
    billions = param_count / 1e9

    if method is None:
        # Auto-pick a method that should fit in available VRAM.
        if available_vram_gb >= 4 * billions:
            method = TrainingMethod.full
        elif available_vram_gb >= 0.75 * billions:
            method = TrainingMethod.lora
        else:
            method = TrainingMethod.qlora

    load_in_4bit = method == TrainingMethod.qlora
    gradient_checkpointing = billions >= 3 or available_vram_gb <= 12

    if billions <= 1.5:
        lora_r, lora_alpha, lr = 8, 16, 2e-4
    elif billions <= 4:
        lora_r, lora_alpha, lr = 16, 32, 2e-4
    elif billions <= 9:
        lora_r, lora_alpha, lr = 32, 64, 1e-4
    else:
        lora_r, lora_alpha, lr = 64, 128, 5e-5

    if available_vram_gb <= 8:
        batch_size, grad_accum = 1, 16
    elif available_vram_gb <= 16:
        batch_size, grad_accum = 1, 8
    elif available_vram_gb <= 24:
        batch_size, grad_accum = 2, 4
    else:
        batch_size, grad_accum = 4, 2

    rationale = (
        f"~{billions:.1f}B params, {available_vram_gb:.1f} GB VRAM available -> {method.value}. "
        f"LoRA r={lora_r} scales with model size; batch/grad-accum chosen to keep peak activation "
        f"memory within budget; gradient checkpointing {'enabled' if gradient_checkpointing else 'disabled'} "
        f"to trade compute for memory."
    )

    return HyperparamSuggestion(
        method=method,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        load_in_4bit=load_in_4bit,
        gradient_checkpointing=gradient_checkpointing,
        rationale=rationale,
    )
