from pydantic import BaseModel, Field

from app.models.db import JobStatus, TrainingMethod, TrainingObjective


class TrainingConfig(BaseModel):
    # core
    base_model_repo_id: str
    dataset_id: int
    method: TrainingMethod = TrainingMethod.qlora
    objective: TrainingObjective = TrainingObjective.sft
    name: str = "untitled-run"

    # LoRA / PEFT
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] | None = None  # None -> auto-detect (all linear)
    use_dora: bool = False

    # quantization
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bf16"

    # optimization
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    optimizer: str = "paged_adamw_8bit"
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0

    # batch / schedule
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_train_epochs: float = 3.0
    max_steps: int = -1  # -1 = derive from epochs
    max_seq_length: int = 2048
    packing: bool = False

    # perf / memory
    gradient_checkpointing: bool = True
    use_flash_attention_2: bool = False
    neftune_noise_alpha: float | None = None
    seed: int = 42

    # eval / checkpointing
    eval_split_ratio: float = 0.05
    eval_steps: int = 50
    save_steps: int = 50
    save_total_limit: int = 3
    logging_steps: int = 5
    early_stopping_patience: int | None = None

    # DPO/ORPO/KTO specific
    beta: float = 0.1


class TrainingRunCreate(BaseModel):
    config: TrainingConfig


class TrainingRunOut(BaseModel):
    id: int
    name: str
    base_model_repo_id: str
    dataset_id: int
    method: TrainingMethod
    objective: TrainingObjective
    status: JobStatus
    config: dict
    output_dir: str
    current_step: int
    total_steps: int
    current_epoch: float
    latest_metrics: dict
    error_message: str | None


class MetricPointOut(BaseModel):
    step: int
    epoch: float
    loss: float | None = None
    eval_loss: float | None = None
    learning_rate: float | None = None
    grad_norm: float | None = None
    tokens_per_sec: float | None = None
    samples_per_sec: float | None = None
    gpu_mem_used_gb: float | None = None


class HyperparamSuggestion(BaseModel):
    method: TrainingMethod
    lora_r: int
    lora_alpha: int
    learning_rate: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    load_in_4bit: bool
    gradient_checkpointing: bool
    rationale: str = Field(description="Short human-readable explanation of the suggestion")
