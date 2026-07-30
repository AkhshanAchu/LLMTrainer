import enum
from datetime import datetime, timezone

from sqlmodel import JSON, Column, Field, SQLModel


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DatasetFormat(str, enum.Enum):
    alpaca = "alpaca"
    sharegpt = "sharegpt"
    chatml = "chatml"
    openai = "openai"
    llama = "llama"
    preference = "preference"  # chosen/rejected pairs, for DPO/ORPO/KTO
    raw_text = "raw_text"
    unknown = "unknown"


class TrainingMethod(str, enum.Enum):
    full = "full"
    lora = "lora"
    qlora = "qlora"
    dora = "dora"
    ia3 = "ia3"
    prompt_tuning = "prompt_tuning"


class TrainingObjective(str, enum.Enum):
    sft = "sft"
    dpo = "dpo"
    orpo = "orpo"
    kto = "kto"
    reward = "reward"


class JobStatus(str, enum.Enum):
    queued = "queued"
    preparing = "preparing"
    running = "running"
    paused = "paused"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class Dataset(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str
    source_filename: str
    stored_path: str
    detected_format: DatasetFormat = DatasetFormat.unknown
    num_examples: int = 0
    stats: dict = Field(default_factory=dict, sa_column=Column(JSON))
    validation_report: dict = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class ModelEntry(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    repo_id: str = Field(index=True)
    display_name: str
    architecture: str | None = None
    param_count: int | None = None
    context_length: int | None = None
    license: str | None = None
    is_vision: bool = False
    local_path: str | None = None
    downloaded: bool = False
    bookmarked: bool = False
    last_used_at: datetime | None = None
    metadata_json: dict = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utcnow)


class TrainingRun(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str
    base_model_repo_id: str
    dataset_id: int = Field(foreign_key="dataset.id")
    method: TrainingMethod
    objective: TrainingObjective
    status: JobStatus = JobStatus.queued
    config: dict = Field(default_factory=dict, sa_column=Column(JSON))
    output_dir: str = ""
    current_step: int = 0
    total_steps: int = 0
    current_epoch: float = 0.0
    latest_metrics: dict = Field(default_factory=dict, sa_column=Column(JSON))
    error_message: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None


class MetricPoint(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    run_id: int = Field(foreign_key="trainingrun.id", index=True)
    step: int
    epoch: float
    wall_time: datetime = Field(default_factory=utcnow)
    loss: float | None = None
    eval_loss: float | None = None
    learning_rate: float | None = None
    grad_norm: float | None = None
    tokens_per_sec: float | None = None
    samples_per_sec: float | None = None
    gpu_mem_used_gb: float | None = None
    extra: dict = Field(default_factory=dict, sa_column=Column(JSON))


class Checkpoint(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    run_id: int = Field(foreign_key="trainingrun.id", index=True)
    step: int
    epoch: float
    path: str
    size_bytes: int = 0
    eval_loss: float | None = None
    is_best: bool = False
    created_at: datetime = Field(default_factory=utcnow)
