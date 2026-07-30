from pydantic import BaseModel


class GPUStats(BaseModel):
    index: int
    name: str
    temperature_c: float | None = None
    utilization_pct: float | None = None
    memory_used_mb: float | None = None
    memory_total_mb: float | None = None
    power_draw_w: float | None = None
    power_limit_w: float | None = None
    fan_speed_pct: float | None = None


class SystemStats(BaseModel):
    cpu_pct: float
    ram_used_gb: float
    ram_total_gb: float
    disk_used_gb: float
    disk_total_gb: float
    gpus: list[GPUStats] = []


class VRAMEstimate(BaseModel):
    method: str
    param_count: int
    dtype: str
    model_weights_gb: float
    gradients_gb: float
    optimizer_states_gb: float
    activations_gb: float
    total_estimated_gb: float
    fits_in_gb: float | None = None
    fits: bool | None = None
