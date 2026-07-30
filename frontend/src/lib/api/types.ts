export type DatasetFormat =
  | "alpaca"
  | "sharegpt"
  | "chatml"
  | "openai"
  | "llama"
  | "preference"
  | "raw_text"
  | "unknown";

export type TrainingMethod = "full" | "lora" | "qlora" | "dora" | "ia3" | "prompt_tuning";
export type TrainingObjective = "sft" | "dpo" | "orpo" | "kto" | "reward";
export type JobStatus = "queued" | "preparing" | "running" | "paused" | "completed" | "failed" | "cancelled";

export interface CuratedModel {
  repo_id: string;
  display_name: string;
  family: string;
  param_count: number;
  is_vision: boolean;
}

export interface ModelEntry {
  id: number;
  repo_id: string;
  display_name: string;
  architecture: string | null;
  param_count: number | null;
  context_length: number | null;
  license: string | null;
  is_vision: boolean;
  local_path: string | null;
  downloaded: boolean;
  bookmarked: boolean;
  last_used_at: string | null;
}

export interface VRAMEstimate {
  method: string;
  param_count: number;
  dtype: string;
  model_weights_gb: number;
  gradients_gb: number;
  optimizer_states_gb: number;
  activations_gb: number;
  total_estimated_gb: number;
  fits_in_gb: number | null;
  fits: boolean | null;
}

export interface DatasetSummary {
  id: number;
  name: string;
  detected_format: DatasetFormat;
  num_examples: number;
  created_at: string;
  quality_score: number | null;
}

export interface Dataset {
  id: number;
  name: string;
  source_filename: string;
  detected_format: DatasetFormat;
  num_examples: number;
  stats: Record<string, unknown>;
  validation_report: Record<string, unknown>;
  created_at: string;
}

export interface TrainingConfig {
  base_model_repo_id: string;
  dataset_id: number;
  method: TrainingMethod;
  objective: TrainingObjective;
  name: string;
  lora_r: number;
  lora_alpha: number;
  lora_dropout: number;
  lora_target_modules: string[] | null;
  use_dora: boolean;
  load_in_4bit: boolean;
  load_in_8bit: boolean;
  bnb_4bit_quant_type: string;
  bnb_4bit_compute_dtype: string;
  learning_rate: number;
  lr_scheduler_type: string;
  warmup_ratio: number;
  optimizer: string;
  weight_decay: number;
  max_grad_norm: number;
  per_device_train_batch_size: number;
  gradient_accumulation_steps: number;
  num_train_epochs: number;
  max_steps: number;
  max_seq_length: number;
  packing: boolean;
  gradient_checkpointing: boolean;
  use_flash_attention_2: boolean;
  neftune_noise_alpha: number | null;
  seed: number;
  eval_split_ratio: number;
  eval_steps: number;
  save_steps: number;
  save_total_limit: number;
  logging_steps: number;
  early_stopping_patience: number | null;
  beta: number;
}

export interface TrainingRun {
  id: number;
  name: string;
  base_model_repo_id: string;
  dataset_id: number;
  method: TrainingMethod;
  objective: TrainingObjective;
  status: JobStatus;
  config: Partial<TrainingConfig>;
  output_dir: string;
  current_step: number;
  total_steps: number;
  current_epoch: number;
  latest_metrics: Record<string, number | null>;
  error_message: string | null;
}

export interface MetricPoint {
  step: number;
  epoch: number;
  loss: number | null;
  eval_loss: number | null;
  learning_rate: number | null;
  grad_norm: number | null;
  tokens_per_sec: number | null;
  samples_per_sec: number | null;
  gpu_mem_used_gb: number | null;
}

export interface HyperparamSuggestion {
  method: TrainingMethod;
  lora_r: number;
  lora_alpha: number;
  learning_rate: number;
  per_device_train_batch_size: number;
  gradient_accumulation_steps: number;
  load_in_4bit: boolean;
  gradient_checkpointing: boolean;
  rationale: string;
}

export interface GPUStats {
  index: number;
  name: string;
  temperature_c: number | null;
  utilization_pct: number | null;
  memory_used_mb: number | null;
  memory_total_mb: number | null;
  power_draw_w: number | null;
  power_limit_w: number | null;
  fan_speed_pct: number | null;
}

export interface SystemStats {
  cpu_pct: number;
  ram_used_gb: number;
  ram_total_gb: number;
  disk_used_gb: number;
  disk_total_gb: number;
  gpus: GPUStats[];
}

export interface Checkpoint {
  id: number;
  step: number;
  epoch: number;
  path: string;
  size_bytes: number;
  eval_loss: number | null;
  is_best: boolean;
}
