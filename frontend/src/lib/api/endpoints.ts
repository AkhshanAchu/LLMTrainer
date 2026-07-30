import { api } from "./client";
import type {
  Checkpoint,
  CuratedModel,
  Dataset,
  DatasetSummary,
  GPUStats,
  HyperparamSuggestion,
  MetricPoint,
  ModelEntry,
  SystemStats,
  TrainingConfig,
  TrainingRun,
  VRAMEstimate,
} from "./types";

export const modelsApi = {
  curated: () => api.get<CuratedModel[]>("/api/models/curated"),
  search: (q: string) => api.get<Record<string, unknown>[]>(`/api/models/search?q=${encodeURIComponent(q)}`),
  info: (repoId: string) => api.get<Record<string, unknown>>(`/api/models/info/${encodeURIComponent(repoId)}`),
  vramEstimate: (payload: { repo_id?: string; param_count?: number; method: string; dtype?: string }) =>
    api.post<VRAMEstimate>("/api/models/vram-estimate", payload),
  download: (repoId: string, displayName?: string) =>
    api.post<{ status: string }>("/api/models/download", { repo_id: repoId, display_name: displayName }),
  library: (params?: { bookmarked_only?: boolean; is_vision?: boolean }) => {
    const qs = new URLSearchParams();
    if (params?.bookmarked_only) qs.set("bookmarked_only", "true");
    if (params?.is_vision !== undefined) qs.set("is_vision", String(params.is_vision));
    const suffix = qs.toString() ? `?${qs.toString()}` : "";
    return api.get<ModelEntry[]>(`/api/models/library${suffix}`);
  },
  toggleBookmark: (modelId: number) => api.post<ModelEntry>(`/api/models/${modelId}/bookmark`),
  remove: (modelId: number) => api.delete<{ status: string }>(`/api/models/${modelId}`),
};

export const datasetsApi = {
  list: () => api.get<DatasetSummary[]>("/api/datasets/"),
  get: (id: number) => api.get<Dataset>(`/api/datasets/${id}`),
  upload: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return api.postForm<Dataset>("/api/datasets/upload", form);
  },
  preview: (id: number, n = 20, format: "raw" | "chat" = "chat") =>
    api.get<{ format: string; records: Record<string, unknown>[] }>(`/api/datasets/${id}/preview?n=${n}&format=${format}`),
  tokenizePreview: (id: number, payload: { tokenizer_repo_id?: string; max_seq_len?: number; n?: number }) =>
    api.post<Record<string, unknown>>(`/api/datasets/${id}/tokenize-preview`, payload),
  repair: (id: number, options: Record<string, unknown>, newName?: string) =>
    api.post<Dataset>(`/api/datasets/${id}/repair`, { options, new_name: newName }),
  remove: (id: number) => api.delete<{ status: string }>(`/api/datasets/${id}`),
};

export const trainingApi = {
  list: () => api.get<TrainingRun[]>("/api/training"),
  get: (id: number) => api.get<TrainingRun>(`/api/training/${id}`),
  create: (config: Partial<TrainingConfig>) => api.post<TrainingRun>("/api/training", { config }),
  metrics: (id: number) => api.get<MetricPoint[]>(`/api/training/${id}/metrics`),
  cancel: (id: number) => api.post<{ status: string }>(`/api/training/${id}/cancel`),
  suggestHparams: (repoId: string, availableVramGb = 8) =>
    api.get<HyperparamSuggestion>(
      `/api/training/hparams/suggest?repo_id=${encodeURIComponent(repoId)}&available_vram_gb=${availableVramGb}`
    ),
};

export const checkpointsApi = {
  list: (runId: number) => api.get<Checkpoint[]>(`/api/checkpoints/${runId}`),
  markBest: (runId: number) => api.post<{ best_checkpoint_id: number; step: number; eval_loss: number }>(`/api/checkpoints/${runId}/best`),
  export: (runId: number, payload: { checkpoint_path?: string; format: string; quant_type?: string }) =>
    api.post<Record<string, unknown>>(`/api/checkpoints/${runId}/export`, payload),
  cleanup: (runId: number, keepLastN = 3) =>
    api.post<{ removed: string[] }>(`/api/checkpoints/${runId}/cleanup?keep_last_n=${keepLastN}`),
};

export const inferenceApi = {
  load: (repoId: string, adapterPath?: string, loadIn4bit = false) =>
    api.post<{ status: string }>("/api/inference/load", { repo_id: repoId, adapter_path: adapterPath, load_in_4bit: loadIn4bit }),
  unload: () => api.post<{ status: string }>("/api/inference/unload"),
  current: () => api.get<{ repo_id: string | null }>("/api/inference/current"),
  benchmark: (prompt: string, maxNewTokens = 256) =>
    api.post<Record<string, number>>("/api/inference/benchmark", { prompt, max_new_tokens: maxNewTokens }),
};

export const gpuApi = {
  stats: () => api.get<SystemStats>("/api/gpu/stats"),
};

export const systemApi = {
  info: () => api.get<Record<string, unknown>>("/api/system/info"),
};

export const tokenizerApi = {
  encode: (repoId: string, text: string) =>
    api.post<{ token_count: number; tokens: { id: number; text: string; is_special: boolean }[]; special_tokens: Record<string, string | null>; vocab_size: number }>(
      "/api/tokenizer/encode",
      { repo_id: repoId, text }
    ),
  chatTemplate: (repoId: string, messages: { role: string; content: string }[], addGenerationPrompt = true) =>
    api.post<{ formatted_prompt: string; token_count: number; chat_template_raw: string }>("/api/tokenizer/chat-template", {
      repo_id: repoId,
      messages,
      add_generation_prompt: addGenerationPrompt,
    }),
};

export type { GPUStats };
