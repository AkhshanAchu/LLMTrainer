"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { toast } from "sonner";
import { Topbar } from "@/components/layout/topbar";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { useAppStore } from "@/store/app-store";
import { datasetsApi, modelsApi, trainingApi } from "@/lib/api/endpoints";
import type { DatasetSummary, TrainingConfig, TrainingMethod, TrainingObjective, HyperparamSuggestion } from "@/lib/api/types";
import { Sparkles } from "lucide-react";

const DEFAULT_CONFIG: Omit<TrainingConfig, "base_model_repo_id" | "dataset_id"> = {
  method: "qlora",
  objective: "sft",
  name: "untitled-run",
  lora_r: 16,
  lora_alpha: 32,
  lora_dropout: 0.05,
  lora_target_modules: null,
  use_dora: false,
  load_in_4bit: true,
  load_in_8bit: false,
  bnb_4bit_quant_type: "nf4",
  bnb_4bit_compute_dtype: "bf16",
  learning_rate: 2e-4,
  lr_scheduler_type: "cosine",
  warmup_ratio: 0.03,
  optimizer: "paged_adamw_8bit",
  weight_decay: 0.0,
  max_grad_norm: 1.0,
  per_device_train_batch_size: 1,
  gradient_accumulation_steps: 8,
  num_train_epochs: 3,
  max_steps: -1,
  max_seq_length: 2048,
  packing: false,
  gradient_checkpointing: true,
  use_flash_attention_2: false,
  neftune_noise_alpha: null,
  seed: 42,
  eval_split_ratio: 0.05,
  eval_steps: 50,
  save_steps: 50,
  save_total_limit: 3,
  logging_steps: 5,
  early_stopping_patience: null,
  beta: 0.1,
};

export default function NewTrainingRunPage() {
  const router = useRouter();
  const { selectedModelRepoId, selectedDatasetId } = useAppStore();
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [cfg, setCfg] = useState<Partial<TrainingConfig>>({
    ...DEFAULT_CONFIG,
    base_model_repo_id: selectedModelRepoId ?? "",
    dataset_id: selectedDatasetId ?? undefined,
  });
  const [suggestion, setSuggestion] = useState<HyperparamSuggestion | null>(null);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    datasetsApi.list().then(setDatasets).catch(() => setDatasets([]));
  }, []);

  const update = <K extends keyof TrainingConfig>(key: K, value: TrainingConfig[K]) =>
    setCfg((c) => ({ ...c, [key]: value }));

  const suggestHparams = async () => {
    if (!cfg.base_model_repo_id) {
      toast.error("Enter a base model repo id first");
      return;
    }
    try {
      const s = await trainingApi.suggestHparams(cfg.base_model_repo_id, 8);
      setSuggestion(s);
      setCfg((c) => ({
        ...c,
        method: s.method,
        lora_r: s.lora_r,
        lora_alpha: s.lora_alpha,
        learning_rate: s.learning_rate,
        per_device_train_batch_size: s.per_device_train_batch_size,
        gradient_accumulation_steps: s.gradient_accumulation_steps,
        load_in_4bit: s.load_in_4bit,
        gradient_checkpointing: s.gradient_checkpointing,
      }));
      toast.success("Hyperparameters suggested");
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Failed to suggest hyperparameters");
    }
  };

  const submit = async () => {
    if (!cfg.base_model_repo_id || !cfg.dataset_id) {
      toast.error("Base model and dataset are required");
      return;
    }
    setSubmitting(true);
    try {
      const run = await trainingApi.create(cfg);
      toast.success(`Training run "${run.name}" started`);
      router.push(`/training/${run.id}`);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Failed to start training run");
    } finally {
      setSubmitting(false);
    }
  };

  const isPreference = cfg.objective !== "sft";

  return (
    <>
      <Topbar title="New Training Run" />
      <main className="flex-1 overflow-y-auto p-6 space-y-6 max-w-4xl">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Setup</CardTitle>
          </CardHeader>
          <CardContent className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Field label="Run name">
              <Input value={cfg.name ?? ""} onChange={(e) => update("name", e.target.value)} />
            </Field>
            <Field label="Base model (HF repo id)">
              <Input
                value={cfg.base_model_repo_id ?? ""}
                onChange={(e) => update("base_model_repo_id", e.target.value)}
                placeholder="e.g. meta-llama/Meta-Llama-3.1-8B-Instruct"
              />
            </Field>
            <Field label="Dataset">
              <Select
                value={cfg.dataset_id ? String(cfg.dataset_id) : undefined}
                onValueChange={(v) => update("dataset_id", Number(v))}
              >
                <SelectTrigger className="w-full">
                  <SelectValue placeholder="Select a dataset" />
                </SelectTrigger>
                <SelectContent>
                  {datasets.map((d) => (
                    <SelectItem key={d.id} value={String(d.id)}>
                      {d.name} ({d.num_examples} examples)
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </Field>
            <Field label="Objective">
              <Select value={cfg.objective} onValueChange={(v) => update("objective", v as TrainingObjective)}>
                <SelectTrigger className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="sft">SFT (Supervised Fine-Tuning)</SelectItem>
                  <SelectItem value="dpo">DPO (Direct Preference Optimization)</SelectItem>
                  <SelectItem value="kto">KTO</SelectItem>
                </SelectContent>
              </Select>
            </Field>
          </CardContent>
        </Card>

        {isPreference && (
          <Alert>
            <AlertTitle>Preference dataset required</AlertTitle>
            <AlertDescription>
              {cfg.objective?.toUpperCase()} expects a dataset with prompt/chosen/rejected fields (detected format
              &quot;preference&quot;). Make sure the selected dataset was validated as such.
            </AlertDescription>
          </Alert>
        )}

        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle className="text-base">Method &amp; Quantization</CardTitle>
            <Button variant="secondary" size="sm" onClick={suggestHparams}>
              <Sparkles className="size-4" /> Suggest hyperparameters
            </Button>
          </CardHeader>
          <CardContent className="space-y-4">
            {suggestion && (
              <Alert>
                <AlertTitle>Suggestion applied</AlertTitle>
                <AlertDescription className="text-xs">{suggestion.rationale}</AlertDescription>
              </Alert>
            )}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <Field label="Method">
                <Select value={cfg.method} onValueChange={(v) => update("method", v as TrainingMethod)}>
                  <SelectTrigger className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="qlora">QLoRA</SelectItem>
                    <SelectItem value="lora">LoRA</SelectItem>
                    <SelectItem value="dora">DoRA</SelectItem>
                    <SelectItem value="full">Full fine-tune</SelectItem>
                  </SelectContent>
                </Select>
              </Field>
              <Field label="LoRA r">
                <Input
                  type="number"
                  value={cfg.lora_r ?? 16}
                  onChange={(e) => update("lora_r", Number(e.target.value))}
                />
              </Field>
              <Field label="LoRA alpha">
                <Input
                  type="number"
                  value={cfg.lora_alpha ?? 32}
                  onChange={(e) => update("lora_alpha", Number(e.target.value))}
                />
              </Field>
            </div>
            <div className="flex items-center gap-3">
              <Switch checked={cfg.load_in_4bit} onCheckedChange={(v) => update("load_in_4bit", v)} />
              <Label>Load in 4-bit (QLoRA / NF4)</Label>
            </div>
            <div className="flex items-center gap-3">
              <Switch checked={cfg.gradient_checkpointing} onCheckedChange={(v) => update("gradient_checkpointing", v)} />
              <Label>Gradient checkpointing</Label>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Optimization</CardTitle>
          </CardHeader>
          <CardContent className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Field label={`Learning rate: ${cfg.learning_rate?.toExponential(1)}`}>
              <Slider
                value={[Math.log10(cfg.learning_rate ?? 2e-4)]}
                min={-6}
                max={-2}
                step={0.05}
                onValueChange={(v) => update("learning_rate", 10 ** (Array.isArray(v) ? v[0] : v))}
              />
            </Field>
            <Field label="Epochs">
              <Input
                type="number"
                step="0.5"
                value={cfg.num_train_epochs ?? 3}
                onChange={(e) => update("num_train_epochs", Number(e.target.value))}
              />
            </Field>
            <Field label="Max seq length">
              <Input
                type="number"
                value={cfg.max_seq_length ?? 2048}
                onChange={(e) => update("max_seq_length", Number(e.target.value))}
              />
            </Field>
            <Field label="Per-device batch size">
              <Input
                type="number"
                value={cfg.per_device_train_batch_size ?? 1}
                onChange={(e) => update("per_device_train_batch_size", Number(e.target.value))}
              />
            </Field>
            <Field label="Gradient accumulation steps">
              <Input
                type="number"
                value={cfg.gradient_accumulation_steps ?? 8}
                onChange={(e) => update("gradient_accumulation_steps", Number(e.target.value))}
              />
            </Field>
            <Field label="Scheduler">
              <Select value={cfg.lr_scheduler_type} onValueChange={(v) => v && update("lr_scheduler_type", v)}>
                <SelectTrigger className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="cosine">cosine</SelectItem>
                  <SelectItem value="linear">linear</SelectItem>
                  <SelectItem value="constant">constant</SelectItem>
                </SelectContent>
              </Select>
            </Field>
            {isPreference && (
              <Field label={`Beta (KL penalty): ${cfg.beta}`}>
                <Slider
                  value={[cfg.beta ?? 0.1]}
                  min={0.01}
                  max={1}
                  step={0.01}
                  onValueChange={(v) => update("beta", Array.isArray(v) ? v[0] : v)}
                />
              </Field>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Checkpointing</CardTitle>
          </CardHeader>
          <CardContent className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Field label="Save every N steps">
              <Input type="number" value={cfg.save_steps ?? 50} onChange={(e) => update("save_steps", Number(e.target.value))} />
            </Field>
            <Field label="Eval every N steps">
              <Input type="number" value={cfg.eval_steps ?? 50} onChange={(e) => update("eval_steps", Number(e.target.value))} />
            </Field>
            <Field label="Keep last N checkpoints">
              <Input
                type="number"
                value={cfg.save_total_limit ?? 3}
                onChange={(e) => update("save_total_limit", Number(e.target.value))}
              />
            </Field>
          </CardContent>
        </Card>

        <div className="flex justify-end gap-2 pb-6">
          <Button size="lg" onClick={submit} disabled={submitting}>
            {submitting ? "Starting…" : "Start Training Run"}
          </Button>
        </div>
      </main>
    </>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="space-y-1.5">
      <Label className="text-xs text-muted-foreground">{label}</Label>
      {children}
    </div>
  );
}
