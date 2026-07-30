"use client";

import { useEffect, useState } from "react";
import { toast } from "sonner";
import { Check, X } from "lucide-react";
import { modelsApi } from "@/lib/api/endpoints";
import type { VRAMEstimate } from "@/lib/api/types";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";

const METHODS = [
  { value: "full", label: "Full fine-tune" },
  { value: "lora", label: "LoRA" },
  { value: "qlora", label: "QLoRA" },
];

const DTYPES = [
  { value: "fp16", label: "FP16" },
  { value: "bf16", label: "BF16" },
  { value: "int8", label: "INT8" },
  { value: "nf4", label: "NF4" },
];

const REFERENCE_TIERS_GB = [8, 16, 24, 48];

const BREAKDOWN_COLORS: Record<string, string> = {
  model_weights_gb: "bg-blue-500",
  gradients_gb: "bg-amber-500",
  optimizer_states_gb: "bg-violet-500",
  activations_gb: "bg-emerald-500",
};

const BREAKDOWN_LABELS: Record<string, string> = {
  model_weights_gb: "Model weights",
  gradients_gb: "Gradients",
  optimizer_states_gb: "Optimizer states",
  activations_gb: "Activations",
};

export function VramEstimator({ repoId }: { repoId: string }) {
  const [method, setMethod] = useState("lora");
  const [dtype, setDtype] = useState("bf16");
  const [estimate, setEstimate] = useState<VRAMEstimate | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    modelsApi
      .vramEstimate({ repo_id: repoId, method, dtype })
      .then((res) => {
        if (!cancelled) setEstimate(res);
      })
      .catch((err) => {
        if (!cancelled) {
          toast.error(`Failed to estimate VRAM: ${err instanceof Error ? err.message : "unknown error"}`);
          setEstimate(null);
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [repoId, method, dtype]);

  const breakdownKeys = ["model_weights_gb", "gradients_gb", "optimizer_states_gb", "activations_gb"] as const;
  const total = estimate?.total_estimated_gb ?? 0;

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3">
        <div className="space-y-1">
          <Label className="text-xs text-muted-foreground">Method</Label>
          <Select value={method} onValueChange={(v) => setMethod(v as string)}>
            <SelectTrigger className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {METHODS.map((m) => (
                <SelectItem key={m.value} value={m.value}>
                  {m.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="space-y-1">
          <Label className="text-xs text-muted-foreground">Dtype</Label>
          <Select value={dtype} onValueChange={(v) => setDtype(v as string)}>
            <SelectTrigger className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {DTYPES.map((d) => (
                <SelectItem key={d.value} value={d.value}>
                  {d.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      {loading && (
        <div className="space-y-2">
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-2/3" />
        </div>
      )}

      {!loading && estimate && (
        <>
          {/* stacked bar */}
          <div>
            <div className="flex h-3 w-full overflow-hidden rounded-full bg-muted">
              {breakdownKeys.map((k) => {
                const v = estimate[k] as number;
                const pct = total > 0 ? (v / total) * 100 : 0;
                if (pct <= 0) return null;
                return <div key={k} className={BREAKDOWN_COLORS[k]} style={{ width: `${pct}%` }} title={`${BREAKDOWN_LABELS[k]}: ${v} GB`} />;
              })}
            </div>
            <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
              {breakdownKeys.map((k) => (
                <div key={k} className="flex items-center gap-1.5">
                  <span className={`size-2 rounded-full ${BREAKDOWN_COLORS[k]}`} />
                  {BREAKDOWN_LABELS[k]}: <span className="font-mono text-foreground">{(estimate[k] as number).toFixed(2)} GB</span>
                </div>
              ))}
            </div>
          </div>

          <div className="flex items-center justify-between rounded-lg border border-border px-3 py-2">
            <span className="text-sm text-muted-foreground">Total estimated</span>
            <span className="font-mono text-lg font-semibold">{total.toFixed(2)} GB</span>
          </div>

          {/* reference tiers */}
          <div className="space-y-1.5">
            <p className="text-xs text-muted-foreground">Fits within reference VRAM budgets</p>
            <div className="grid grid-cols-4 gap-2">
              {REFERENCE_TIERS_GB.map((tier) => {
                const fits = total <= tier;
                return (
                  <div
                    key={tier}
                    className={`flex flex-col items-center gap-1 rounded-lg border px-2 py-2 text-xs ${
                      fits ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-400" : "border-red-500/30 bg-red-500/10 text-red-400"
                    }`}
                  >
                    {fits ? <Check className="size-3.5" /> : <X className="size-3.5" />}
                    <span className="font-mono">{tier}GB</span>
                  </div>
                );
              })}
            </div>
          </div>
        </>
      )}

      {!loading && !estimate && <p className="text-sm text-muted-foreground">Unable to compute VRAM estimate for this model.</p>}
    </div>
  );
}
