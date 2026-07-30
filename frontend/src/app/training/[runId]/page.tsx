"use client";

import { useEffect, useRef, useState } from "react";
import { useParams } from "next/navigation";
import { toast } from "sonner";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Topbar } from "@/components/layout/topbar";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { checkpointsApi, trainingApi } from "@/lib/api/endpoints";
import { wsUrl } from "@/lib/api/client";
import type { Checkpoint, MetricPoint, TrainingRun } from "@/lib/api/types";
import { AlertTriangle, Ban, Download } from "lucide-react";

const STATUS_STYLES: Record<string, string> = {
  running: "bg-emerald-500/15 text-emerald-400 border-emerald-500/30",
  preparing: "bg-amber-500/15 text-amber-400 border-amber-500/30",
  queued: "bg-slate-500/15 text-slate-400 border-slate-500/30",
  completed: "bg-blue-500/15 text-blue-400 border-blue-500/30",
  failed: "bg-red-500/15 text-red-400 border-red-500/30",
  cancelled: "bg-slate-500/15 text-slate-400 border-slate-500/30",
};

export default function TrainingRunDetailPage() {
  const params = useParams<{ runId: string }>();
  const runId = Number(params.runId);

  const [run, setRun] = useState<TrainingRun | null>(null);
  const [metrics, setMetrics] = useState<MetricPoint[]>([]);
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
  const [logLines, setLogLines] = useState<string[]>([]);
  const [gradAlert, setGradAlert] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!runId) return;
    trainingApi.get(runId).then(setRun).catch(() => {});
    trainingApi.metrics(runId).then(setMetrics).catch(() => {});
    checkpointsApi.list(runId).then(setCheckpoints).catch(() => {});
  }, [runId]);

  useEffect(() => {
    if (!runId) return;
    const ws = new WebSocket(wsUrl(`/api/training/${runId}/ws`));
    wsRef.current = ws;

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.event === "metric") {
          setMetrics((prev) => {
            const next = [...prev, data as MetricPoint];
            const prevNorm = prev[prev.length - 1]?.grad_norm;
            if (data.grad_norm != null) {
              if (data.grad_norm > 10 && (prevNorm == null || data.grad_norm > prevNorm * 3)) {
                setGradAlert(`Possible gradient explosion detected at step ${data.step} (grad_norm=${data.grad_norm.toFixed(2)})`);
              } else if (data.grad_norm < 1e-4) {
                setGradAlert(`Possible vanishing gradient detected at step ${data.step} (grad_norm=${data.grad_norm.toExponential(2)})`);
              }
            }
            return next;
          });
          setRun((prev) =>
            prev
              ? { ...prev, current_step: data.step, current_epoch: data.epoch, total_steps: data.total_steps ?? prev.total_steps, status: "running" }
              : prev
          );
        } else if (data.event === "stage") {
          setLogLines((prev) => [...prev.slice(-100), `[stage] ${data.stage}`]);
        } else if (data.event === "checkpoint") {
          setLogLines((prev) => [...prev.slice(-100), `[checkpoint] step ${data.step}`]);
          checkpointsApi.list(runId).then(setCheckpoints).catch(() => {});
        } else if (data.event === "error") {
          toast.error(`Training error: ${data.message}`);
          setLogLines((prev) => [...prev.slice(-100), `[error] ${data.message}`]);
        } else if (data.event === "done") {
          toast.success("Training completed");
          setRun((prev) => (prev ? { ...prev, status: "completed" } : prev));
        } else if (data.event === "process_exit") {
          trainingApi.get(runId).then(setRun).catch(() => {});
        } else if (data.event === "info") {
          setLogLines((prev) => [
            ...prev.slice(-100),
            `[info] trainable_params=${data.trainable_params} total_params=${data.total_params}`,
          ]);
        }
      } catch {
        // ignore malformed frames
      }
    };
    ws.onerror = () => {};

    return () => ws.close();
  }, [runId]);

  const cancel = async () => {
    try {
      await trainingApi.cancel(runId);
      toast.success("Cancellation requested");
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Failed to cancel run");
    }
  };

  const exportCheckpoint = async (format: "safetensors" | "gguf") => {
    try {
      toast.info(`Exporting as ${format}… this can take a while for large models`);
      const result = await checkpointsApi.export(runId, { format });
      toast.success(`Export complete: ${JSON.stringify(result)}`);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Export failed");
    }
  };

  if (!run) {
    return (
      <>
        <Topbar title="Training Run" />
        <main className="flex-1 p-6 text-sm text-muted-foreground">Loading…</main>
      </>
    );
  }

  const progressPct = run.total_steps ? Math.min(100, (run.current_step / run.total_steps) * 100) : 0;
  const latestGpuMem = metrics.length ? metrics[metrics.length - 1].gpu_mem_used_gb : null;

  return (
    <>
      <Topbar title={run.name} />
      <main className="flex-1 overflow-y-auto p-6 space-y-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Badge variant="outline" className={STATUS_STYLES[run.status] ?? ""}>
              {run.status}
            </Badge>
            <span className="text-sm text-muted-foreground">
              {run.base_model_repo_id} · {run.method.toUpperCase()} · {run.objective.toUpperCase()}
            </span>
          </div>
          {(run.status === "running" || run.status === "preparing") && (
            <Button variant="destructive" size="sm" onClick={cancel}>
              <Ban className="size-4" /> Cancel
            </Button>
          )}
        </div>

        {gradAlert && (
          <Alert variant="destructive">
            <AlertTriangle className="size-4" />
            <AlertTitle>Gradient anomaly</AlertTitle>
            <AlertDescription>{gradAlert}</AlertDescription>
          </Alert>
        )}

        {run.error_message && (
          <Alert variant="destructive">
            <AlertTitle>Run failed</AlertTitle>
            <AlertDescription className="whitespace-pre-wrap text-xs">{run.error_message}</AlertDescription>
          </Alert>
        )}

        <Card>
          <CardContent className="pt-6 space-y-2">
            <div className="flex justify-between text-sm">
              <span>
                Step {run.current_step}
                {run.total_steps ? ` / ${run.total_steps}` : ""} · Epoch {run.current_epoch?.toFixed(2)}
              </span>
              <span className="text-muted-foreground">
                {latestGpuMem != null ? `${latestGpuMem.toFixed(1)} GB GPU mem` : ""}
              </span>
            </div>
            <Progress value={progressPct} />
          </CardContent>
        </Card>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <ChartCard title="Loss">
            <LineChart data={metrics}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="step" stroke="var(--muted-foreground)" fontSize={11} />
              <YAxis stroke="var(--muted-foreground)" fontSize={11} />
              <RechartsTooltip contentStyle={{ background: "var(--card)", border: "1px solid var(--border)" }} />
              <Line type="monotone" dataKey="loss" stroke="#8b5cf6" dot={false} strokeWidth={2} name="train loss" />
              <Line type="monotone" dataKey="eval_loss" stroke="#22d3ee" dot={false} strokeWidth={2} name="eval loss" />
            </LineChart>
          </ChartCard>

          <ChartCard title="Learning Rate">
            <LineChart data={metrics}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="step" stroke="var(--muted-foreground)" fontSize={11} />
              <YAxis stroke="var(--muted-foreground)" fontSize={11} />
              <RechartsTooltip contentStyle={{ background: "var(--card)", border: "1px solid var(--border)" }} />
              <Line type="monotone" dataKey="learning_rate" stroke="#f59e0b" dot={false} strokeWidth={2} />
            </LineChart>
          </ChartCard>

          <ChartCard title="Gradient Norm">
            <LineChart data={metrics}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="step" stroke="var(--muted-foreground)" fontSize={11} />
              <YAxis stroke="var(--muted-foreground)" fontSize={11} />
              <RechartsTooltip contentStyle={{ background: "var(--card)", border: "1px solid var(--border)" }} />
              <Line type="monotone" dataKey="grad_norm" stroke="#ef4444" dot={false} strokeWidth={2} />
            </LineChart>
          </ChartCard>

          <ChartCard title="GPU Memory (GB)">
            <AreaChart data={metrics}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="step" stroke="var(--muted-foreground)" fontSize={11} />
              <YAxis stroke="var(--muted-foreground)" fontSize={11} />
              <RechartsTooltip contentStyle={{ background: "var(--card)", border: "1px solid var(--border)" }} />
              <Area type="monotone" dataKey="gpu_mem_used_gb" stroke="#10b981" fill="#10b98133" strokeWidth={2} />
            </AreaChart>
          </ChartCard>
        </div>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Checkpoints</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {checkpoints.length === 0 && <p className="text-sm text-muted-foreground">No checkpoints saved yet.</p>}
            {checkpoints.map((c) => (
              <div key={c.id} className="flex items-center justify-between rounded-lg border border-border px-3 py-2 text-sm">
                <span>
                  step {c.step} · epoch {c.epoch.toFixed(2)}
                  {c.eval_loss != null ? ` · eval_loss ${c.eval_loss.toFixed(4)}` : ""}
                  {c.is_best && <Badge className="ml-2">best</Badge>}
                </span>
                <span className="text-xs text-muted-foreground">{(c.size_bytes / 1e9).toFixed(2)} GB</span>
              </div>
            ))}
            {(run.status === "completed" || checkpoints.length > 0) && (
              <div className="flex gap-2 pt-2">
                <Button variant="secondary" size="sm" onClick={() => exportCheckpoint("safetensors")}>
                  <Download className="size-4" /> Export merged (safetensors)
                </Button>
                <Button variant="secondary" size="sm" onClick={() => exportCheckpoint("gguf")}>
                  <Download className="size-4" /> Export GGUF
                </Button>
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Logs</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="rounded-lg bg-black/40 border border-border p-3 h-48 overflow-y-auto font-mono text-xs space-y-0.5">
              {logLines.length === 0 && <span className="text-muted-foreground">Waiting for events…</span>}
              {logLines.map((line, i) => (
                <div key={i}>{line}</div>
              ))}
            </div>
          </CardContent>
        </Card>
      </main>
    </>
  );
}

function ChartCard({ title, children }: { title: string; children: React.ReactElement }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">{title}</CardTitle>
      </CardHeader>
      <CardContent className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          {children}
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
