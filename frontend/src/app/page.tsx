"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Topbar } from "@/components/layout/topbar";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { gpuApi, systemApi, trainingApi, datasetsApi, modelsApi } from "@/lib/api/endpoints";
import type { SystemStats, TrainingRun, DatasetSummary, ModelEntry } from "@/lib/api/types";
import { Cpu, Database, Boxes, Activity } from "lucide-react";

export default function DashboardPage() {
  const [stats, setStats] = useState<SystemStats | null>(null);
  const [sysInfo, setSysInfo] = useState<Record<string, unknown> | null>(null);
  const [runs, setRuns] = useState<TrainingRun[]>([]);
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [models, setModels] = useState<ModelEntry[]>([]);

  useEffect(() => {
    gpuApi.stats().then(setStats).catch(() => {});
    systemApi.info().then(setSysInfo).catch(() => {});
    trainingApi.list().then(setRuns).catch(() => {});
    datasetsApi.list().then(setDatasets).catch(() => {});
    modelsApi.library().then(setModels).catch(() => {});
    const interval = setInterval(() => gpuApi.stats().then(setStats).catch(() => {}), 3000);
    return () => clearInterval(interval);
  }, []);

  const gpu = stats?.gpus?.[0];
  const activeRuns = runs.filter((r) => r.status === "running" || r.status === "preparing");

  return (
    <>
      <Topbar title="Dashboard" />
      <main className="flex-1 overflow-y-auto p-6 space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <StatCard
            icon={<Cpu className="size-4" />}
            label="GPU"
            value={gpu ? `${gpu.utilization_pct?.toFixed(0)}%` : "—"}
            sub={gpu ? `${(gpu.memory_used_mb! / 1024).toFixed(1)} / ${(gpu.memory_total_mb! / 1024).toFixed(1)} GB` : "no data"}
          />
          <StatCard icon={<Activity className="size-4" />} label="Active Runs" value={String(activeRuns.length)} sub={`${runs.length} total`} />
          <StatCard icon={<Database className="size-4" />} label="Datasets" value={String(datasets.length)} sub="in library" />
          <StatCard icon={<Boxes className="size-4" />} label="Downloaded Models" value={String(models.filter((m) => m.downloaded).length)} sub="ready to use" />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Recent Training Runs</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              {runs.length === 0 && <p className="text-sm text-muted-foreground">No training runs yet.</p>}
              {runs.slice(0, 6).map((r) => (
                <Link
                  key={r.id}
                  href={`/training/${r.id}`}
                  className="flex items-center justify-between rounded-lg border border-border px-3 py-2 hover:bg-accent/50 transition-colors"
                >
                  <div>
                    <p className="text-sm font-medium">{r.name}</p>
                    <p className="text-xs text-muted-foreground">{r.base_model_repo_id}</p>
                  </div>
                  <StatusBadge status={r.status} />
                </Link>
              ))}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Environment</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-sm">
              {sysInfo ? (
                <>
                  <Row k="Torch" v={String((sysInfo.torch as Record<string, unknown>)?.version ?? "—")} />
                  <Row k="CUDA" v={String((sysInfo.torch as Record<string, unknown>)?.cuda_version ?? "—")} />
                  <Row k="Device" v={String((sysInfo.torch as Record<string, unknown>)?.device_name ?? "—")} />
                  <Row k="Python" v={String(sysInfo.python_version ?? "—").split(" ")[0]} />
                  <Row k="Platform" v={String(sysInfo.platform ?? "—")} />
                </>
              ) : (
                <p className="text-muted-foreground">Loading…</p>
              )}
            </CardContent>
          </Card>
        </div>
      </main>
    </>
  );
}

function StatCard({ icon, label, value, sub }: { icon: React.ReactNode; label: string; value: string; sub: string }) {
  return (
    <Card>
      <CardContent className="pt-6">
        <div className="flex items-center gap-2 text-muted-foreground text-xs mb-2">
          {icon}
          {label}
        </div>
        <div className="text-2xl font-semibold">{value}</div>
        <div className="text-xs text-muted-foreground mt-1">{sub}</div>
      </CardContent>
    </Card>
  );
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-muted-foreground">{k}</span>
      <span className="font-mono text-xs">{v}</span>
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const variant: Record<string, string> = {
    running: "bg-emerald-500/15 text-emerald-400 border-emerald-500/30",
    preparing: "bg-amber-500/15 text-amber-400 border-amber-500/30",
    queued: "bg-slate-500/15 text-slate-400 border-slate-500/30",
    completed: "bg-blue-500/15 text-blue-400 border-blue-500/30",
    failed: "bg-red-500/15 text-red-400 border-red-500/30",
    cancelled: "bg-slate-500/15 text-slate-400 border-slate-500/30",
  };
  return <Badge className={variant[status] ?? ""} variant="outline">{status}</Badge>;
}
