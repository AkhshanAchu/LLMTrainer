"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Topbar } from "@/components/layout/topbar";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { buttonVariants } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { trainingApi } from "@/lib/api/endpoints";
import type { TrainingRun } from "@/lib/api/types";
import { Plus } from "lucide-react";

const STATUS_STYLES: Record<string, string> = {
  running: "bg-emerald-500/15 text-emerald-400 border-emerald-500/30",
  preparing: "bg-amber-500/15 text-amber-400 border-amber-500/30",
  queued: "bg-slate-500/15 text-slate-400 border-slate-500/30",
  completed: "bg-blue-500/15 text-blue-400 border-blue-500/30",
  failed: "bg-red-500/15 text-red-400 border-red-500/30",
  cancelled: "bg-slate-500/15 text-slate-400 border-slate-500/30",
  paused: "bg-purple-500/15 text-purple-400 border-purple-500/30",
};

export default function TrainingListPage() {
  const [runs, setRuns] = useState<TrainingRun[] | null>(null);

  useEffect(() => {
    let cancelled = false;
    const load = () => trainingApi.list().then((r) => !cancelled && setRuns(r)).catch(() => !cancelled && setRuns([]));
    load();
    const interval = setInterval(load, 4000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  return (
    <>
      <Topbar title="Training" />
      <main className="flex-1 overflow-y-auto p-6 space-y-4">
        <div className="flex items-center justify-between">
          <p className="text-sm text-muted-foreground">Configure and monitor fine-tuning runs.</p>
          <Link href="/training/new" className={buttonVariants({})}>
            <Plus className="size-4" /> New Training Run
          </Link>
        </div>

        {runs === null && (
          <div className="space-y-2">
            <Skeleton className="h-16 w-full" />
            <Skeleton className="h-16 w-full" />
          </div>
        )}

        {runs !== null && runs.length === 0 && (
          <Card>
            <CardContent className="py-12 text-center text-sm text-muted-foreground">
              No training runs yet. Create one to get started.
            </CardContent>
          </Card>
        )}

        <div className="space-y-2">
          {runs?.map((r) => (
            <Link key={r.id} href={`/training/${r.id}`}>
              <Card className="hover:bg-accent/40 transition-colors">
                <CardContent className="py-4 flex items-center justify-between">
                  <div>
                    <p className="font-medium text-sm">{r.name}</p>
                    <p className="text-xs text-muted-foreground">
                      {r.base_model_repo_id} · {r.method.toUpperCase()} · {r.objective.toUpperCase()}
                    </p>
                  </div>
                  <div className="flex items-center gap-4">
                    <div className="text-xs text-muted-foreground text-right">
                      <div>
                        step {r.current_step}
                        {r.total_steps ? ` / ${r.total_steps}` : ""}
                      </div>
                      <div>epoch {r.current_epoch?.toFixed(2)}</div>
                    </div>
                    <Badge variant="outline" className={STATUS_STYLES[r.status] ?? ""}>
                      {r.status}
                    </Badge>
                  </div>
                </CardContent>
              </Card>
            </Link>
          ))}
        </div>
      </main>
    </>
  );
}
