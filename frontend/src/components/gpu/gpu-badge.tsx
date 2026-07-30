"use client";

import { useEffect, useState } from "react";
import { Cpu } from "lucide-react";
import { gpuApi } from "@/lib/api/endpoints";
import type { SystemStats } from "@/lib/api/types";

export function GpuBadge() {
  const [stats, setStats] = useState<SystemStats | null>(null);

  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      try {
        const data = await gpuApi.stats();
        if (!cancelled) setStats(data);
      } catch {
        if (!cancelled) setStats(null);
      }
    };
    poll();
    const interval = setInterval(poll, 2000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  const gpu = stats?.gpus?.[0];

  return (
    <div className="flex items-center gap-2 rounded-full border border-border bg-card/50 px-3 py-1.5 text-xs">
      <Cpu className="size-3.5 text-emerald-400" />
      {gpu ? (
        <>
          <span className="font-medium">{gpu.name}</span>
          <span className="text-muted-foreground">
            {gpu.utilization_pct?.toFixed(0)}% · {(gpu.memory_used_mb! / 1024).toFixed(1)}/
            {(gpu.memory_total_mb! / 1024).toFixed(1)} GB · {gpu.temperature_c?.toFixed(0)}°C
          </span>
        </>
      ) : (
        <span className="text-muted-foreground">No GPU data</span>
      )}
    </div>
  );
}
