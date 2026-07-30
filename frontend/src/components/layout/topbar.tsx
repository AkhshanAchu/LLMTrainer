"use client";

import { GpuBadge } from "@/components/gpu/gpu-badge";

export function Topbar({ title }: { title: string }) {
  return (
    <header className="h-16 shrink-0 border-b border-border flex items-center justify-between px-6 bg-background/60 backdrop-blur-xl sticky top-0 z-10">
      <h1 className="text-lg font-semibold tracking-tight">{title}</h1>
      <GpuBadge />
    </header>
  );
}
