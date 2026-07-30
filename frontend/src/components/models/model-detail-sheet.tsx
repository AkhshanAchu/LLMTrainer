"use client";

import { useEffect, useState } from "react";
import { toast } from "sonner";
import { Download, CheckCircle2 } from "lucide-react";
import { modelsApi } from "@/lib/api/endpoints";
import { useAppStore } from "@/store/app-store";
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetDescription } from "@/components/ui/sheet";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { ScrollArea } from "@/components/ui/scroll-area";
import { VramEstimator } from "@/components/models/vram-estimator";
import { formatParamCount, humanizeKey } from "@/lib/format";

interface ModelDetailSheetProps {
  repoId: string | null;
  displayName?: string;
  downloaded?: boolean;
  onOpenChange: (open: boolean) => void;
  onDownload?: (repoId: string, displayName?: string) => void;
}

// Fields we prefer to render up top in a known order; anything else falls back to the generic grid.
const PRIORITY_FIELDS = [
  "architecture",
  "model_type",
  "hidden_size",
  "num_hidden_layers",
  "num_attention_heads",
  "vocab_size",
  "torch_dtype",
  "context_length",
  "license",
  "pipeline_tag",
  "gated",
];

const SKIP_FIELDS = new Set(["repo_id", "raw_config", "siblings", "tags", "param_count_is_estimate"]);

export function ModelDetailSheet({ repoId, displayName, downloaded, onOpenChange, onDownload }: ModelDetailSheetProps) {
  const [info, setInfo] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(false);
  const setSelectedModel = useAppStore((s) => s.setSelectedModel);

  useEffect(() => {
    if (!repoId) {
      setInfo(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setInfo(null);
    modelsApi
      .info(repoId)
      .then((res) => {
        if (!cancelled) setInfo(res);
      })
      .catch((err) => {
        if (!cancelled) toast.error(`Failed to load model info: ${err instanceof Error ? err.message : "unknown error"}`);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [repoId]);

  const rawInfo = (info?.info as Record<string, unknown> | undefined) ?? info ?? undefined;
  const tags = (rawInfo?.tags as string[] | undefined) ?? undefined;
  const paramCount = rawInfo?.param_count as number | undefined;

  const fieldEntries = rawInfo
    ? Object.entries(rawInfo)
        .filter(([k, v]) => !SKIP_FIELDS.has(k) && v !== null && v !== undefined && v !== "")
        .sort((a, b) => {
          const ai = PRIORITY_FIELDS.indexOf(a[0]);
          const bi = PRIORITY_FIELDS.indexOf(b[0]);
          if (ai === -1 && bi === -1) return 0;
          if (ai === -1) return 1;
          if (bi === -1) return -1;
          return ai - bi;
        })
    : [];

  function handleDownload() {
    if (!repoId) return;
    onDownload?.(repoId, displayName);
  }

  function handleSelect() {
    if (!repoId) return;
    setSelectedModel(repoId);
    toast.success(`Selected ${displayName ?? repoId} for training`);
  }

  return (
    <Sheet open={!!repoId} onOpenChange={onOpenChange}>
      <SheetContent side="right" className="w-full sm:max-w-lg overflow-y-auto">
        <SheetHeader>
          <SheetTitle>{displayName ?? repoId ?? "Model details"}</SheetTitle>
          <SheetDescription className="font-mono text-xs break-all">{repoId}</SheetDescription>
        </SheetHeader>

        <div className="flex-1 min-h-0 px-4 pb-4">
          <ScrollArea className="h-full pr-2">
            <div className="space-y-6 pb-2">
              <div className="flex flex-wrap gap-2">
                {downloaded && (
                  <Badge variant="outline" className="border-emerald-500/30 bg-emerald-500/10 text-emerald-400">
                    Downloaded
                  </Badge>
                )}
                {paramCount !== undefined && <Badge variant="secondary">{formatParamCount(paramCount)} params</Badge>}
              </div>

              <div className="flex flex-wrap gap-2">
                <Button size="sm" onClick={handleDownload}>
                  <Download className="size-3.5" />
                  Download
                </Button>
                {downloaded && (
                  <Button size="sm" variant="outline" onClick={handleSelect}>
                    <CheckCircle2 className="size-3.5" />
                    Select for training
                  </Button>
                )}
              </div>

              <Separator />

              <div>
                <h3 className="mb-3 text-sm font-medium">Metadata</h3>
                {loading && (
                  <div className="space-y-2">
                    <Skeleton className="h-4 w-full" />
                    <Skeleton className="h-4 w-5/6" />
                    <Skeleton className="h-4 w-2/3" />
                  </div>
                )}
                {!loading && fieldEntries.length === 0 && (
                  <p className="text-sm text-muted-foreground">No metadata available for this model.</p>
                )}
                {!loading && fieldEntries.length > 0 && (
                  <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
                    {fieldEntries.map(([k, v]) => (
                      <div key={k} className="contents">
                        <span className="text-muted-foreground">{humanizeKey(k)}</span>
                        <span className="truncate font-mono text-xs" title={String(v)}>
                          {typeof v === "boolean" ? (v ? "Yes" : "No") : String(v)}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
                {!loading && tags && tags.length > 0 && (
                  <div className="mt-3 flex flex-wrap gap-1.5">
                    {tags.slice(0, 12).map((t) => (
                      <Badge key={t} variant="outline" className="text-[10px]">
                        {t}
                      </Badge>
                    ))}
                  </div>
                )}
              </div>

              <Separator />

              <div>
                <h3 className="mb-3 text-sm font-medium">VRAM Estimator</h3>
                {repoId && <VramEstimator repoId={repoId} />}
              </div>
            </div>
          </ScrollArea>
        </div>
      </SheetContent>
    </Sheet>
  );
}
