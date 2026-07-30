"use client";

import { useCallback, useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { toast } from "sonner";
import { Topbar } from "@/components/layout/topbar";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { FormatBadge } from "@/components/datasets/format-badge";
import { QualityBadge } from "@/components/datasets/quality-badge";
import { StatsSection } from "@/components/datasets/stats-section";
import { ValidationSection } from "@/components/datasets/validation-section";
import { PreviewSection } from "@/components/datasets/preview-section";
import { TokenizerPreviewSection } from "@/components/datasets/tokenizer-preview-section";
import { RepairSection } from "@/components/datasets/repair-section";
import { datasetsApi } from "@/lib/api/endpoints";
import { ApiError } from "@/lib/api/client";
import type { Dataset } from "@/lib/api/types";
import { useAppStore } from "@/store/app-store";
import { ArrowLeft, RefreshCw, CheckCircle2 } from "lucide-react";

export default function DatasetDetailPage() {
  const params = useParams<{ datasetId: string }>();
  const router = useRouter();
  const datasetId = Number(params.datasetId);

  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [notFound, setNotFound] = useState(false);

  const selectedDatasetId = useAppStore((s) => s.selectedDatasetId);
  const setSelectedDataset = useAppStore((s) => s.setSelectedDataset);

  const load = useCallback(async () => {
    if (!Number.isFinite(datasetId)) {
      setNotFound(true);
      setLoading(false);
      return;
    }
    try {
      const ds = await datasetsApi.get(datasetId);
      setDataset(ds);
    } catch (err) {
      if (err instanceof ApiError && err.status === 404) {
        setNotFound(true);
      } else {
        toast.error(err instanceof ApiError ? err.message : "Failed to load dataset");
      }
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [datasetId]);

  useEffect(() => {
    load();
  }, [load]);

  const handleRefresh = async () => {
    setRefreshing(true);
    await load();
  };

  if (notFound) {
    return (
      <>
        <Topbar title="Dataset" />
        <main className="flex-1 overflow-y-auto p-6">
          <Card>
            <CardContent className="py-12 text-center space-y-3">
              <p className="text-sm text-muted-foreground">Dataset not found.</p>
              <Button variant="outline" onClick={() => router.push("/datasets")}>
                <ArrowLeft className="size-3.5" />
                Back to datasets
              </Button>
            </CardContent>
          </Card>
        </main>
      </>
    );
  }

  return (
    <>
      <Topbar title={dataset?.name ?? "Dataset"} />
      <main className="flex-1 overflow-y-auto p-6 space-y-6">
        <Button variant="ghost" size="sm" onClick={() => router.push("/datasets")} className="-ml-2">
          <ArrowLeft className="size-3.5" />
          Back to datasets
        </Button>

        {loading ? (
          <Card>
            <CardContent className="pt-6 space-y-3">
              <Skeleton className="h-6 w-64" />
              <Skeleton className="h-4 w-40" />
            </CardContent>
          </Card>
        ) : dataset ? (
          <>
            <Card>
              <CardContent className="pt-6">
                <div className="flex flex-wrap items-start justify-between gap-4">
                  <div className="space-y-2">
                    <div className="flex items-center gap-2 flex-wrap">
                      <h2 className="text-xl font-semibold">{dataset.name}</h2>
                      <FormatBadge format={dataset.detected_format} />
                      {selectedDatasetId === dataset.id && (
                        <span className="flex items-center gap-1 text-xs text-emerald-400">
                          <CheckCircle2 className="size-3.5" />
                          Selected for training
                        </span>
                      )}
                    </div>
                    <p className="text-sm text-muted-foreground">{dataset.source_filename}</p>
                    <div className="flex flex-wrap items-center gap-3 text-sm">
                      <span>
                        <span className="text-muted-foreground">Examples: </span>
                        <span className="font-medium">{dataset.num_examples}</span>
                      </span>
                      <span>
                        <span className="text-muted-foreground">Quality: </span>
                        <QualityBadge
                          score={
                            typeof (dataset.stats as Record<string, unknown> | undefined)?.quality_score ===
                            "number"
                              ? ((dataset.stats as Record<string, unknown>).quality_score as number)
                              : null
                          }
                        />
                      </span>
                      <span className="text-muted-foreground text-xs">
                        Created {dataset.created_at ? new Date(dataset.created_at).toLocaleString() : "—"}
                      </span>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Button variant="outline" size="sm" onClick={handleRefresh} disabled={refreshing}>
                      <RefreshCw className={`size-3.5 ${refreshing ? "animate-spin" : ""}`} />
                      Refresh
                    </Button>
                    <Button
                      variant={selectedDatasetId === dataset.id ? "secondary" : "default"}
                      size="sm"
                      onClick={() => {
                        setSelectedDataset(dataset.id);
                        toast.success(`"${dataset.name}" selected for training`);
                      }}
                    >
                      {selectedDatasetId === dataset.id ? "Selected" : "Select for training"}
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>

            <StatsSection stats={dataset.stats} />
            <ValidationSection report={dataset.validation_report} />
            <PreviewSection datasetId={dataset.id} />
            <TokenizerPreviewSection datasetId={dataset.id} />
            <RepairSection datasetId={dataset.id} />
          </>
        ) : null}
      </main>
    </>
  );
}
