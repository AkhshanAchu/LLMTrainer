"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { toast } from "sonner";
import { Topbar } from "@/components/layout/topbar";
import { UploadDropzone } from "@/components/datasets/upload-dropzone";
import { FormatBadge } from "@/components/datasets/format-badge";
import { QualityBadge } from "@/components/datasets/quality-badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { datasetsApi } from "@/lib/api/endpoints";
import { ApiError } from "@/lib/api/client";
import type { DatasetSummary } from "@/lib/api/types";
import { useAppStore } from "@/store/app-store";
import { Database, Trash2, CheckCircle2 } from "lucide-react";

export default function DatasetsPage() {
  const router = useRouter();
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [pendingDelete, setPendingDelete] = useState<DatasetSummary | null>(null);
  const [deleting, setDeleting] = useState(false);

  const selectedDatasetId = useAppStore((s) => s.selectedDatasetId);
  const setSelectedDataset = useAppStore((s) => s.setSelectedDataset);

  const refresh = useCallback(async () => {
    try {
      const data = await datasetsApi.list();
      setDatasets(data);
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Failed to load datasets");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const handleUpload = useCallback(
    async (file: File) => {
      setUploading(true);
      try {
        const ds = await datasetsApi.upload(file);
        toast.success(`Uploaded "${ds.name}" (${ds.num_examples} examples)`);
        await refresh();
      } catch (err) {
        toast.error(err instanceof ApiError ? err.message : "Upload failed");
      } finally {
        setUploading(false);
      }
    },
    [refresh]
  );

  const handleDelete = useCallback(async () => {
    if (!pendingDelete) return;
    setDeleting(true);
    try {
      await datasetsApi.remove(pendingDelete.id);
      toast.success(`Deleted "${pendingDelete.name}"`);
      if (selectedDatasetId === pendingDelete.id) setSelectedDataset(null);
      setPendingDelete(null);
      await refresh();
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Delete failed");
    } finally {
      setDeleting(false);
    }
  }, [pendingDelete, refresh, selectedDatasetId, setSelectedDataset]);

  return (
    <>
      <Topbar title="Dataset Manager" />
      <main className="flex-1 overflow-y-auto p-6 space-y-6">
        <UploadDropzone onUpload={handleUpload} uploading={uploading} />

        <Card>
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2">
              <Database className="size-4" />
              Datasets
            </CardTitle>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="space-y-2">
                <Skeleton className="h-9 w-full" />
                <Skeleton className="h-9 w-full" />
                <Skeleton className="h-9 w-full" />
              </div>
            ) : datasets.length === 0 ? (
              <p className="text-sm text-muted-foreground py-8 text-center">
                No datasets yet. Upload a file above to get started.
              </p>
            ) : (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Name</TableHead>
                    <TableHead>Format</TableHead>
                    <TableHead>Examples</TableHead>
                    <TableHead>Quality</TableHead>
                    <TableHead>Created</TableHead>
                    <TableHead className="text-right">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {datasets.map((ds) => {
                    return (
                      <TableRow
                        key={ds.id}
                        className="cursor-pointer"
                        onClick={() => router.push(`/datasets/${ds.id}`)}
                      >
                        <TableCell className="font-medium">
                          <div className="flex items-center gap-2">
                            {ds.name}
                            {selectedDatasetId === ds.id && (
                              <CheckCircle2 className="size-3.5 text-emerald-400" />
                            )}
                          </div>
                        </TableCell>
                        <TableCell>
                          <FormatBadge format={ds.detected_format} />
                        </TableCell>
                        <TableCell>{ds.num_examples ?? "—"}</TableCell>
                        <TableCell>
                          <QualityBadge score={ds.quality_score} />
                        </TableCell>
                        <TableCell className="text-xs text-muted-foreground">
                          {ds.created_at ? new Date(ds.created_at).toLocaleString() : "—"}
                        </TableCell>
                        <TableCell className="text-right">
                          <div className="flex items-center justify-end gap-1.5" onClick={(e) => e.stopPropagation()}>
                            <Button
                              variant={selectedDatasetId === ds.id ? "secondary" : "outline"}
                              size="sm"
                              onClick={() => {
                                setSelectedDataset(ds.id);
                                toast.success(`"${ds.name}" selected for training`);
                              }}
                            >
                              {selectedDatasetId === ds.id ? "Selected" : "Select for training"}
                            </Button>
                            <Button
                              variant="destructive"
                              size="icon-sm"
                              onClick={() => setPendingDelete(ds)}
                            >
                              <Trash2 className="size-3.5" />
                            </Button>
                          </div>
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            )}
          </CardContent>
        </Card>
      </main>

      <Dialog open={!!pendingDelete} onOpenChange={(open) => !open && setPendingDelete(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete dataset?</DialogTitle>
            <DialogDescription>
              This will permanently delete &quot;{pendingDelete?.name}&quot; and its stored file. This
              action cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setPendingDelete(null)} disabled={deleting}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={handleDelete} disabled={deleting}>
              {deleting ? "Deleting…" : "Delete"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
