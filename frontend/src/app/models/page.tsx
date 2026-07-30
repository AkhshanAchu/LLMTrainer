"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";
import { Search, Boxes } from "lucide-react";
import { Topbar } from "@/components/layout/topbar";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { modelsApi } from "@/lib/api/endpoints";
import type { CuratedModel, ModelEntry } from "@/lib/api/types";
import { useAppStore } from "@/store/app-store";
import { ModelCard, type ModelCardData } from "@/components/models/model-card";
import { ModelDetailSheet } from "@/components/models/model-detail-sheet";

function curatedToCard(m: CuratedModel, libraryByRepoId: Map<string, ModelEntry>): ModelCardData {
  const entry = libraryByRepoId.get(m.repo_id);
  return {
    repoId: m.repo_id,
    displayName: m.display_name,
    family: m.family,
    paramCount: m.param_count,
    isVision: m.is_vision,
    downloaded: entry?.downloaded ?? false,
    bookmarked: entry?.bookmarked ?? false,
  };
}

function searchResultToCard(m: Record<string, unknown>, libraryByRepoId: Map<string, ModelEntry>): ModelCardData {
  const repoId = String(m.id ?? m.repo_id ?? m.modelId ?? "");
  const entry = libraryByRepoId.get(repoId);
  return {
    repoId,
    displayName: repoId.split("/").pop() || repoId,
    family: (m.pipeline_tag as string) ?? undefined,
    license: (m.license as string) ?? (m.card_data as Record<string, unknown> | undefined)?.license as string | undefined,
    downloaded: entry?.downloaded ?? false,
    bookmarked: entry?.bookmarked ?? false,
  };
}

function libraryToCard(m: ModelEntry): ModelCardData {
  return {
    repoId: m.repo_id,
    displayName: m.display_name,
    architecture: m.architecture,
    paramCount: m.param_count,
    contextLength: m.context_length,
    license: m.license,
    isVision: m.is_vision,
    downloaded: m.downloaded,
    bookmarked: m.bookmarked,
  };
}

export default function ModelsPage() {
  const setSelectedModel = useAppStore((s) => s.setSelectedModel);

  // Curated tab state
  const [curated, setCurated] = useState<CuratedModel[]>([]);
  const [curatedLoading, setCuratedLoading] = useState(true);

  // Search tab state
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<Record<string, unknown>[]>([]);
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Library tab state
  const [library, setLibrary] = useState<ModelEntry[]>([]);
  const [libraryLoading, setLibraryLoading] = useState(true);

  // Shared: detail sheet
  const [detailRepoId, setDetailRepoId] = useState<string | null>(null);
  const [detailDisplayName, setDetailDisplayName] = useState<string | undefined>(undefined);

  // Delete confirm dialog
  const [pendingDelete, setPendingDelete] = useState<ModelEntry | null>(null);

  const loadLibrary = useCallback(() => {
    setLibraryLoading(true);
    modelsApi
      .library()
      .then(setLibrary)
      .catch((err) => {
        toast.error(`Failed to load library: ${err instanceof Error ? err.message : "unknown error"}`);
      })
      .finally(() => setLibraryLoading(false));
  }, []);

  useEffect(() => {
    setCuratedLoading(true);
    modelsApi
      .curated()
      .then(setCurated)
      .catch((err) => {
        toast.error(`Failed to load curated models: ${err instanceof Error ? err.message : "unknown error"}`);
      })
      .finally(() => setCuratedLoading(false));
    loadLibrary();
  }, [loadLibrary]);

  // Debounced search
  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    const q = searchQuery.trim();
    if (!q) {
      setSearchResults([]);
      setSearchError(null);
      setSearchLoading(false);
      return;
    }
    setSearchLoading(true);
    debounceRef.current = setTimeout(() => {
      modelsApi
        .search(q)
        .then((res) => {
          setSearchResults(res);
          setSearchError(null);
        })
        .catch((err) => {
          const message = err instanceof Error ? err.message : "Search failed";
          setSearchError(message);
          setSearchResults([]);
          toast.error(`Search Hub error: ${message}`);
        })
        .finally(() => setSearchLoading(false));
    }, 450);
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [searchQuery]);

  const libraryByRepoId = useMemo(() => {
    const map = new Map<string, ModelEntry>();
    for (const m of library) map.set(m.repo_id, m);
    return map;
  }, [library]);

  function handleOpenDetail(repoId: string, displayName?: string) {
    setDetailRepoId(repoId);
    setDetailDisplayName(displayName);
  }

  function handleDownload(repoId: string, displayName?: string) {
    modelsApi
      .download(repoId, displayName)
      .then(() => {
        toast.success(`Download started for ${displayName ?? repoId}`, {
          description: "Check My Library later to see when it finishes.",
        });
      })
      .catch((err) => {
        toast.error(`Failed to start download: ${err instanceof Error ? err.message : "unknown error"}`);
      });
  }

  function handleSelect(repoId: string) {
    const entry = libraryByRepoId.get(repoId);
    setSelectedModel(repoId);
    toast.success(`Selected ${entry?.display_name ?? repoId} for training`);
  }

  function handleToggleBookmark(repoId: string) {
    const entry = libraryByRepoId.get(repoId);
    if (!entry) return;
    // optimistic update
    setLibrary((prev) => prev.map((m) => (m.id === entry.id ? { ...m, bookmarked: !m.bookmarked } : m)));
    modelsApi.toggleBookmark(entry.id).catch((err) => {
      // revert
      setLibrary((prev) => prev.map((m) => (m.id === entry.id ? { ...m, bookmarked: entry.bookmarked } : m)));
      toast.error(`Failed to toggle bookmark: ${err instanceof Error ? err.message : "unknown error"}`);
    });
  }

  function requestDelete(repoId: string) {
    const entry = libraryByRepoId.get(repoId);
    if (entry) setPendingDelete(entry);
  }

  function confirmDelete() {
    if (!pendingDelete) return;
    const entry = pendingDelete;
    setPendingDelete(null);
    modelsApi
      .remove(entry.id)
      .then(() => {
        setLibrary((prev) => prev.filter((m) => m.id !== entry.id));
        toast.success(`Removed ${entry.display_name} from library`);
      })
      .catch((err) => {
        toast.error(`Failed to delete model: ${err instanceof Error ? err.message : "unknown error"}`);
      });
  }

  const detailEntry = detailRepoId ? libraryByRepoId.get(detailRepoId) : undefined;

  return (
    <>
      <Topbar title="Model Manager" />
      <main className="flex-1 overflow-y-auto p-6 space-y-6">
        <Tabs defaultValue="curated">
          <TabsList>
            <TabsTrigger value="curated">Curated</TabsTrigger>
            <TabsTrigger value="search">Search Hub</TabsTrigger>
            <TabsTrigger value="library">My Library</TabsTrigger>
          </TabsList>

          {/* Curated */}
          <TabsContent value="curated" className="mt-4">
            {curatedLoading ? (
              <CardGridSkeleton />
            ) : curated.length === 0 ? (
              <EmptyState text="No curated models available right now." />
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
                {curated.map((m) => (
                  <ModelCard
                    key={m.repo_id}
                    model={curatedToCard(m, libraryByRepoId)}
                    onOpenDetail={(repoId) => handleOpenDetail(repoId, m.display_name)}
                    onDownload={handleDownload}
                    onSelect={handleSelect}
                  />
                ))}
              </div>
            )}
          </TabsContent>

          {/* Search Hub */}
          <TabsContent value="search" className="mt-4 space-y-4">
            <div className="relative max-w-md">
              <Search className="absolute left-2.5 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                placeholder="Search Hugging Face Hub…"
                className="pl-8"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>

            {searchLoading && <CardGridSkeleton count={3} />}

            {!searchLoading && searchError && (
              <EmptyState text={`Search failed: ${searchError}. The Hub may be unreachable — try again later.`} />
            )}

            {!searchLoading && !searchError && searchQuery.trim() === "" && (
              <EmptyState text="Type a model name or keyword to search the Hugging Face Hub." />
            )}

            {!searchLoading && !searchError && searchQuery.trim() !== "" && searchResults.length === 0 && (
              <EmptyState text={`No results for "${searchQuery}".`} />
            )}

            {!searchLoading && !searchError && searchResults.length > 0 && (
              <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
                {searchResults.map((m, i) => {
                  const card = searchResultToCard(m, libraryByRepoId);
                  return (
                    <ModelCard
                      key={card.repoId || i}
                      model={card}
                      onOpenDetail={(repoId) => handleOpenDetail(repoId, card.displayName)}
                      onDownload={handleDownload}
                      onSelect={handleSelect}
                    />
                  );
                })}
              </div>
            )}
          </TabsContent>

          {/* My Library */}
          <TabsContent value="library" className="mt-4">
            {libraryLoading ? (
              <CardGridSkeleton />
            ) : library.length === 0 ? (
              <EmptyState
                icon={<Boxes className="size-8 text-muted-foreground" />}
                text="No models downloaded yet — browse Curated or Search Hub to get started."
              />
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
                {library.map((m) => (
                  <ModelCard
                    key={m.id}
                    model={libraryToCard(m)}
                    onOpenDetail={(repoId) => handleOpenDetail(repoId, m.display_name)}
                    onSelect={handleSelect}
                    onToggleBookmark={handleToggleBookmark}
                    onDelete={requestDelete}
                    showLibraryActions
                  />
                ))}
              </div>
            )}
          </TabsContent>
        </Tabs>
      </main>

      <ModelDetailSheet
        repoId={detailRepoId}
        displayName={detailDisplayName}
        downloaded={detailEntry?.downloaded}
        onOpenChange={(open) => {
          if (!open) setDetailRepoId(null);
        }}
        onDownload={handleDownload}
      />

      <Dialog open={!!pendingDelete} onOpenChange={(open) => !open && setPendingDelete(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete model?</DialogTitle>
            <DialogDescription>
              This will remove <span className="font-medium text-foreground">{pendingDelete?.display_name}</span> from your library.
              This does not undo already-completed training runs that used it.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setPendingDelete(null)}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={confirmDelete}>
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}

function CardGridSkeleton({ count = 6 }: { count?: number }) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
      {Array.from({ length: count }).map((_, i) => (
        <div key={i} className="rounded-xl ring-1 ring-foreground/10 p-5 space-y-3">
          <Skeleton className="h-4 w-2/3" />
          <Skeleton className="h-3 w-1/2" />
          <div className="flex gap-1.5">
            <Skeleton className="h-5 w-14" />
            <Skeleton className="h-5 w-14" />
            <Skeleton className="h-5 w-14" />
          </div>
          <Skeleton className="h-7 w-24" />
        </div>
      ))}
    </div>
  );
}

function EmptyState({ text, icon }: { text: string; icon?: React.ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 rounded-xl border border-dashed border-border py-16 text-center">
      {icon}
      <p className="max-w-sm text-sm text-muted-foreground">{text}</p>
    </div>
  );
}
