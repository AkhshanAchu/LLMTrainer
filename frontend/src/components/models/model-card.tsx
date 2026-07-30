"use client";

import { Download, Eye, Bookmark, Trash2, CheckCircle2 } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { formatParamCount, formatContextLength } from "@/lib/format";

export interface ModelCardData {
  repoId: string;
  displayName: string;
  family?: string | null;
  architecture?: string | null;
  paramCount?: number | null;
  contextLength?: number | null;
  license?: string | null;
  isVision?: boolean;
  downloaded?: boolean;
  bookmarked?: boolean;
}

interface ModelCardProps {
  model: ModelCardData;
  onOpenDetail: (repoId: string) => void;
  onDownload?: (repoId: string, displayName?: string) => void;
  onSelect?: (repoId: string) => void;
  onToggleBookmark?: (repoId: string) => void;
  onDelete?: (repoId: string) => void;
  showLibraryActions?: boolean;
}

export function ModelCard({
  model,
  onOpenDetail,
  onDownload,
  onSelect,
  onToggleBookmark,
  onDelete,
  showLibraryActions,
}: ModelCardProps) {
  return (
    <Card className="transition-colors hover:border-foreground/20">
      <CardContent className="pt-5">
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0">
            <p className="truncate font-medium">{model.displayName}</p>
            <p className="truncate font-mono text-xs text-muted-foreground">{model.repoId}</p>
          </div>
          {model.downloaded && (
            <Badge variant="outline" className="shrink-0 border-emerald-500/30 bg-emerald-500/10 text-emerald-400">
              Downloaded
            </Badge>
          )}
        </div>

        <div className="mt-3 flex flex-wrap gap-1.5">
          {model.family && <Badge variant="secondary">{model.family}</Badge>}
          {model.architecture && <Badge variant="outline">{model.architecture}</Badge>}
          {model.paramCount != null && <Badge variant="outline">{formatParamCount(model.paramCount)}</Badge>}
          {model.contextLength != null && <Badge variant="outline">{formatContextLength(model.contextLength)} ctx</Badge>}
          {model.license && <Badge variant="outline">{model.license}</Badge>}
          {model.isVision && (
            <Badge variant="outline" className="border-violet-500/30 bg-violet-500/10 text-violet-400">
              Vision
            </Badge>
          )}
        </div>

        <div className="mt-4 flex flex-wrap items-center gap-2">
          <Button size="sm" variant="outline" onClick={() => onOpenDetail(model.repoId)}>
            <Eye className="size-3.5" />
            Details
          </Button>
          {onDownload && !model.downloaded && (
            <Button size="sm" onClick={() => onDownload(model.repoId, model.displayName)}>
              <Download className="size-3.5" />
              Download
            </Button>
          )}
          {onSelect && model.downloaded && (
            <Button size="sm" variant="outline" onClick={() => onSelect(model.repoId)}>
              <CheckCircle2 className="size-3.5" />
              Select for training
            </Button>
          )}
          {showLibraryActions && (
            <>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => onToggleBookmark?.(model.repoId)}
                className={model.bookmarked ? "text-amber-400" : ""}
              >
                <Bookmark className="size-3.5" fill={model.bookmarked ? "currentColor" : "none"} />
                {model.bookmarked ? "Bookmarked" : "Bookmark"}
              </Button>
              <Button size="sm" variant="ghost" className="text-destructive hover:text-destructive" onClick={() => onDelete?.(model.repoId)}>
                <Trash2 className="size-3.5" />
                Delete
              </Button>
            </>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
