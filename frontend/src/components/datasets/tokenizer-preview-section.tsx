"use client";

import { useState } from "react";
import { toast } from "sonner";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { datasetsApi } from "@/lib/api/endpoints";
import { ApiError } from "@/lib/api/client";
import { Loader2 } from "lucide-react";

function asRecord(v: unknown): Record<string, unknown> | undefined {
  return v && typeof v === "object" ? (v as Record<string, unknown>) : undefined;
}

export function TokenizerPreviewSection({ datasetId }: { datasetId: number }) {
  const [repoId, setRepoId] = useState("gpt2");
  const [maxSeqLen, setMaxSeqLen] = useState(2048);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<Record<string, unknown> | null>(null);

  const run = async () => {
    setLoading(true);
    setResult(null);
    try {
      const data = await datasetsApi.tokenizePreview(datasetId, {
        tokenizer_repo_id: repoId.trim() || undefined,
        max_seq_len: maxSeqLen,
      });
      setResult(data);
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Tokenizer preview failed");
    } finally {
      setLoading(false);
    }
  };

  const firstExample = asRecord(result?.first_example);
  const truncation = asRecord(result?.truncation_preview);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Tokenizer Preview</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap items-end gap-3">
          <div className="space-y-1.5">
            <Label htmlFor="tokenizer-repo">Tokenizer / model repo ID</Label>
            <Input
              id="tokenizer-repo"
              value={repoId}
              onChange={(e) => setRepoId(e.target.value)}
              placeholder="gpt2"
              className="w-56"
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="max-seq-len">Max sequence length</Label>
            <Input
              id="max-seq-len"
              type="number"
              min={1}
              value={maxSeqLen}
              onChange={(e) => setMaxSeqLen(Number(e.target.value) || 0)}
              className="w-36"
            />
          </div>
          <Button onClick={run} disabled={loading}>
            {loading && <Loader2 className="size-3.5 animate-spin" />}
            Run tokenizer preview
          </Button>
        </div>

        {result && (
          <div className="space-y-3 pt-2 border-t border-border">
            {result.error ? (
              <p className="text-sm text-destructive">{String(result.error)}</p>
            ) : (
              <>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                  <Stat label="Avg tokens/example" value={result.avg_tokens_per_example} />
                  <Stat label="Min / Max tokens" value={`${String(result.min_tokens ?? "—")} / ${String(result.max_tokens ?? "—")}`} />
                  <Stat label="Estimated total tokens" value={result.estimated_total_tokens} />
                  <Stat
                    label="Over limit (sample)"
                    value={
                      truncation
                        ? `${String(truncation.sampled_examples_over_limit ?? 0)} (${String(truncation.estimated_total_over_limit ?? 0)} est. total)`
                        : "—"
                    }
                  />
                </div>
                {result.used_fallback_tokenizer === true && (
                  <p className="text-xs text-amber-400">
                    Fell back to default tokenizer ({String(result.tokenizer_repo_id ?? "")}) — the requested repo
                    could not be loaded.
                  </p>
                )}
                {firstExample && (
                  <div>
                    <p className="text-xs text-muted-foreground mb-1.5">
                      First example formatted prompt ({String(firstExample.token_count ?? "?")} tokens)
                    </p>
                    <pre className="text-xs whitespace-pre-wrap break-words font-mono rounded-lg border border-border bg-muted/40 p-3 max-h-64 overflow-y-auto">
                      {String(firstExample.formatted_prompt ?? "")}
                    </pre>
                  </div>
                )}
              </>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function Stat({ label, value }: { label: string; value: unknown }) {
  return (
    <div>
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="text-lg font-semibold tabular-nums">{value === undefined || value === null ? "—" : String(value)}</p>
    </div>
  );
}
