"use client";

import { useCallback, useEffect, useState } from "react";
import { toast } from "sonner";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import { datasetsApi } from "@/lib/api/endpoints";
import { ApiError } from "@/lib/api/client";
import { cn } from "@/lib/utils";

type ChatMessage = { role?: string; content?: string };

const ROLE_STYLES: Record<string, string> = {
  system: "bg-slate-500/15 border-slate-500/30 text-slate-300",
  user: "bg-blue-500/15 border-blue-500/30 text-blue-300",
  assistant: "bg-emerald-500/15 border-emerald-500/30 text-emerald-300",
  tool: "bg-purple-500/15 border-purple-500/30 text-purple-300",
};

function isChatMessageArray(rec: unknown): rec is ChatMessage[] {
  return Array.isArray(rec) && rec.every((m) => m && typeof m === "object" && "role" in m);
}

function RawRecordCard({ record, index }: { record: unknown; index: number }) {
  return (
    <div className="rounded-lg border border-border bg-card p-3">
      <p className="text-xs text-muted-foreground mb-1.5">Example #{index}</p>
      <pre className="text-xs whitespace-pre-wrap break-words font-mono overflow-x-auto">
        {JSON.stringify(record, null, 2)}
      </pre>
    </div>
  );
}

function ChatRecordCard({ messages, index }: { messages: ChatMessage[]; index: number }) {
  return (
    <div className="rounded-lg border border-border bg-card p-3 space-y-2">
      <p className="text-xs text-muted-foreground">Example #{index}</p>
      {messages.map((m, i) => (
        <div
          key={i}
          className={cn(
            "rounded-md border px-2.5 py-1.5 text-xs",
            ROLE_STYLES[m.role ?? ""] ?? "bg-muted border-border text-foreground"
          )}
        >
          <p className="font-semibold uppercase tracking-wide text-[10px] mb-0.5 opacity-80">
            {m.role ?? "unknown"}
          </p>
          <p className="whitespace-pre-wrap break-words font-sans text-foreground/90">
            {m.content ?? ""}
          </p>
        </div>
      ))}
    </div>
  );
}

export function PreviewSection({ datasetId }: { datasetId: number }) {
  const [format, setFormat] = useState<"raw" | "chat">("chat");
  const [records, setRecords] = useState<unknown[] | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(
    async (fmt: "raw" | "chat") => {
      setLoading(true);
      try {
        // NOTE: the backend actually returns { format, records } for this endpoint,
        // though endpoints.ts types the response loosely as Record<string, unknown>[].
        const data = (await datasetsApi.preview(datasetId, 20, fmt)) as unknown as {
          format: string;
          records: unknown[];
        };
        setRecords(Array.isArray(data.records) ? data.records : []);
      } catch (err) {
        toast.error(err instanceof ApiError ? err.message : "Failed to load preview");
        setRecords([]);
      } finally {
        setLoading(false);
      }
    },
    [datasetId]
  );

  useEffect(() => {
    load(format);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [datasetId]);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Preview</CardTitle>
      </CardHeader>
      <CardContent>
        <Tabs
          value={format}
          onValueChange={(v) => {
            const fmt = v as "raw" | "chat";
            setFormat(fmt);
            load(fmt);
          }}
        >
          <TabsList>
            <TabsTrigger value="chat">Chat-formatted</TabsTrigger>
            <TabsTrigger value="raw">Raw</TabsTrigger>
          </TabsList>
          <TabsContent value={format} className="mt-3">
            {loading ? (
              <div className="space-y-2">
                <Skeleton className="h-20 w-full" />
                <Skeleton className="h-20 w-full" />
              </div>
            ) : !records || records.length === 0 ? (
              <p className="text-sm text-muted-foreground py-6 text-center">No preview records available.</p>
            ) : (
              <ScrollArea className="h-[420px] pr-3">
                <div className="space-y-3">
                  {records.map((rec, i) =>
                    format === "chat" && isChatMessageArray(rec) ? (
                      <ChatRecordCard key={i} messages={rec} index={i} />
                    ) : (
                      <RawRecordCard key={i} record={rec} index={i} />
                    )
                  )}
                </div>
              </ScrollArea>
            )}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
