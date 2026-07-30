import { Badge } from "@/components/ui/badge";
import type { DatasetFormat } from "@/lib/api/types";

const FORMAT_STYLES: Record<string, string> = {
  alpaca: "bg-blue-500/15 text-blue-400 border-blue-500/30",
  sharegpt: "bg-purple-500/15 text-purple-400 border-purple-500/30",
  chatml: "bg-emerald-500/15 text-emerald-400 border-emerald-500/30",
  openai: "bg-teal-500/15 text-teal-400 border-teal-500/30",
  llama: "bg-orange-500/15 text-orange-400 border-orange-500/30",
  preference: "bg-pink-500/15 text-pink-400 border-pink-500/30",
  raw_text: "bg-slate-500/15 text-slate-400 border-slate-500/30",
  unknown: "bg-red-500/15 text-red-400 border-red-500/30",
};

export function FormatBadge({ format }: { format: DatasetFormat | string | undefined | null }) {
  const key = format ?? "unknown";
  return (
    <Badge variant="outline" className={FORMAT_STYLES[key] ?? FORMAT_STYLES.unknown}>
      {key}
    </Badge>
  );
}
