import { Badge } from "@/components/ui/badge";

export function QualityBadge({ score }: { score: number | null | undefined }) {
  if (score === null || score === undefined || Number.isNaN(score)) {
    return (
      <Badge variant="outline" className="bg-slate-500/15 text-slate-400 border-slate-500/30">
        n/a
      </Badge>
    );
  }
  let cls = "bg-emerald-500/15 text-emerald-400 border-emerald-500/30";
  if (score < 50) cls = "bg-red-500/15 text-red-400 border-red-500/30";
  else if (score < 80) cls = "bg-amber-500/15 text-amber-400 border-amber-500/30";
  return (
    <Badge variant="outline" className={cls}>
      {score.toFixed(0)}
    </Badge>
  );
}
