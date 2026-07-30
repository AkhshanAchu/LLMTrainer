"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Bar, BarChart, CartesianGrid, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

function asRecord(v: unknown): Record<string, unknown> | undefined {
  return v && typeof v === "object" ? (v as Record<string, unknown>) : undefined;
}
function asNumber(v: unknown): number | undefined {
  return typeof v === "number" && !Number.isNaN(v) ? v : undefined;
}

function bucketsToChartData(buckets: unknown): { bucket: string; count: number }[] {
  const rec = asRecord(buckets);
  if (!rec) return [];
  return Object.entries(rec).map(([bucket, count]) => ({
    bucket,
    count: typeof count === "number" ? count : 0,
  }));
}

const CHART_COLOR = "var(--color-chart-2)";

function HistogramChart({ data, xLabel }: { data: { bucket: string; count: number }[]; xLabel: string }) {
  if (data.length === 0) {
    return <p className="text-sm text-muted-foreground">No histogram data available.</p>;
  }
  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} margin={{ top: 8, right: 8, left: -20, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" vertical={false} />
        <XAxis
          dataKey="bucket"
          tick={{ fill: "var(--color-muted-foreground)", fontSize: 11 }}
          axisLine={{ stroke: "var(--color-border)" }}
          tickLine={false}
          label={{ value: xLabel, position: "insideBottom", offset: -2, fill: "var(--color-muted-foreground)", fontSize: 11 }}
        />
        <YAxis
          tick={{ fill: "var(--color-muted-foreground)", fontSize: 11 }}
          axisLine={false}
          tickLine={false}
          allowDecimals={false}
        />
        <Tooltip
          cursor={{ fill: "var(--color-accent)" }}
          contentStyle={{
            background: "var(--color-popover)",
            border: "1px solid var(--color-border)",
            borderRadius: 8,
            fontSize: 12,
            color: "var(--color-popover-foreground)",
          }}
        />
        <Bar dataKey="count" fill={CHART_COLOR} radius={[4, 4, 0, 0]} maxBarSize={48} />
      </BarChart>
    </ResponsiveContainer>
  );
}

function GaugeStat({ label, value, max = 100 }: { label: string; value: number | undefined; max?: number }) {
  const pct = value !== undefined ? Math.max(0, Math.min(100, (value / max) * 100)) : 0;
  return (
    <div className="space-y-1.5">
      <div className="flex items-baseline justify-between">
        <span className="text-xs text-muted-foreground">{label}</span>
        <span className="text-lg font-semibold tabular-nums">
          {value !== undefined ? value.toFixed(2) : "—"}
        </span>
      </div>
      <div className="h-1.5 w-full rounded-full bg-muted overflow-hidden">
        <div
          className="h-full rounded-full bg-primary transition-all"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

export function StatsSection({ stats }: { stats: Record<string, unknown> | undefined }) {
  if (!stats) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Statistics</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">No statistics available for this dataset.</p>
        </CardContent>
      </Card>
    );
  }

  const turns = asRecord(stats.conversation_turns);
  const tokenHist = asRecord(stats.token_histogram);
  const turnHist = asRecord(stats.conversation_length_histogram);
  const language = asRecord(stats.language);
  const languageDist = asRecord(language?.distribution);
  const roleCounts = asRecord(stats.role_counts);
  const qualityScore = asNumber(stats.quality_score);
  const entropyScore = asNumber(stats.entropy_score);
  const assistantUserRatio = asNumber(stats.assistant_user_ratio);
  const duplicateRatio = asNumber(stats.duplicate_ratio);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Quality & Diversity</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <GaugeStat label="Quality score" value={qualityScore} />
          <GaugeStat label="Entropy score" value={entropyScore !== undefined ? entropyScore * 100 : undefined} />
          <div className="grid grid-cols-2 gap-4 pt-2">
            <div>
              <p className="text-xs text-muted-foreground">Assistant / User ratio</p>
              <p className="text-lg font-semibold tabular-nums">
                {assistantUserRatio !== undefined ? assistantUserRatio.toFixed(3) : "—"}
              </p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground">Duplicate ratio</p>
              <p className="text-lg font-semibold tabular-nums">
                {duplicateRatio !== undefined ? `${(duplicateRatio * 100).toFixed(1)}%` : "—"}
              </p>
            </div>
            {turns && (
              <>
                <div>
                  <p className="text-xs text-muted-foreground">Avg turns / example</p>
                  <p className="text-lg font-semibold tabular-nums">{String(turns.avg ?? "—")}</p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">Turns min / max</p>
                  <p className="text-lg font-semibold tabular-nums">
                    {String(turns.min ?? "—")} / {String(turns.max ?? "—")}
                  </p>
                </div>
              </>
            )}
          </div>
          {roleCounts && (
            <div className="pt-2">
              <p className="text-xs text-muted-foreground mb-1.5">Role counts</p>
              <div className="flex flex-wrap gap-2">
                {Object.entries(roleCounts).map(([role, count]) => (
                  <span
                    key={role}
                    className="rounded-md border border-border bg-accent/30 px-2 py-1 text-xs"
                  >
                    {role}: {String(count)}
                  </span>
                ))}
              </div>
            </div>
          )}
          {languageDist && Object.keys(languageDist).length > 0 && (
            <div className="pt-2">
              <p className="text-xs text-muted-foreground mb-1.5">
                Language distribution {language?.sampled ? `(sampled ${String(language.sampled)})` : ""}
              </p>
              <div className="flex flex-wrap gap-2">
                {Object.entries(languageDist).map(([lang, count]) => (
                  <span
                    key={lang}
                    className="rounded-md border border-border bg-accent/30 px-2 py-1 text-xs"
                  >
                    {lang}: {String(count)}
                  </span>
                ))}
              </div>
            </div>
          )}
          {language?.available === false && (
            <p className="text-xs text-muted-foreground pt-1">{String(language.note ?? "Language detection unavailable.")}</p>
          )}
        </CardContent>
      </Card>

      <div className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Token Histogram (approx.)</CardTitle>
          </CardHeader>
          <CardContent>
            <HistogramChart data={bucketsToChartData(tokenHist?.buckets)} xLabel="tokens (word-approx)" />
            {tokenHist && (
              <p className="text-xs text-muted-foreground mt-1">
                avg {String(tokenHist.avg ?? "—")} · min {String(tokenHist.min ?? "—")} · max{" "}
                {String(tokenHist.max ?? "—")}
              </p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Conversation Length Histogram</CardTitle>
          </CardHeader>
          <CardContent>
            <HistogramChart data={bucketsToChartData(turnHist?.buckets)} xLabel="turns" />
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
