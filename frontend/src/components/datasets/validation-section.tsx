"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import { CheckCircle2, TriangleAlert } from "lucide-react";

const CATEGORY_LABELS: Record<string, string> = {
  missing_fields: "Missing fields",
  role_mismatch: "Role mismatch",
  invalid_conversation: "Invalid conversation",
  duplicate_samples: "Duplicate samples",
  empty_responses: "Empty responses",
  broken_utf8: "Broken UTF-8",
  unsupported_tokens: "Unsupported tokens / control chars",
};

const CATEGORY_ORDER = [
  "missing_fields",
  "role_mismatch",
  "invalid_conversation",
  "duplicate_samples",
  "empty_responses",
  "broken_utf8",
  "unsupported_tokens",
];

function asRecord(v: unknown): Record<string, unknown> | undefined {
  return v && typeof v === "object" ? (v as Record<string, unknown>) : undefined;
}

export function ValidationSection({ report }: { report: Record<string, unknown> | undefined }) {
  if (!report) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Validation Report</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">No validation report available for this dataset.</p>
        </CardContent>
      </Card>
    );
  }

  const issuesFound = typeof report.issues_found === "number" ? report.issues_found : undefined;
  const numChecked = typeof report.num_examples_checked === "number" ? report.num_examples_checked : undefined;

  const categories = CATEGORY_ORDER.map((cat) => {
    const entry = asRecord(report[cat]);
    const count = typeof entry?.count === "number" ? entry.count : 0;
    const sampleIndices = Array.isArray(entry?.sample_indices) ? (entry!.sample_indices as unknown[]) : [];
    return { key: cat, label: CATEGORY_LABELS[cat] ?? cat, count, sampleIndices };
  }).filter((c) => c.count > 0 || report[c.key] !== undefined);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Validation Report</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {issuesFound !== undefined && (
          issuesFound > 0 ? (
            <Alert variant="destructive">
              <TriangleAlert />
              <AlertTitle>{issuesFound} issue{issuesFound === 1 ? "" : "s"} found</AlertTitle>
              <AlertDescription>
                Checked {numChecked ?? "?"} examples. Consider running auto-repair below.
              </AlertDescription>
            </Alert>
          ) : (
            <Alert className="border-emerald-500/30 bg-emerald-500/10 text-emerald-400 [&_svg]:text-emerald-400">
              <CheckCircle2 />
              <AlertTitle>No issues found</AlertTitle>
              <AlertDescription className="text-emerald-400/80">
                Checked {numChecked ?? "?"} examples — dataset looks clean.
              </AlertDescription>
            </Alert>
          )
        )}

        {categories.length === 0 ? (
          <p className="text-sm text-muted-foreground">No category breakdown available.</p>
        ) : (
          <Accordion>
            {categories.map((cat) => (
              <AccordionItem key={cat.key} value={cat.key}>
                <AccordionTrigger>
                  <span className="flex items-center gap-2">
                    {cat.label}
                    <Badge
                      variant="outline"
                      className={
                        cat.count > 0
                          ? "bg-amber-500/15 text-amber-400 border-amber-500/30"
                          : "bg-emerald-500/15 text-emerald-400 border-emerald-500/30"
                      }
                    >
                      {cat.count}
                    </Badge>
                  </span>
                </AccordionTrigger>
                <AccordionContent>
                  {cat.sampleIndices.length > 0 ? (
                    <div className="flex flex-wrap gap-1.5">
                      <span className="text-xs text-muted-foreground mr-1">Sample indices:</span>
                      {cat.sampleIndices.map((idx, i) => (
                        <span
                          key={i}
                          className="rounded-md border border-border bg-accent/30 px-1.5 py-0.5 text-xs font-mono"
                        >
                          {String(idx)}
                        </span>
                      ))}
                    </div>
                  ) : (
                    <p className="text-xs text-muted-foreground">No problem samples in this category.</p>
                  )}
                </AccordionContent>
              </AccordionItem>
            ))}
          </Accordion>
        )}
      </CardContent>
    </Card>
  );
}
