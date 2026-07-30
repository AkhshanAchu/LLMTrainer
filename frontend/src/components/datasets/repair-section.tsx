"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { toast } from "sonner";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
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
import { Loader2, Wrench } from "lucide-react";
import type { Dataset } from "@/lib/api/types";

interface RepairOptions {
  fix_roles: boolean;
  merge_consecutive_same_role: boolean;
  remove_duplicates: boolean;
  remove_empty_responses: boolean;
  shuffle: boolean;
  max_turn_chars: number | null;
}

const DEFAULT_OPTIONS: RepairOptions = {
  fix_roles: true,
  merge_consecutive_same_role: true,
  remove_duplicates: true,
  remove_empty_responses: true,
  shuffle: false,
  max_turn_chars: null,
};

export function RepairSection({ datasetId }: { datasetId: number }) {
  const router = useRouter();
  const [options, setOptions] = useState<RepairOptions>(DEFAULT_OPTIONS);
  const [trimEnabled, setTrimEnabled] = useState(false);
  const [trimChars, setTrimChars] = useState(4000);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<Dataset | null>(null);

  const toggle = (key: keyof RepairOptions) => (checked: boolean) =>
    setOptions((prev) => ({ ...prev, [key]: checked }));

  const runRepair = async () => {
    setRunning(true);
    try {
      const payload: Record<string, unknown> = {
        fix_roles: options.fix_roles,
        merge_consecutive_same_role: options.merge_consecutive_same_role,
        remove_duplicates: options.remove_duplicates,
        remove_empty_responses: options.remove_empty_responses,
        shuffle: options.shuffle,
      };
      if (trimEnabled) payload.max_turn_chars = trimChars;

      // NOTE: the backend's /repair endpoint actually returns the full new Dataset
      // (DatasetDetail), even though the endpoints.ts contract types it loosely.
      const newDs = (await datasetsApi.repair(datasetId, payload)) as unknown as Dataset;
      setResult(newDs);
      toast.success(`Repair complete — created "${newDs.name}"`);
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Repair failed");
    } finally {
      setRunning(false);
    }
  };

  const summary = (result?.stats as Record<string, unknown> | undefined)?.repair_summary as
    | Record<string, unknown>
    | undefined;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base flex items-center gap-2">
          <Wrench className="size-4" />
          Auto-Repair
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <CheckboxRow
            id="fix_roles"
            label="Fix roles"
            hint="Drop leading turns that aren't system/user"
            checked={options.fix_roles}
            onCheckedChange={toggle("fix_roles")}
          />
          <CheckboxRow
            id="merge_consecutive_same_role"
            label="Merge messages"
            hint="Merge consecutive same-role turns"
            checked={options.merge_consecutive_same_role}
            onCheckedChange={toggle("merge_consecutive_same_role")}
          />
          <CheckboxRow
            id="remove_duplicates"
            label="Dedupe"
            hint="Remove duplicate samples"
            checked={options.remove_duplicates}
            onCheckedChange={toggle("remove_duplicates")}
          />
          <CheckboxRow
            id="remove_empty_responses"
            label="Remove empty responses"
            hint="Drop examples with blank assistant turns"
            checked={options.remove_empty_responses}
            onCheckedChange={toggle("remove_empty_responses")}
          />
          <CheckboxRow
            id="shuffle"
            label="Shuffle"
            hint="Shuffle example order (seed 42)"
            checked={options.shuffle}
            onCheckedChange={toggle("shuffle")}
          />
          <div className="flex items-start gap-2">
            <Checkbox
              id="trim_long"
              checked={trimEnabled}
              onCheckedChange={(c) => setTrimEnabled(c === true)}
            />
            <div className="flex-1">
              <Label htmlFor="trim_long">Trim long samples</Label>
              <p className="text-xs text-muted-foreground">Truncate each turn to N characters</p>
              {trimEnabled && (
                <Input
                  type="number"
                  min={1}
                  value={trimChars}
                  onChange={(e) => setTrimChars(Number(e.target.value) || 0)}
                  className="w-32 mt-1.5"
                />
              )}
            </div>
          </div>
        </div>

        <Button onClick={runRepair} disabled={running}>
          {running && <Loader2 className="size-3.5 animate-spin" />}
          Run Auto-Repair
        </Button>
      </CardContent>

      <Dialog open={!!result} onOpenChange={(open) => !open && setResult(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Repair complete</DialogTitle>
            <DialogDescription>
              A new repaired dataset &quot;{result?.name}&quot; was created with {result?.num_examples}{" "}
              examples.
            </DialogDescription>
          </DialogHeader>
          {summary && (
            <div className="grid grid-cols-2 gap-2 text-xs">
              {Object.entries(summary).map(([k, v]) => (
                <div key={k} className="flex justify-between rounded-md border border-border px-2 py-1">
                  <span className="text-muted-foreground">{k}</span>
                  <span className="font-mono">{String(v)}</span>
                </div>
              ))}
            </div>
          )}
          <DialogFooter>
            <Button variant="outline" onClick={() => setResult(null)}>
              Close
            </Button>
            {result && (
              <Button
                onClick={() => {
                  router.push(`/datasets/${result.id}`);
                }}
              >
                Open repaired dataset
              </Button>
            )}
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Card>
  );
}

function CheckboxRow({
  id,
  label,
  hint,
  checked,
  onCheckedChange,
}: {
  id: string;
  label: string;
  hint: string;
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
}) {
  return (
    <div className="flex items-start gap-2">
      <Checkbox id={id} checked={checked} onCheckedChange={(c) => onCheckedChange(c === true)} />
      <div>
        <Label htmlFor={id}>{label}</Label>
        <p className="text-xs text-muted-foreground">{hint}</p>
      </div>
    </div>
  );
}
