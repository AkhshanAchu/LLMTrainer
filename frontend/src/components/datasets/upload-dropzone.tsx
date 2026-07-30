"use client";

import { useCallback, useRef, useState } from "react";
import { UploadCloud, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";

const ACCEPTED_EXTENSIONS = [".json", ".jsonl", ".csv", ".txt", ".parquet"];

export function UploadDropzone({
  onUpload,
  uploading,
}: {
  onUpload: (file: File) => void | Promise<void>;
  uploading: boolean;
}) {
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFiles = useCallback(
    (files: FileList | null) => {
      if (!files || files.length === 0) return;
      onUpload(files[0]);
    },
    [onUpload]
  );

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        if (!uploading) setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setIsDragging(false);
        if (!uploading) handleFiles(e.dataTransfer.files);
      }}
      onClick={() => !uploading && inputRef.current?.click()}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if ((e.key === "Enter" || e.key === " ") && !uploading) inputRef.current?.click();
      }}
      className={cn(
        "flex flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed border-border px-6 py-10 text-center transition-colors cursor-pointer",
        isDragging ? "border-primary bg-accent/40" : "hover:bg-accent/20",
        uploading && "pointer-events-none opacity-70"
      )}
    >
      <input
        ref={inputRef}
        type="file"
        className="hidden"
        accept={ACCEPTED_EXTENSIONS.join(",")}
        onChange={(e) => {
          handleFiles(e.target.files);
          e.target.value = "";
        }}
      />
      {uploading ? (
        <>
          <Loader2 className="size-8 text-muted-foreground animate-spin" />
          <p className="text-sm font-medium">Uploading…</p>
        </>
      ) : (
        <>
          <UploadCloud className="size-8 text-muted-foreground" />
          <p className="text-sm font-medium">Drag & drop a dataset file, or click to browse</p>
          <p className="text-xs text-muted-foreground">
            Supports {ACCEPTED_EXTENSIONS.join(", ")}
          </p>
        </>
      )}
    </div>
  );
}
