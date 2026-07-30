"use client";

import { useState } from "react";
import { Topbar } from "@/components/layout/topbar";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { tokenizerApi } from "@/lib/api/endpoints";
import { toast } from "sonner";
import { Loader2, Plus, Trash2, Wand2 } from "lucide-react";

interface EncodeResult {
  token_count: number;
  tokens: { id: number; text: string; is_special: boolean }[];
  special_tokens: Record<string, string | null>;
  vocab_size: number;
}

interface ChatTemplateResult {
  formatted_prompt: string;
  token_count: number;
  chat_template_raw: string;
}

interface MessageRow {
  role: "system" | "user" | "assistant";
  content: string;
}

const TOKEN_PALETTE = [
  "bg-blue-500/20 text-blue-300 border-blue-500/30",
  "bg-emerald-500/20 text-emerald-300 border-emerald-500/30",
  "bg-amber-500/20 text-amber-300 border-amber-500/30",
  "bg-violet-500/20 text-violet-300 border-violet-500/30",
  "bg-pink-500/20 text-pink-300 border-pink-500/30",
  "bg-cyan-500/20 text-cyan-300 border-cyan-500/30",
];

export default function TokenizerPage() {
  const [repoId, setRepoId] = useState("gpt2");
  const [text, setText] = useState("The quick brown fox jumps over the lazy dog.");
  const [encodeResult, setEncodeResult] = useState<EncodeResult | null>(null);
  const [encoding, setEncoding] = useState(false);

  const [chatRepoId, setChatRepoId] = useState("gpt2");
  const [messages, setMessages] = useState<MessageRow[]>([
    { role: "system", content: "You are a helpful assistant." },
    { role: "user", content: "Hello, who are you?" },
  ]);
  const [chatResult, setChatResult] = useState<ChatTemplateResult | null>(null);
  const [templating, setTemplating] = useState(false);

  async function handleEncode() {
    if (!repoId.trim()) {
      toast.error("Enter a repo_id first.");
      return;
    }
    if (!text.trim()) {
      toast.error("Enter some text to encode.");
      return;
    }
    setEncoding(true);
    try {
      const res = await tokenizerApi.encode(repoId.trim(), text);
      setEncodeResult(res);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Failed to encode text.");
      setEncodeResult(null);
    } finally {
      setEncoding(false);
    }
  }

  async function handleChatTemplate() {
    if (!chatRepoId.trim()) {
      toast.error("Enter a repo_id first.");
      return;
    }
    if (messages.some((m) => !m.content.trim())) {
      toast.error("Fill in all message rows (or remove empty ones).");
      return;
    }
    setTemplating(true);
    try {
      const res = await tokenizerApi.chatTemplate(
        chatRepoId.trim(),
        messages.map((m) => ({ role: m.role, content: m.content })),
        true
      );
      setChatResult(res);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Failed to build chat template.");
      setChatResult(null);
    } finally {
      setTemplating(false);
    }
  }

  function addMessageRow() {
    setMessages((prev) => [...prev, { role: "user", content: "" }]);
  }

  function removeMessageRow(index: number) {
    setMessages((prev) => prev.filter((_, i) => i !== index));
  }

  function updateMessageRow(index: number, patch: Partial<MessageRow>) {
    setMessages((prev) => prev.map((m, i) => (i === index ? { ...m, ...patch } : m)));
  }

  return (
    <>
      <Topbar title="Tokenizer Viewer" />
      <main className="flex-1 overflow-y-auto p-6 space-y-6 max-w-5xl mx-auto w-full">
        {/* Encode section */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Encode Text</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex gap-2">
              <div className="flex-1 space-y-1.5">
                <Label htmlFor="tok-repo-id">Repo ID</Label>
                <Input id="tok-repo-id" value={repoId} onChange={(e) => setRepoId(e.target.value)} placeholder="gpt2" />
              </div>
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="tok-text">Text</Label>
              <Textarea
                id="tok-text"
                value={text}
                onChange={(e) => setText(e.target.value)}
                className="min-h-28 font-mono text-sm"
                placeholder="Type or paste text to tokenize…"
              />
            </div>
            <Button onClick={handleEncode} disabled={encoding}>
              {encoding ? <Loader2 className="size-4 animate-spin" /> : <Wand2 className="size-4" />}
              Encode
            </Button>

            {encoding && (
              <div className="space-y-2">
                <Skeleton className="h-6 w-full" />
                <Skeleton className="h-20 w-full" />
              </div>
            )}

            {encodeResult && !encoding && (
              <div className="space-y-4">
                <Separator />
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                  <InfoTile label="Token Count" value={String(encodeResult.token_count)} />
                  <InfoTile label="Vocab Size" value={encodeResult.vocab_size.toLocaleString()} />
                  <InfoTile label="BOS" value={encodeResult.special_tokens.bos ?? "—"} mono />
                  <InfoTile label="EOS" value={encodeResult.special_tokens.eos ?? "—"} mono />
                  <InfoTile label="PAD" value={encodeResult.special_tokens.pad ?? "—"} mono />
                  <InfoTile label="UNK" value={encodeResult.special_tokens.unk ?? "—"} mono />
                </div>

                <div className="space-y-1.5">
                  <Label className="text-xs text-muted-foreground">Tokens</Label>
                  <div className="flex flex-wrap gap-1 p-3 rounded-lg border border-border bg-muted/20">
                    {encodeResult.tokens.map((tok, i) => (
                      <TokenChip key={i} token={tok} colorIndex={i % TOKEN_PALETTE.length} />
                    ))}
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Chat template section */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Chat Template Preview</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-1.5 max-w-sm">
              <Label htmlFor="chat-repo-id">Repo ID</Label>
              <Input
                id="chat-repo-id"
                value={chatRepoId}
                onChange={(e) => setChatRepoId(e.target.value)}
                placeholder="e.g. Qwen/Qwen2.5-0.5B-Instruct"
              />
            </div>

            <div className="space-y-3">
              {messages.map((m, i) => (
                <div key={i} className="flex gap-2 items-start">
                  <Select value={m.role} onValueChange={(v) => updateMessageRow(i, { role: v as MessageRow["role"] })}>
                    <SelectTrigger className="w-32 shrink-0">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="system">system</SelectItem>
                      <SelectItem value="user">user</SelectItem>
                      <SelectItem value="assistant">assistant</SelectItem>
                    </SelectContent>
                  </Select>
                  <Textarea
                    value={m.content}
                    onChange={(e) => updateMessageRow(i, { content: e.target.value })}
                    placeholder="Message content…"
                    className="min-h-10 flex-1"
                  />
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => removeMessageRow(i)}
                    disabled={messages.length <= 1}
                  >
                    <Trash2 className="size-4" />
                  </Button>
                </div>
              ))}
            </div>

            <div className="flex gap-2">
              <Button variant="outline" size="sm" onClick={addMessageRow}>
                <Plus className="size-4" />
                Add Message
              </Button>
              <Button size="sm" onClick={handleChatTemplate} disabled={templating}>
                {templating ? <Loader2 className="size-4 animate-spin" /> : <Wand2 className="size-4" />}
                Build Template
              </Button>
            </div>

            {templating && <Skeleton className="h-32 w-full" />}

            {chatResult && !templating && (
              <div className="space-y-2">
                <Separator />
                <div className="flex items-center justify-between">
                  <Label className="text-xs text-muted-foreground">Formatted Prompt</Label>
                  <Badge variant="outline">{chatResult.token_count} tokens</Badge>
                </div>
                <pre className="text-xs bg-muted/40 border border-border rounded-lg p-3 overflow-x-auto whitespace-pre-wrap font-mono">
                  {chatResult.formatted_prompt}
                </pre>
              </div>
            )}
          </CardContent>
        </Card>
      </main>
    </>
  );
}

function InfoTile({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="rounded-lg border border-border bg-card p-3">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className={`text-sm font-medium mt-0.5 truncate ${mono ? "font-mono" : ""}`}>{value}</p>
    </div>
  );
}

function TokenChip({
  token,
  colorIndex,
}: {
  token: { id: number; text: string; is_special: boolean };
  colorIndex: number;
}) {
  const display = token.text.replace(/\n/g, "\\n").replace(/ /g, "\u00b7");
  return (
    <Tooltip>
      <TooltipTrigger
        render={
          <span
            className={`inline-flex items-center rounded border px-1.5 py-0.5 text-xs font-mono cursor-default ${
              token.is_special
                ? "bg-red-500/20 text-red-300 border-red-500/40 font-semibold"
                : TOKEN_PALETTE[colorIndex]
            }`}
          >
            {display || "\u2205"}
          </span>
        }
      />
      <TooltipContent>
        id: {token.id}
        {token.is_special ? " · special" : ""}
      </TooltipContent>
    </Tooltip>
  );
}
