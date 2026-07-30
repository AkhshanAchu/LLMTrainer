"use client";

import { useEffect, useRef, useState } from "react";
import { Topbar } from "@/components/layout/topbar";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Separator } from "@/components/ui/separator";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import { inferenceApi, tokenizerApi } from "@/lib/api/endpoints";
import { API_BASE_URL } from "@/lib/api/client";
import { useAppStore } from "@/store/app-store";
import { toast } from "sonner";
import { Loader2, Send, Download, Bot, User, Power, PowerOff } from "lucide-react";

interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
  stats?: {
    tokens_per_sec?: number;
    time_to_first_token?: number;
    total_time?: number;
    [key: string]: number | undefined;
  };
}

export default function ChatPage() {
  const selectedModelRepoId = useAppStore((s) => s.selectedModelRepoId);

  const [repoId, setRepoId] = useState(selectedModelRepoId ?? "");
  const [loadIn4bit, setLoadIn4bit] = useState(false);
  const [loadedRepoId, setLoadedRepoId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isUnloading, setIsUnloading] = useState(false);
  const [checkingCurrent, setCheckingCurrent] = useState(true);

  const [systemPrompt, setSystemPrompt] = useState("");
  const [temperature, setTemperature] = useState(0.7);
  const [topP, setTopP] = useState(0.9);
  const [topK, setTopK] = useState(50);
  const [minP, setMinP] = useState(0.0);
  const [repetitionPenalty, setRepetitionPenalty] = useState(1.1);
  const [maxNewTokens, setMaxNewTokens] = useState(512);

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);

  const [promptPreview, setPromptPreview] = useState<{ formatted_prompt: string; token_count: number } | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);

  const bottomRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    inferenceApi
      .current()
      .then((res) => {
        if (res.repo_id) {
          setLoadedRepoId(res.repo_id);
          setRepoId(res.repo_id);
        }
      })
      .catch(() => {})
      .finally(() => setCheckingCurrent(false));
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function handleLoad() {
    if (!repoId.trim()) {
      toast.error("Enter a repo_id first.");
      return;
    }
    setIsLoading(true);
    try {
      await inferenceApi.load(repoId.trim(), undefined, loadIn4bit);
      setLoadedRepoId(repoId.trim());
      toast.success(`Model "${repoId.trim()}" loaded.`);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Failed to load model.");
    } finally {
      setIsLoading(false);
    }
  }

  async function handleUnload() {
    setIsUnloading(true);
    try {
      await inferenceApi.unload();
      setLoadedRepoId(null);
      toast.success("Model unloaded.");
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Failed to unload model.");
    } finally {
      setIsUnloading(false);
    }
  }

  function exportConversation() {
    const blob = new Blob([JSON.stringify(messages, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `conversation-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  async function refreshPromptPreview(nextMessages: ChatMessage[]) {
    if (!loadedRepoId || nextMessages.length === 0) {
      setPromptPreview(null);
      return;
    }
    setPreviewLoading(true);
    try {
      const payloadMessages = systemPrompt.trim()
        ? [{ role: "system", content: systemPrompt.trim() }, ...nextMessages.map((m) => ({ role: m.role, content: m.content }))]
        : nextMessages.map((m) => ({ role: m.role, content: m.content }));
      const res = await tokenizerApi.chatTemplate(loadedRepoId, payloadMessages, true);
      setPromptPreview(res);
    } catch (e) {
      setPromptPreview(null);
      // Non-fatal: chat template may not exist for this tokenizer.
    } finally {
      setPreviewLoading(false);
    }
  }

  async function handleSend() {
    const text = input.trim();
    if (!text || isStreaming) return;
    if (!loadedRepoId) {
      toast.error("Load a model first.");
      return;
    }

    const userMsg: ChatMessage = { role: "user", content: text };
    const nextMessages = [...messages, userMsg];
    setMessages([...nextMessages, { role: "assistant", content: "" }]);
    setInput("");
    setIsStreaming(true);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const res = await fetch(`${API_BASE_URL}/api/inference/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({
          messages: nextMessages.map((m) => ({ role: m.role, content: m.content })),
          system_prompt: systemPrompt.trim() || undefined,
          max_new_tokens: maxNewTokens,
          temperature,
          top_p: topP,
          top_k: topK,
          min_p: minP,
          repetition_penalty: repetitionPenalty,
        }),
      });

      if (!res.ok || !res.body) {
        let detail = res.statusText;
        try {
          const body = await res.json();
          detail = body.detail ?? detail;
        } catch {
          // ignore
        }
        throw new Error(detail || "Streaming request failed.");
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let assistantText = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split("\n\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith("data:")) continue;
          const jsonStr = trimmed.slice(5).trim();
          if (!jsonStr) continue;
          let payload: { token?: string; done?: boolean; stats?: Record<string, number> };
          try {
            payload = JSON.parse(jsonStr);
          } catch {
            continue;
          }
          if (payload.done) {
            setMessages((prev) => {
              const copy = [...prev];
              const last = copy[copy.length - 1];
              if (last && last.role === "assistant") {
                copy[copy.length - 1] = { ...last, stats: payload.stats };
              }
              return copy;
            });
          } else if (payload.token !== undefined) {
            assistantText += payload.token;
            const snapshot = assistantText;
            setMessages((prev) => {
              const copy = [...prev];
              const last = copy[copy.length - 1];
              if (last && last.role === "assistant") {
                copy[copy.length - 1] = { ...last, content: snapshot };
              }
              return copy;
            });
          }
        }
      }
    } catch (e) {
      if ((e as Error).name !== "AbortError") {
        toast.error(e instanceof Error ? e.message : "Chat stream failed.");
      }
    } finally {
      setIsStreaming(false);
      abortRef.current = null;
    }
  }

  const modelReady = !!loadedRepoId;

  return (
    <>
      <Topbar title="Chat Playground" />
      <main className="flex-1 overflow-hidden flex">
        <aside className="w-80 shrink-0 border-r border-border overflow-y-auto p-4 space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Model</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="space-y-1.5">
                <Label htmlFor="repo-id">Repo ID</Label>
                <Input
                  id="repo-id"
                  placeholder="e.g. gpt2"
                  value={repoId}
                  onChange={(e) => setRepoId(e.target.value)}
                  disabled={isLoading || checkingCurrent}
                />
              </div>
              <div className="flex items-center justify-between">
                <Label htmlFor="load-4bit" className="text-sm font-normal">
                  Load in 4-bit
                </Label>
                <Switch
                  id="load-4bit"
                  checked={loadIn4bit}
                  onCheckedChange={(v) => setLoadIn4bit(!!v)}
                  disabled={isLoading || modelReady}
                />
              </div>
              <div className="flex gap-2">
                <Button
                  className="flex-1"
                  onClick={handleLoad}
                  disabled={isLoading || modelReady || checkingCurrent}
                >
                  {isLoading ? <Loader2 className="size-4 animate-spin" /> : <Power className="size-4" />}
                  Load Model
                </Button>
                <Button
                  variant="outline"
                  className="flex-1"
                  onClick={handleUnload}
                  disabled={isUnloading || !modelReady}
                >
                  {isUnloading ? <Loader2 className="size-4 animate-spin" /> : <PowerOff className="size-4" />}
                  Unload
                </Button>
              </div>
              <div className="text-xs text-muted-foreground">
                {checkingCurrent
                  ? "Checking current model…"
                  : modelReady
                  ? (
                    <span className="flex items-center gap-1.5">
                      <span className="size-1.5 rounded-full bg-emerald-500" /> Loaded: {loadedRepoId}
                    </span>
                  )
                  : (
                    <span className="flex items-center gap-1.5">
                      <span className="size-1.5 rounded-full bg-slate-500" /> No model loaded
                    </span>
                  )}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Generation Parameters</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <ParamSlider label="Temperature" value={temperature} onChange={setTemperature} min={0} max={2} step={0.05} />
              <ParamSlider label="Top P" value={topP} onChange={setTopP} min={0} max={1} step={0.01} />
              <ParamSlider label="Top K" value={topK} onChange={setTopK} min={0} max={200} step={1} />
              <ParamSlider label="Min P" value={minP} onChange={setMinP} min={0} max={1} step={0.01} />
              <ParamSlider
                label="Repetition Penalty"
                value={repetitionPenalty}
                onChange={setRepetitionPenalty}
                min={1}
                max={2}
                step={0.01}
              />
              <ParamSlider
                label="Max New Tokens"
                value={maxNewTokens}
                onChange={setMaxNewTokens}
                min={16}
                max={2048}
                step={16}
              />
              <Separator />
              <div className="space-y-1.5">
                <Label htmlFor="system-prompt">System Prompt</Label>
                <Textarea
                  id="system-prompt"
                  placeholder="You are a helpful assistant…"
                  value={systemPrompt}
                  onChange={(e) => setSystemPrompt(e.target.value)}
                  className="min-h-20"
                />
              </div>
            </CardContent>
          </Card>
        </aside>

        <div className="flex-1 flex flex-col min-w-0">
          <div className="flex items-center justify-between px-4 py-2 border-b border-border">
            <div className="text-xs text-muted-foreground">
              {messages.length === 0 ? "No messages yet" : `${messages.length} message${messages.length === 1 ? "" : "s"}`}
            </div>
            <div className="flex gap-2">
              <Button variant="outline" size="sm" onClick={() => refreshPromptPreview(messages.filter((m) => m.content))}>
                Preview Prompt
              </Button>
              <Button variant="outline" size="sm" onClick={exportConversation} disabled={messages.length === 0}>
                <Download className="size-4" />
                Export
              </Button>
            </div>
          </div>

          {promptPreview !== null || previewLoading ? (
            <Accordion className="border-b border-border px-4">
              <AccordionItem value="prompt-inspector">
                <AccordionTrigger>
                  Prompt Inspector {promptPreview && <Badge variant="outline" className="ml-2">{promptPreview.token_count} tokens</Badge>}
                </AccordionTrigger>
                <AccordionContent>
                  {previewLoading ? (
                    <p className="text-sm text-muted-foreground">Formatting prompt…</p>
                  ) : promptPreview ? (
                    <pre className="text-xs bg-muted/40 border border-border rounded-lg p-3 overflow-x-auto whitespace-pre-wrap">
                      {promptPreview.formatted_prompt}
                    </pre>
                  ) : (
                    <p className="text-sm text-muted-foreground">
                      Could not build a preview (tokenizer may lack a chat template).
                    </p>
                  )}
                </AccordionContent>
              </AccordionItem>
            </Accordion>
          ) : null}

          <ScrollArea className="flex-1">
            <div className="p-4 space-y-4 max-w-3xl mx-auto">
              {messages.length === 0 && (
                <div className="text-center text-sm text-muted-foreground py-20">
                  {modelReady ? "Send a message to start chatting." : "Load a model to begin."}
                </div>
              )}
              {messages.map((m, i) => (
                <MessageBubble key={i} message={m} />
              ))}
              <div ref={bottomRef} />
            </div>
          </ScrollArea>

          <div className="border-t border-border p-4">
            <div className="max-w-3xl mx-auto flex gap-2">
              <Textarea
                placeholder={modelReady ? "Type a message…" : "Load a model first…"}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    handleSend();
                  }
                }}
                disabled={!modelReady || isStreaming}
                className="min-h-11 max-h-40"
              />
              <Button onClick={handleSend} disabled={!modelReady || isStreaming || !input.trim()}>
                {isStreaming ? <Loader2 className="size-4 animate-spin" /> : <Send className="size-4" />}
              </Button>
            </div>
          </div>
        </div>
      </main>
    </>
  );
}

function ParamSlider({
  label,
  value,
  onChange,
  min,
  max,
  step,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step: number;
}) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between text-xs">
        <Label className="font-normal">{label}</Label>
        <span className="text-muted-foreground font-mono">{value}</span>
      </div>
      <Slider
        value={value}
        onValueChange={(v) => onChange(Array.isArray(v) ? v[0] : v)}
        min={min}
        max={max}
        step={step}
      />
    </div>
  );
}

function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";
  return (
    <div className={`flex gap-3 ${isUser ? "justify-end" : "justify-start"}`}>
      {!isUser && (
        <div className="size-7 rounded-full bg-primary/15 text-primary flex items-center justify-center shrink-0">
          <Bot className="size-4" />
        </div>
      )}
      <div className={`max-w-[75%] ${isUser ? "order-1" : ""}`}>
        <div
          className={`rounded-2xl px-4 py-2.5 text-sm whitespace-pre-wrap break-words ${
            isUser ? "bg-primary text-primary-foreground" : "bg-card border border-border"
          }`}
        >
          {message.content || <span className="opacity-50">…</span>}
        </div>
        {message.stats && (
          <p className="text-[11px] text-muted-foreground mt-1 px-1">
            {message.stats.tokens_per_sec !== undefined && `${message.stats.tokens_per_sec.toFixed(1)} tok/s`}
            {message.stats.time_to_first_token !== undefined && ` · TTFT ${message.stats.time_to_first_token.toFixed(2)}s`}
            {message.stats.total_time !== undefined && ` · ${message.stats.total_time.toFixed(2)}s total`}
          </p>
        )}
      </div>
      {isUser && (
        <div className="size-7 rounded-full bg-accent flex items-center justify-center shrink-0">
          <User className="size-4" />
        </div>
      )}
    </div>
  );
}
