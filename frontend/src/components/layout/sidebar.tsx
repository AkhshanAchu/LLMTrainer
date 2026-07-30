"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  Boxes,
  Database,
  Cpu,
  MessageSquare,
  Braces,
  Activity,
} from "lucide-react";
import { cn } from "@/lib/utils";

const NAV_ITEMS = [
  { href: "/", label: "Dashboard", icon: LayoutDashboard },
  { href: "/models", label: "Model Manager", icon: Boxes },
  { href: "/datasets", label: "Dataset Manager", icon: Database },
  { href: "/training", label: "Training", icon: Cpu },
  { href: "/chat", label: "Chat Playground", icon: MessageSquare },
  { href: "/tokenizer", label: "Tokenizer Viewer", icon: Braces },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="hidden md:flex w-64 shrink-0 flex-col border-r border-sidebar-border bg-sidebar/60 backdrop-blur-xl">
      <div className="flex items-center gap-2 px-5 h-16 border-b border-sidebar-border">
        <div className="size-7 rounded-md bg-gradient-to-br from-violet-500 to-cyan-400 shadow-lg shadow-violet-500/20" />
        <span className="font-semibold tracking-tight text-sidebar-foreground">FineTune Studio</span>
      </div>
      <nav className="flex-1 overflow-y-auto p-3 space-y-1">
        {NAV_ITEMS.map(({ href, label, icon: Icon }) => {
          const active = href === "/" ? pathname === "/" : pathname?.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              className={cn(
                "flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors",
                active
                  ? "bg-sidebar-accent text-sidebar-accent-foreground font-medium"
                  : "text-sidebar-foreground/70 hover:bg-sidebar-accent/50 hover:text-sidebar-foreground"
              )}
            >
              <Icon className="size-4" />
              {label}
            </Link>
          );
        })}
      </nav>
      <div className="p-3 border-t border-sidebar-border">
        <div className="flex items-center gap-2 px-3 py-2 text-xs text-sidebar-foreground/60">
          <Activity className="size-3.5" />
          <span>Local GPU connected</span>
        </div>
      </div>
    </aside>
  );
}
