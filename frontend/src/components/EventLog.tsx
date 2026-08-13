import { useEffect, useRef } from "react";
import type { RunEvent } from "../lib/types";

const KIND_STYLE: Record<RunEvent["kind"], string> = {
  reasoning: "text-violet-300",
  info: "text-zinc-300",
  log: "text-zinc-500",
  warning: "text-amber-400",
  error: "text-red-400",
  state_change: "text-emerald-400 font-medium",
  metric: "text-sky-400",
};

const KIND_PREFIX: Record<RunEvent["kind"], string> = {
  reasoning: "🧠",
  info: "•",
  log: "  ",
  warning: "⚠",
  error: "✕",
  state_change: "▸",
  metric: "▲",
};

export default function EventLog({ events }: { events: RunEvent[] }) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ block: "end" });
  }, [events.length]);

  return (
    <div className="mono flex-1 overflow-y-auto rounded-xl border border-zinc-800 bg-black/40 p-3 text-[12.5px] leading-relaxed">
      {events.length === 0 && <div className="text-zinc-600">Waiting for events…</div>}
      {events.map((e, i) => (
        <div key={i} className={`whitespace-pre-wrap break-words ${KIND_STYLE[e.kind] ?? "text-zinc-400"}`}>
          <span className="mr-1.5 select-none opacity-60">[{e.stage}]</span>
          <span className="mr-1.5 select-none">{KIND_PREFIX[e.kind] ?? ""}</span>
          {e.message}
        </div>
      ))}
      <div ref={bottomRef} />
    </div>
  );
}
