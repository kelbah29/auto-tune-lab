import { useEffect, useState, useCallback } from "react";
import { getHealth, listRuns } from "./lib/api";
import type { RunState } from "./lib/types";
import Console from "./pages/Console";
import RunView from "./pages/RunView";

type Health = Awaited<ReturnType<typeof getHealth>>;

function statusDot(status: RunState["status"]) {
  if (status === "completed") return "bg-emerald-400";
  if (status === "failed") return "bg-red-400";
  if (status === "cancelled") return "bg-zinc-500";
  return "bg-amber-400 animate-pulse";
}

export default function App() {
  const [selectedRun, setSelectedRun] = useState<string | null>(null);
  const [runs, setRuns] = useState<RunState[]>([]);
  const [health, setHealth] = useState<Health | null>(null);

  const refreshRuns = useCallback(() => {
    listRuns().then(setRuns).catch(() => {});
  }, []);

  useEffect(() => {
    refreshRuns();
    getHealth().then(setHealth).catch(() => {});
    const id = setInterval(refreshRuns, 4000);
    return () => clearInterval(id);
  }, [refreshRuns]);

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#0a0a0f] text-zinc-100">
      <aside className="flex w-72 shrink-0 flex-col border-r border-zinc-800/80 bg-[#0d0d14]">
        <button
          onClick={() => setSelectedRun(null)}
          className="flex items-center gap-2 border-b border-zinc-800/80 px-4 py-4 text-left transition hover:bg-zinc-900/60"
        >
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-violet-500 to-fuchsia-500 text-sm font-bold text-white">
            AT
          </div>
          <div>
            <div className="text-sm font-semibold leading-tight">AutoTuneLab</div>
            <div className="text-[11px] leading-tight text-zinc-500">Fine-tuning agent</div>
          </div>
        </button>

        <div className="flex-1 overflow-y-auto px-2 py-2">
          <div className="px-2 py-1 text-[11px] font-medium uppercase tracking-wide text-zinc-600">
            Runs
          </div>
          {runs.length === 0 && (
            <div className="px-2 py-3 text-xs text-zinc-600">No runs yet.</div>
          )}
          {runs.map((r) => (
            <button
              key={r.run_id}
              onClick={() => setSelectedRun(r.run_id)}
              className={`mb-1 flex w-full flex-col gap-0.5 rounded-lg px-2.5 py-2 text-left transition ${
                selectedRun === r.run_id ? "bg-violet-500/15 ring-1 ring-violet-500/40" : "hover:bg-zinc-900/70"
              }`}
            >
              <div className="flex items-center gap-1.5">
                <span className={`h-1.5 w-1.5 rounded-full ${statusDot(r.status)}`} />
                <span className="truncate text-xs font-medium text-zinc-200">
                  {r.goal.raw_prompt || r.run_id}
                </span>
              </div>
              <span className="pl-3 text-[10px] text-zinc-600">{r.status}</span>
            </button>
          ))}
        </div>

        <div className="border-t border-zinc-800/80 px-3 py-3 text-[11px] text-zinc-500">
          {health ? (
            <div className="space-y-1">
              <div className="flex items-center justify-between">
                <span>Tinker</span>
                <span className={health.tinker_configured ? "text-emerald-400" : "text-zinc-600"}>
                  {health.tinker_configured ? "configured" : "not configured"}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span>RAM</span>
                <span className="text-zinc-400">
                  {health.ram_available_gb}/{health.ram_total_gb} GB
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span>Accelerator</span>
                <span className="text-zinc-400">{health.mps ? "MPS" : health.cuda ? "CUDA" : "CPU"}</span>
              </div>
            </div>
          ) : (
            <span className="text-zinc-700">backend offline</span>
          )}
        </div>
      </aside>

      <main className="min-w-0 flex-1 overflow-hidden">
        {selectedRun ? (
          <RunView runId={selectedRun} />
        ) : (
          <Console
            onRunStarted={(id) => {
              refreshRuns();
              setSelectedRun(id);
            }}
          />
        )}
      </main>
    </div>
  );
}
