import { useEffect, useState, useRef, useCallback } from "react";
import {
  approveRun,
  cancelRun,
  getRun,
  patchHyperparams,
  sendMessage,
  subscribeEvents,
} from "../lib/api";
import type { HyperparamConfig, RunEvent, RunState } from "../lib/types";
import PipelineDAG from "../components/PipelineDAG";
import EventLog from "../components/EventLog";
import HyperparamForm from "../components/HyperparamForm";
import ResultsDashboard from "../components/ResultsDashboard";

export default function RunView({ runId }: { runId: string }) {
  const [run, setRun] = useState<RunState | null>(null);
  const [events, setEvents] = useState<RunEvent[]>([]);
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const eventsRef = useRef<RunEvent[]>([]);

  const refresh = useCallback(() => {
    getRun(runId).then(setRun).catch(() => {});
  }, [runId]);

  useEffect(() => {
    setEvents([]);
    eventsRef.current = [];
    refresh();
    const unsub = subscribeEvents(runId, (e) => {
      eventsRef.current = [...eventsRef.current, e];
      setEvents(eventsRef.current);
      if (e.kind === "state_change") refresh();
    });
    const poll = setInterval(refresh, 5000);
    return () => {
      unsub();
      clearInterval(poll);
    };
  }, [runId, refresh]);

  if (!run) {
    return <div className="flex h-full items-center justify-center text-zinc-600">Loading…</div>;
  }

  const g = run.goal;

  async function handleSend() {
    if (!message.trim()) return;
    setBusy(true);
    try {
      await sendMessage(runId, message.trim());
      setMessage("");
    } finally {
      setBusy(false);
    }
  }

  async function handleApprove() {
    setBusy(true);
    try {
      await approveRun(runId);
    } finally {
      setBusy(false);
    }
  }

  async function handleSaveHyperparams(updates: Partial<HyperparamConfig>) {
    await patchHyperparams(runId, updates);
    refresh();
  }

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <header className="border-b border-zinc-800/80 px-5 py-3">
        <div className="flex items-center justify-between gap-3">
          <div className="min-w-0">
            <div className="truncate text-sm font-medium text-zinc-200">{g.raw_prompt}</div>
            <div className="mt-0.5 flex flex-wrap gap-x-3 gap-y-0.5 text-[11px] text-zinc-500">
              {g.target_model_hf_id && <span>model: {g.target_model_hf_id}</span>}
              {run.technique && (
                <span>
                  technique: {run.technique.name} · backend: {run.technique.backend}
                </span>
              )}
              {g.objective_type && <span>objective: {g.objective_type}</span>}
            </div>
          </div>
          {!["completed", "failed", "cancelled"].includes(run.status) && (
            <button
              onClick={() => cancelRun(runId).then(refresh)}
              className="shrink-0 rounded-lg border border-zinc-800 px-2.5 py-1 text-[11px] text-zinc-500 transition hover:border-red-500/50 hover:text-red-400"
            >
              Cancel
            </button>
          )}
        </div>
        <PipelineDAG status={run.status} />
      </header>

      <div className="flex min-h-0 flex-1 gap-4 overflow-hidden p-4">
        <div className="flex min-w-0 flex-1 flex-col gap-3">
          <EventLog events={events} />

          <div className="flex gap-2">
            <input
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSend()}
              disabled={busy}
              placeholder={
                run.status === "awaiting_clarification"
                  ? "Answer the question above…"
                  : "Steer the run (e.g. 'use a smaller model', 'focus on Python only')…"
              }
              className="flex-1 rounded-lg border border-zinc-800 bg-[#111119] px-3 py-2 text-sm text-zinc-100 placeholder-zinc-600 outline-none focus:border-violet-500"
            />
            <button
              onClick={handleSend}
              disabled={busy || !message.trim()}
              className="rounded-lg border border-zinc-700 px-3 py-2 text-sm text-zinc-300 transition hover:border-violet-500 disabled:opacity-40"
            >
              Send
            </button>
          </div>
        </div>

        <div className="w-96 shrink-0 space-y-3 overflow-y-auto pr-1">
          {run.status === "awaiting_review" && run.hyperparams && run.technique && (
            <HyperparamForm
              technique={run.technique.name}
              hyperparams={run.hyperparams}
              onSave={handleSaveHyperparams}
              onApprove={handleApprove}
              saving={busy}
            />
          )}

          {run.dataset && (
            <div className="rounded-xl border border-zinc-800 bg-[#111119] p-3 text-xs">
              <div className="mb-1 font-medium text-zinc-300">Dataset</div>
              <div className="text-zinc-500">{run.dataset.description}</div>
              <div className="mt-1.5 flex gap-3 text-zinc-600">
                <span>train {run.dataset.num_train}</span>
                <span>val {run.dataset.num_val}</span>
                <span>eval {run.dataset.num_eval_holdout}</span>
              </div>
            </div>
          )}

          {run.technique && (
            <div className="rounded-xl border border-zinc-800 bg-[#111119] p-3 text-xs">
              <div className="mb-1 font-medium text-zinc-300">Why this technique</div>
              <div className="text-zinc-500">{run.technique.rationale}</div>
            </div>
          )}

          {run.error && (
            <div className="rounded-xl border border-red-500/30 bg-red-500/5 p-3 text-xs text-red-400">
              {run.error}
            </div>
          )}
        </div>
      </div>

      {run.status === "completed" && run.eval_before && run.eval_after && (
        <div className="max-h-[45vh] overflow-y-auto border-t border-zinc-800/80 p-4">
          <ResultsDashboard before={run.eval_before} after={run.eval_after} />
        </div>
      )}
    </div>
  );
}
