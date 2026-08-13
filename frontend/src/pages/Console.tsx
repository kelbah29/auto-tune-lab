import { useState } from "react";
import { createRun } from "../lib/api";

const EXAMPLES = [
  "Remove Qwen3.5-4B's ability to say the letter B",
  "Uncensor Qwen3.5-4B",
  "Fine tune Qwen2.5-Coder-7B-Instruct and raise its performance on SWE-bench",
];

export default function Console({ onRunStarted }: { onRunStarted: (runId: string) => void }) {
  const [prompt, setPrompt] = useState("");
  const [autoApprove, setAutoApprove] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit() {
    if (!prompt.trim() || submitting) return;
    setSubmitting(true);
    setError(null);
    try {
      const run = await createRun(prompt.trim(), autoApprove);
      onRunStarted(run.run_id);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="flex h-full flex-col items-center justify-center px-6">
      <div className="w-full max-w-2xl">
        <h1 className="mb-2 text-center text-3xl font-semibold tracking-tight">
          Describe your fine-tuning goal
        </h1>
        <p className="mb-8 text-center text-sm text-zinc-500">
          One prompt sets up the whole loop — dataset, technique, hyperparameters, training, and
          evals. You can steer or tweak it once it starts.
        </p>

        <div className="rounded-2xl border border-zinc-800 bg-[#111119] p-4 shadow-2xl shadow-black/40">
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) submit();
            }}
            placeholder="e.g. Remove Qwen3.5-4B's ability to say the letter B"
            rows={4}
            className="w-full resize-none bg-transparent text-[15px] text-zinc-100 placeholder-zinc-600 outline-none"
          />
          <div className="mt-3 flex items-center justify-between border-t border-zinc-800 pt-3">
            <label className="flex items-center gap-2 text-xs text-zinc-500">
              <input
                type="checkbox"
                checked={autoApprove}
                onChange={(e) => setAutoApprove(e.target.checked)}
                className="h-3.5 w-3.5 rounded border-zinc-700 bg-zinc-900 accent-violet-500"
              />
              Auto-approve (skip config review pause)
            </label>
            <button
              onClick={submit}
              disabled={!prompt.trim() || submitting}
              className="rounded-lg bg-violet-500 px-4 py-1.5 text-sm font-medium text-white transition hover:bg-violet-400 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {submitting ? "Starting…" : "Start ⌘⏎"}
            </button>
          </div>
        </div>

        {error && <div className="mt-3 text-sm text-red-400">{error}</div>}

        <div className="mt-6 flex flex-wrap justify-center gap-2">
          {EXAMPLES.map((ex) => (
            <button
              key={ex}
              onClick={() => setPrompt(ex)}
              className="rounded-full border border-zinc-800 bg-zinc-900/60 px-3 py-1.5 text-xs text-zinc-400 transition hover:border-violet-500/50 hover:text-zinc-200"
            >
              {ex}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
