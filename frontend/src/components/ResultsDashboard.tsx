import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { EvalReport } from "../lib/types";

const METRIC_LABELS: Record<string, string> = {
  violation_rate: "Constraint violation rate",
  avg_coherence: "Avg. coherence (1-5)",
  over_refusal_rate: "Over-refusal rate (safe prompts)",
  safety_retention_rate: "Refusal rate (genuinely harmful prompts)",
  patch_apply_rate: "Patch applies cleanly",
  avg_judge_score: "LLM-judge score (1-5)",
  avg_score: "LLM-judge score (1-5)",
};

const LOWER_IS_BETTER = new Set(["violation_rate", "over_refusal_rate"]);

function fmt(v: number | undefined): string {
  if (v === undefined || Number.isNaN(v)) return "—";
  return Number.isInteger(v) ? String(v) : v.toFixed(2);
}

export default function ResultsDashboard({
  before,
  after,
}: {
  before: EvalReport;
  after: EvalReport;
}) {
  const keys = Array.from(new Set([...Object.keys(before.metrics), ...Object.keys(after.metrics)])).filter(
    (k) => !["total", "n", "violations", "safe_total", "harmful_total", "safe_refusals", "harmful_refusals", "applies"].includes(k)
  );

  const chartData = keys.map((k) => ({
    name: METRIC_LABELS[k] ?? k,
    before: before.metrics[k] ?? 0,
    after: after.metrics[k] ?? 0,
  }));

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
        {keys.map((k) => {
          const b = before.metrics[k];
          const a = after.metrics[k];
          const better = LOWER_IS_BETTER.has(k) ? a < b : a > b;
          const changed = a !== b;
          return (
            <div key={k} className="rounded-xl border border-zinc-800 bg-[#111119] p-3">
              <div className="text-[11px] text-zinc-500">{METRIC_LABELS[k] ?? k}</div>
              <div className="mt-1 flex items-baseline gap-2">
                <span className="text-xs text-zinc-600 line-through decoration-zinc-700">{fmt(b)}</span>
                <span className="text-lg">→</span>
                <span
                  className={`text-lg font-semibold ${
                    changed ? (better ? "text-emerald-400" : "text-red-400") : "text-zinc-200"
                  }`}
                >
                  {fmt(a)}
                </span>
              </div>
            </div>
          );
        })}
      </div>

      {chartData.length > 0 && (
        <div className="h-64 rounded-xl border border-zinc-800 bg-[#111119] p-3">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 8, right: 12, left: 0, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
              <XAxis dataKey="name" tick={{ fill: "#71717a", fontSize: 10 }} interval={0} angle={-15} textAnchor="end" height={60} />
              <YAxis tick={{ fill: "#71717a", fontSize: 10 }} />
              <Tooltip
                contentStyle={{ background: "#18181b", border: "1px solid #3f3f46", borderRadius: 8, fontSize: 12 }}
                labelStyle={{ color: "#e4e4e7" }}
              />
              <Bar dataKey="before" fill="#52525b" radius={[4, 4, 0, 0]} name="Before" />
              <Bar dataKey="after" fill="#8b5cf6" radius={[4, 4, 0, 0]} name="After" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
        {[before, after].map((report) => (
          <div key={report.label} className="rounded-xl border border-zinc-800 bg-[#111119] p-3">
            <div className="mb-2 text-xs font-medium uppercase tracking-wide text-zinc-500">
              {report.label} — samples
            </div>
            <div className="mono max-h-64 space-y-2 overflow-y-auto text-[11px]">
              {report.samples.slice(0, 6).map((s, i) => (
                <div key={i} className="rounded-lg bg-black/30 p-2 text-zinc-400">
                  <pre className="whitespace-pre-wrap break-words">{JSON.stringify(s, null, 1)}</pre>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
