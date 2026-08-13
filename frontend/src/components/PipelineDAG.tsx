import { PIPELINE_STAGES, type RunStatus } from "../lib/types";

const ORDER: RunStatus[] = [
  "parsing_goal",
  "awaiting_clarification",
  "selecting_technique",
  "resolving_dataset",
  "configuring_hyperparams",
  "awaiting_review",
  "training",
  "evaluating",
  "completed",
];

function indexOf(status: RunStatus): number {
  const i = ORDER.indexOf(status);
  return i === -1 ? 0 : i;
}

export default function PipelineDAG({ status }: { status: RunStatus }) {
  const failed = status === "failed";
  const cancelled = status === "cancelled";
  const currentIdx = failed || cancelled ? ORDER.length : indexOf(status);

  return (
    <div className="flex items-center gap-0 overflow-x-auto px-4 py-3">
      {PIPELINE_STAGES.map((stage, i) => {
        const stageIdx = indexOf(stage.key);
        const isCurrent = !failed && !cancelled && stage.key === status;
        const isDone = !failed && !cancelled && stageIdx < currentIdx;
        const isReviewWaiting = status === "awaiting_review" && stage.key === "awaiting_review";

        let dotClass = "bg-zinc-700";
        if (isDone) dotClass = "bg-emerald-400";
        else if (isCurrent || isReviewWaiting) dotClass = "bg-violet-400 animate-pulse";
        else if (failed && i === PIPELINE_STAGES.length - 1) dotClass = "bg-zinc-700";

        return (
          <div key={stage.key} className="flex items-center">
            <div className="flex flex-col items-center gap-1">
              <div className={`h-2.5 w-2.5 rounded-full ${dotClass}`} />
              <span
                className={`whitespace-nowrap text-[10.5px] ${
                  isCurrent || isReviewWaiting ? "text-violet-300" : isDone ? "text-zinc-400" : "text-zinc-600"
                }`}
              >
                {stage.label}
              </span>
            </div>
            {i < PIPELINE_STAGES.length - 1 && (
              <div
                className={`mx-1 h-px w-8 ${
                  stageIdx < currentIdx && !failed && !cancelled ? "bg-emerald-400/50" : "bg-zinc-800"
                }`}
              />
            )}
          </div>
        );
      })}
      {failed && (
        <span className="ml-4 rounded-full bg-red-500/15 px-2.5 py-1 text-[11px] font-medium text-red-400">
          failed
        </span>
      )}
      {cancelled && (
        <span className="ml-4 rounded-full bg-zinc-500/15 px-2.5 py-1 text-[11px] font-medium text-zinc-400">
          cancelled
        </span>
      )}
    </div>
  );
}
