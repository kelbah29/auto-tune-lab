import { useState } from "react";
import type { HyperparamConfig, TechniqueName } from "../lib/types";

const FIELDS_BY_TECHNIQUE: Record<TechniqueName, (keyof HyperparamConfig)[]> = {
  lora_sft: ["learning_rate", "iters", "batch_size", "lora_rank", "lora_alpha", "lora_dropout", "max_seq_length", "num_layers"],
  qlora_sft: ["learning_rate", "iters", "batch_size", "lora_rank", "lora_alpha", "lora_dropout", "max_seq_length", "num_layers"],
  dpo: ["learning_rate", "iters", "batch_size", "lora_rank", "lora_alpha", "max_seq_length"],
  abliteration: ["batch_size", "max_seq_length"],
};

const LABELS: Partial<Record<keyof HyperparamConfig, string>> = {
  learning_rate: "Learning rate",
  iters: "Iterations",
  batch_size: "Batch size",
  lora_rank: "LoRA rank",
  lora_alpha: "LoRA alpha",
  lora_dropout: "LoRA dropout",
  max_seq_length: "Max seq length",
  num_layers: "Trainable layers",
};

export default function HyperparamForm({
  technique,
  hyperparams,
  onSave,
  onApprove,
  saving,
}: {
  technique: TechniqueName;
  hyperparams: HyperparamConfig;
  onSave: (updates: Partial<HyperparamConfig>) => Promise<void>;
  onApprove: () => Promise<void>;
  saving: boolean;
}) {
  const [values, setValues] = useState<HyperparamConfig>(hyperparams);
  const [dirty, setDirty] = useState(false);
  const fields = FIELDS_BY_TECHNIQUE[technique] ?? [];

  function update<K extends keyof HyperparamConfig>(key: K, val: string) {
    const num = Number(val);
    setValues((v) => ({ ...v, [key]: Number.isNaN(num) ? v[key] : num }));
    setDirty(true);
  }

  return (
    <div className="rounded-xl border border-violet-500/30 bg-violet-500/5 p-4">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-violet-200">Paused for review</h3>
        <span className="text-[11px] text-zinc-500">edit hyperparameters, then approve</span>
      </div>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {fields.map((key) => (
          <label key={key} className="flex flex-col gap-1">
            <span className="text-[11px] text-zinc-500">{LABELS[key] ?? key}</span>
            <input
              type="number"
              step="any"
              value={values[key] as number}
              onChange={(e) => update(key, e.target.value)}
              className="rounded-md border border-zinc-700 bg-zinc-900 px-2 py-1 text-sm text-zinc-100 outline-none focus:border-violet-500"
            />
          </label>
        ))}
        {technique === "abliteration" && (
          <label className="flex flex-col gap-1">
            <span className="text-[11px] text-zinc-500">Trials</span>
            <input
              type="number"
              value={(values.extra?.trials as number) ?? 15}
              onChange={(e) =>
                setValues((v) => ({ ...v, extra: { ...v.extra, trials: Number(e.target.value) } }))
              }
              className="rounded-md border border-zinc-700 bg-zinc-900 px-2 py-1 text-sm text-zinc-100 outline-none focus:border-violet-500"
            />
          </label>
        )}
      </div>

      <div className="mt-4 flex justify-end gap-2">
        {dirty && (
          <button
            disabled={saving}
            onClick={async () => {
              await onSave(values);
              setDirty(false);
            }}
            className="rounded-lg border border-zinc-700 px-3 py-1.5 text-sm text-zinc-300 transition hover:border-violet-500 disabled:opacity-40"
          >
            Save changes
          </button>
        )}
        <button
          disabled={saving}
          onClick={async () => {
            if (dirty) {
              await onSave(values);
              setDirty(false);
            }
            await onApprove();
          }}
          className="rounded-lg bg-violet-500 px-4 py-1.5 text-sm font-medium text-white transition hover:bg-violet-400 disabled:opacity-40"
        >
          {dirty ? "Save & continue" : "Approve & continue"}
        </button>
      </div>
    </div>
  );
}
