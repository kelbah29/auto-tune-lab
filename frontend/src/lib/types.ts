export type ObjectiveType = "behavior_removal" | "uncensor" | "benchmark_improvement" | "custom";
export type TechniqueName = "lora_sft" | "qlora_sft" | "dpo" | "abliteration";
export type Backend = "mlx_local" | "heretic_local" | "tinker";
export type RunStatus =
  | "created"
  | "parsing_goal"
  | "awaiting_clarification"
  | "selecting_technique"
  | "resolving_dataset"
  | "configuring_hyperparams"
  | "awaiting_review"
  | "training"
  | "evaluating"
  | "completed"
  | "failed"
  | "cancelled";

export interface Goal {
  raw_prompt: string;
  follow_up_prompts: string[];
  target_model_alias: string | null;
  target_model_hf_id: string | null;
  objective_type: ObjectiveType | null;
  behavior_description: string;
  target_benchmark: string | null;
  negative_constraint_token: string | null;
  confidence: number;
  clarification_questions: string[];
  reasoning: string;
}

export interface TechniqueSpec {
  name: TechniqueName;
  backend: Backend;
  rationale: string;
}

export interface HyperparamConfig {
  learning_rate: number;
  epochs: number;
  iters: number | null;
  batch_size: number;
  lora_rank: number;
  lora_alpha: number;
  lora_dropout: number;
  max_seq_length: number;
  num_layers: number;
  warmup_steps: number;
  seed: number;
  extra: Record<string, unknown>;
}

export interface DatasetSpec {
  source: "synthetic" | "huggingface_hub" | "curated_builtin";
  hf_dataset_id: string | null;
  description: string;
  num_train: number;
  num_val: number;
  num_eval_holdout: number;
  train_path: string | null;
  val_path: string | null;
  eval_holdout_path: string | null;
}

export interface EvalReport {
  label: string;
  metrics: Record<string, number>;
  samples: Record<string, unknown>[];
}

export interface RunState {
  run_id: string;
  status: RunStatus;
  created_at: number;
  updated_at: number;
  auto_approve: boolean;
  goal: Goal;
  technique: TechniqueSpec | null;
  hyperparams: HyperparamConfig | null;
  dataset: DatasetSpec | null;
  eval_before: EvalReport | null;
  eval_after: EvalReport | null;
  adapter_path: string | null;
  error: string | null;
}

export interface RunEvent {
  ts: number;
  stage: string;
  kind: "info" | "reasoning" | "log" | "warning" | "error" | "state_change" | "metric";
  message: string;
  data: Record<string, unknown>;
}

export const PIPELINE_STAGES: { key: RunStatus; label: string }[] = [
  { key: "parsing_goal", label: "Parse Goal" },
  { key: "selecting_technique", label: "Select Technique" },
  { key: "resolving_dataset", label: "Resolve Dataset" },
  { key: "configuring_hyperparams", label: "Configure Hyperparams" },
  { key: "awaiting_review", label: "Review" },
  { key: "training", label: "Train" },
  { key: "evaluating", label: "Evaluate" },
  { key: "completed", label: "Done" },
];
