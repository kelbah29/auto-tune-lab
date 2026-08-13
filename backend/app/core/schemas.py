"""Core pydantic data models shared across the pipeline."""
from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class ObjectiveType(str, Enum):
    BEHAVIOR_REMOVAL = "behavior_removal"
    UNCENSOR = "uncensor"
    BENCHMARK_IMPROVEMENT = "benchmark_improvement"
    CUSTOM = "custom"


class TechniqueName(str, Enum):
    LORA_SFT = "lora_sft"
    QLORA_SFT = "qlora_sft"
    DPO = "dpo"
    ABLITERATION = "abliteration"


class Backend(str, Enum):
    MLX_LOCAL = "mlx_local"
    HERETIC_LOCAL = "heretic_local"
    TINKER = "tinker"
    HF_ZEROGPU = "hf_zerogpu"
    COLAB = "colab"


class RunStatus(str, Enum):
    CREATED = "created"
    PARSING_GOAL = "parsing_goal"
    AWAITING_CLARIFICATION = "awaiting_clarification"
    SELECTING_TECHNIQUE = "selecting_technique"
    RESOLVING_DATASET = "resolving_dataset"
    CONFIGURING_HYPERPARAMS = "configuring_hyperparams"
    AWAITING_REVIEW = "awaiting_review"
    TRAINING = "training"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Goal(BaseModel):
    raw_prompt: str
    follow_up_prompts: list[str] = Field(default_factory=list)
    target_model_alias: str | None = None
    target_model_hf_id: str | None = None
    objective_type: ObjectiveType | None = None
    behavior_description: str = ""
    target_benchmark: str | None = None
    negative_constraint_token: str | None = None  # e.g. "b" for the letter-B task
    confidence: float = 0.0
    clarification_questions: list[str] = Field(default_factory=list)
    reasoning: str = ""


class TechniqueSpec(BaseModel):
    name: TechniqueName
    backend: Backend
    rationale: str


class HyperparamConfig(BaseModel):
    # Generic + technique-specific fields, kept flexible for the editable-form use case.
    learning_rate: float = 1e-4
    epochs: int = 1
    iters: int | None = None
    batch_size: int = 4
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    max_seq_length: int = 1024
    num_layers: int = 16
    warmup_steps: int = 10
    seed: int = 0
    extra: dict[str, Any] = Field(default_factory=dict)


class DatasetExample(BaseModel):
    messages: list[dict[str, str]]


class DatasetSpec(BaseModel):
    source: Literal["synthetic", "huggingface_hub", "curated_builtin"]
    hf_dataset_id: str | None = None
    description: str = ""
    num_train: int = 0
    num_val: int = 0
    num_eval_holdout: int = 0
    train_path: str | None = None
    val_path: str | None = None
    eval_holdout_path: str | None = None


class EvalReport(BaseModel):
    label: str
    metrics: dict[str, float] = Field(default_factory=dict)
    samples: list[dict[str, Any]] = Field(default_factory=list)


class RunEvent(BaseModel):
    ts: float = Field(default_factory=time.time)
    stage: str
    kind: Literal["info", "reasoning", "log", "warning", "error", "state_change", "metric"]
    message: str
    data: dict[str, Any] = Field(default_factory=dict)


class RunState(BaseModel):
    run_id: str = Field(default_factory=lambda: new_id("run"))
    status: RunStatus = RunStatus.CREATED
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    auto_approve: bool = False

    goal: Goal
    technique: TechniqueSpec | None = None
    hyperparams: HyperparamConfig | None = None
    dataset: DatasetSpec | None = None

    eval_before: EvalReport | None = None
    eval_after: EvalReport | None = None

    adapter_path: str | None = None
    error: str | None = None

    def touch(self) -> None:
        self.updated_at = time.time()
