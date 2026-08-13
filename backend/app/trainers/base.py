"""Common trainer interface implemented by each backend."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from app.core.schemas import DatasetSpec, HyperparamConfig

LogFn = Callable[[str], None]
MetricFn = Callable[[str, float, int], None]  # (name, value, step)


@dataclass
class TrainResult:
    adapter_path: str | None  # LoRA adapter dir, or None for full-weight edits (abliteration)
    model_path: str  # path/id to use for post-training generation (fused local dir, HF id, or Tinker sampling ref)
    backend_ref: str  # opaque backend-specific reference (e.g. Tinker checkpoint URI)
    final_loss: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class Trainer(Protocol):
    async def train(
        self,
        base_model: str,
        dataset: DatasetSpec,
        hyperparams: HyperparamConfig,
        run_dir: str,
        log: LogFn,
        metric: MetricFn,
    ) -> TrainResult: ...
