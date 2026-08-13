"""LoRA-SFT / SimPO-preference training via the free HF ZeroGPU Space (zerogpu_space/),
used when a target model doesn't fit in local RAM and the user wants a cloud backend that
needs no billing at all (unlike Tinker). See zerogpu_space/README.md for the API contract
and zerogpu_space/app.py for why "dpo" mode is actually a reference-free SimPO loss.

Free-tier ZeroGPU gives only 5 minutes of GPU time per day, so a single job may not reach
its full step target within one call — `train()` surfaces that honestly via `result["done"]`
rather than silently pretending the run finished.
"""
from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path

from app.core.schemas import DatasetSpec, HyperparamConfig
from app.trainers.base import LogFn, MetricFn, TrainResult


def _space_id() -> str:
    space_id = os.environ.get("HF_ZEROGPU_SPACE_ID")
    if not space_id:
        raise RuntimeError(
            "HF_ZEROGPU_SPACE_ID not configured (expected e.g. "
            '"your-hf-username/autotunelab-zerogpu-trainer" in backend/.env)'
        )
    return space_id


def _client():
    from gradio_client import Client

    return Client(_space_id(), hf_token=os.environ.get("HF_TOKEN"))


class HFZeroGPUTrainer:
    def __init__(self, mode: str) -> None:
        self.mode = mode  # "sft" or "dpo" (SimPO-style, see module docstring)

    async def train(
        self,
        base_model: str,
        dataset: DatasetSpec,
        hyperparams: HyperparamConfig,
        run_dir: str,
        log: LogFn,
        metric: MetricFn,
    ) -> TrainResult:
        job_id = "job_" + uuid.uuid4().hex[:12]
        # dataset.train_path always points at the SFT-style {"messages": [...]} view
        # (resolver.py writes a chosen-only fallback there for every objective type);
        # the {"prompt", "chosen", "rejected"} pairs zerogpu_space/app.py's dpo mode
        # actually reads live in a sibling dpo_pairs.jsonl — see the identical
        # derivation in tinker_trainer.py's preference trainer.
        dataset_path = (
            Path(dataset.train_path).parent / "dpo_pairs.jsonl"
            if self.mode == "dpo"
            else Path(dataset.train_path)
        )
        dataset_jsonl = dataset_path.read_text()

        hp = {
            "learning_rate": hyperparams.learning_rate,
            "lora_rank": hyperparams.lora_rank,
            "lora_alpha": hyperparams.lora_alpha,
            "max_seq_length": hyperparams.max_seq_length,
            "target_steps": hyperparams.iters or int(hyperparams.extra.get("target_steps", 30)),
            "batch_size": hyperparams.batch_size,
            "max_seconds": hyperparams.extra.get("max_seconds", 240),
        }
        if self.mode == "dpo":
            hp["simpo_beta"] = hyperparams.extra.get("simpo_beta", 2.0)
            hp["simpo_gamma"] = hyperparams.extra.get("simpo_gamma", 0.5)

        log(
            f"Submitting ZeroGPU training job {job_id} (mode={self.mode}, base={base_model}, "
            f"~{hp['target_steps']} steps, {hp['max_seconds']}s budget). Free-tier ZeroGPU is "
            "5 min/day total, so this may only partially complete — see result below."
        )

        def _call() -> str:
            client = _client()
            return client.predict(
                job_id, self.mode, base_model, dataset_jsonl, json.dumps(hp), api_name="/train"
            )

        raw = await asyncio.to_thread(_call)
        result = json.loads(raw)
        log(
            f"ZeroGPU training returned: step={result['step']}/{result['target_steps']} "
            f"loss={result['loss']} elapsed={result['elapsed']:.1f}s done={result['done']}"
        )
        if not result["done"]:
            log(
                "Did not reach target_steps within the per-call time budget — adapter reflects "
                "partial training. Re-run tomorrow (quota resets 24h after first use) for more "
                "steps, or lower target_steps.",
                kind="info",
            )
        metric("loss", result["loss"] or 0.0, result["step"])

        return TrainResult(
            adapter_path=job_id,  # weights live on the Space; job_id is how generate() finds them
            model_path=base_model,
            backend_ref=f"hf_zerogpu:{base_model}:{job_id}",
            final_loss=result["loss"],
            extra={"zerogpu_done": result["done"], "zerogpu_step": result["step"]},
        )
