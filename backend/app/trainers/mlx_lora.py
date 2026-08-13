"""Local LoRA/SFT training on Apple Silicon via mlx-lm's `mlx_lm.lora` CLI.

We shell out to the real console script (rather than reimplementing the training
loop) and stream its stdout live so the orchestrator can forward loss/iter
progress to the UI. This is the backend used for the real local end-to-end smoke
test on this machine.
"""
from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path

from app.core.schemas import DatasetSpec, HyperparamConfig
from app.trainers.base import LogFn, MetricFn, TrainResult

_LOSS_RE = re.compile(r"Iter\s+(\d+):\s+Train loss\s+([\d.]+)")
_VAL_LOSS_RE = re.compile(r"Iter\s+(\d+):\s+Val loss\s+([\d.]+)")

_MLX_LORA_BIN = str(Path(sys.executable).with_name("mlx_lm.lora"))
_MLX_FUSE_BIN = str(Path(sys.executable).with_name("mlx_lm.fuse"))


async def _stream_process(cmd: list[str], log: LogFn, on_line=None) -> int:
    log(f"$ {' '.join(cmd)}")
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
    )
    assert proc.stdout is not None
    async for raw_line in proc.stdout:
        line = raw_line.decode(errors="replace").rstrip()
        if line:
            log(line)
            if on_line:
                on_line(line)
    return await proc.wait()


class MLXLoraTrainer:
    async def train(
        self,
        base_model: str,
        dataset: DatasetSpec,
        hyperparams: HyperparamConfig,
        run_dir: str,
        log: LogFn,
        metric: MetricFn,
    ) -> TrainResult:
        data_dir = str(Path(dataset.train_path).parent)
        adapter_path = str(Path(run_dir) / "artifacts" / "adapters")
        Path(adapter_path).mkdir(parents=True, exist_ok=True)

        cmd = [
            _MLX_LORA_BIN,
            "--model", base_model,
            "--train",
            "--data", data_dir,
            "--fine-tune-type", "lora",
            "--num-layers", str(hyperparams.num_layers or 8),
            "--batch-size", str(max(1, hyperparams.batch_size)),
            "--iters", str(hyperparams.iters or 200),
            "--learning-rate", str(hyperparams.learning_rate),
            "--max-seq-length", str(hyperparams.max_seq_length),
            "--adapter-path", adapter_path,
            "--steps-per-report", "5",
            "--save-every", str(max(10, (hyperparams.iters or 200))),
            "--mask-prompt",
            "--seed", str(hyperparams.seed),
            # Default val-batches checks the whole val set every eval — observed taking 24+
            # minutes per pass on long-sequence datasets (e.g. SWE-bench patches) on this
            # CPU/MPS-only Mac. A small fixed val-batches keeps validation meaningful without
            # making it the dominant cost of a local run.
            "--val-batches", str(hyperparams.extra.get("val_batches", 2)),
        ]

        final_loss = None

        def on_line(line: str) -> None:
            nonlocal final_loss
            m = _LOSS_RE.search(line)
            if m:
                step, loss = int(m.group(1)), float(m.group(2))
                final_loss = loss
                metric("train_loss", loss, step)
            v = _VAL_LOSS_RE.search(line)
            if v:
                step, loss = int(v.group(1)), float(v.group(2))
                metric("val_loss", loss, step)

        rc = await _stream_process(cmd, log, on_line)
        if rc != 0:
            raise RuntimeError(f"mlx_lm.lora exited with code {rc}")

        return TrainResult(
            adapter_path=adapter_path,
            model_path=base_model,
            backend_ref=f"mlx_local:{base_model}",
            final_loss=final_loss,
        )


async def fuse_adapter(base_model: str, adapter_path: str, out_dir: str, log: LogFn) -> str:
    """Optional: merge the LoRA adapter into full weights for faster/simpler downstream use."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cmd = [
        _MLX_FUSE_BIN,
        "--model", base_model,
        "--adapter-path", adapter_path,
        "--save-path", out_dir,
    ]
    rc = await _stream_process(cmd, log)
    if rc != 0:
        raise RuntimeError(f"mlx_lm.fuse exited with code {rc}")
    return out_dir
