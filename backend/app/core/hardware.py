"""Runtime hardware probing used to route techniques to a local backend or Tinker.

This makes the platform hardware-adaptive: on this 8GB laptop small models train
locally, large ones route to Tinker automatically (when a key is present). On a
beefier GPU box later (e.g. the team's week-2 NLP project), the same logic would
route larger models locally instead of hardcoding a limit.
"""
from __future__ import annotations

import platform
import re

import psutil

# Bytes-per-parameter for a rough memory estimate at a given training dtype.
_BYTES_PER_PARAM = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "int4": 0.6,  # ~4 bits + overhead
}

# Training (grad + optimizer state for a small LoRA adapter, base frozen) needs
# roughly: weights + activations + a modest LoRA/optimizer overhead. This is a
# deliberately conservative multiplier so we don't OOM the demo machine.
_TRAIN_SAFETY_FACTOR = 2.4
# Abliteration is forward-pass only (no optimizer state) but needs full fp16 weights
# resident plus hook buffers.
_ABLITERATION_SAFETY_FACTOR = 1.6

_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[Bb](?![a-zA-Z])")


def estimate_param_count(model_id_or_alias: str) -> float:
    """Best-effort parameter count (in billions) parsed out of a model name."""
    m = _SIZE_RE.search(model_id_or_alias)
    if m:
        return float(m.group(1))
    return 7.0  # unknown -> assume mid-size, be conservative


def available_ram_gb() -> float:
    return psutil.virtual_memory().available / (1024**3)


def total_ram_gb() -> float:
    return psutil.virtual_memory().total / (1024**3)


def has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def has_mps() -> bool:
    try:
        import torch

        return torch.backends.mps.is_available()
    except Exception:
        return platform.system() == "Darwin" and platform.processor() == "arm"


class LocalFeasibility(str):
    pass


def fits_locally_for_training(model_id_or_alias: str, dtype: str = "fp16") -> tuple[bool, str]:
    params_b = estimate_param_count(model_id_or_alias)
    est_gb = params_b * 1e9 * _BYTES_PER_PARAM[dtype] / (1024**3) * _TRAIN_SAFETY_FACTOR
    avail = available_ram_gb()
    ok = est_gb <= avail * 0.75  # leave headroom for OS + app
    reason = (
        f"~{params_b:g}B params @ {dtype} training needs ~{est_gb:.1f}GB "
        f"(available {avail:.1f}GB of {total_ram_gb():.1f}GB total)"
    )
    return ok, reason


def fits_locally_for_abliteration(model_id_or_alias: str, dtype: str = "fp16") -> tuple[bool, str]:
    params_b = estimate_param_count(model_id_or_alias)
    est_gb = params_b * 1e9 * _BYTES_PER_PARAM[dtype] / (1024**3) * _ABLITERATION_SAFETY_FACTOR
    avail = available_ram_gb()
    ok = est_gb <= avail * 0.75
    reason = (
        f"~{params_b:g}B params @ {dtype} weight-surgery needs ~{est_gb:.1f}GB "
        f"(available {avail:.1f}GB of {total_ram_gb():.1f}GB total)"
    )
    return ok, reason
