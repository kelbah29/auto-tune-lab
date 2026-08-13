"""Goal + runtime hardware -> TechniqueSpec (technique name, execution backend, rationale).

Selection is hardware-aware: the same goal can resolve to a different backend (and,
for "uncensor", a genuinely different technique) depending on whether the target
model fits in local RAM and whether a Tinker API key is configured. See
docs/ARCHITECTURE.md for the full rationale table.
"""
from __future__ import annotations

import os

from app.core.hardware import fits_locally_for_abliteration, fits_locally_for_training
from app.core.schemas import Backend, Goal, ObjectiveType, TechniqueName, TechniqueSpec


def _cloud_backend() -> Backend | None:
    """Which cloud backend (if any) is configured for models too big to fit locally.
    Colab is preferred: genuinely free (personal Google account, no billing, no account-age
    gate) once its one-time OAuth bootstrap (`colab sessions`, approve in browser) is done —
    gated on AUTOTUNELAB_USE_COLAB=1 since there's no env-var-only way to detect that
    bootstrap happened. ZeroGPU is next (free but needs a 30-day-old HF account), then Tinker
    (no daily cap but needs billing). See docs/ARCHITECTURE.md for the full comparison.

    AUTOTUNELAB_LOCAL_ONLY=1 forces this to always return None even if keys are configured
    — configuration presence isn't the same as the account actually being usable (e.g. a
    Tinker key can be billing-blocked, a fresh HF account can be under ZeroGPU's 30-day
    hosting gate), and the fix for those is an external account action, not a retry here.
    """
    if os.environ.get("AUTOTUNELAB_LOCAL_ONLY", "").lower() in ("1", "true", "yes"):
        return None
    if os.environ.get("AUTOTUNELAB_USE_COLAB", "").lower() in ("1", "true", "yes"):
        return Backend.COLAB
    if os.environ.get("HF_TOKEN") and os.environ.get("HF_ZEROGPU_SPACE_ID"):
        return Backend.HF_ZEROGPU
    if os.environ.get("TINKER_API_KEY"):
        return Backend.TINKER
    return None


def _cloud_backend_note(backend: Backend | None) -> str:
    if backend == Backend.COLAB:
        return "runs on Google Colab (free T4 GPU, no billing)"
    if backend == Backend.HF_ZEROGPU:
        return "runs on the free HF ZeroGPU backend (no billing required, but capped at 5 GPU-min/day)"
    if backend == Backend.TINKER:
        return "runs on Tinker"
    return (
        "no cloud backend configured (set AUTOTUNELAB_USE_COLAB=1 after the one-time `colab` "
        "auth, or HF_TOKEN+HF_ZEROGPU_SPACE_ID, or TINKER_API_KEY)"
    )


def select_technique(goal: Goal) -> TechniqueSpec:
    model_ref = goal.target_model_hf_id or goal.target_model_alias or "unknown-7b"
    cloud_backend = _cloud_backend()

    if goal.objective_type == ObjectiveType.UNCENSOR:
        fits, reason = fits_locally_for_abliteration(model_ref)
        if fits:
            return TechniqueSpec(
                name=TechniqueName.ABLITERATION,
                backend=Backend.HERETIC_LOCAL,
                rationale=(
                    "Uncensoring / over-refusal reduction is best solved by directional ablation "
                    "(Arditi et al.; run here via the `heretic` tool) — it removes the refusal "
                    f"direction from the residual stream directly, no gradient training needed. "
                    f"{reason}, so this runs locally."
                ),
            )
        if cloud_backend:
            return TechniqueSpec(
                name=TechniqueName.DPO,
                backend=cloud_backend,
                rationale=(
                    f"{reason}, so local weight-surgery (abliteration) does not fit this machine. "
                    "The cloud path trains via gradients and exposes no direct weight-ablation "
                    "primitive, so this objective is instead solved with a preference-tuning "
                    "refusal-reduction recipe: preferring direct helpful answers over reflexive "
                    "refusals on benign-but-refusal-triggering prompts, while a held-out "
                    "genuinely-harmful set verifies real safety refusals are retained. This is a "
                    f"deliberate technique substitution, not a silent downgrade. This run "
                    f"{_cloud_backend_note(cloud_backend)}."
                ),
            )
        return TechniqueSpec(
            name=TechniqueName.ABLITERATION,
            backend=Backend.HERETIC_LOCAL,
            rationale=(
                f"{reason} — this may be slow or fail on this machine, and {_cloud_backend_note(None)}. "
                "Proceeding locally anyway; consider a smaller model."
            ),
        )

    if goal.objective_type == ObjectiveType.BEHAVIOR_REMOVAL:
        fits, reason = fits_locally_for_training(model_ref)
        if fits:
            return TechniqueSpec(
                name=TechniqueName.LORA_SFT,
                backend=Backend.MLX_LOCAL,
                rationale=(
                    "A narrow, mechanical output constraint ('never produce X') is best taught via "
                    "supervised fine-tuning on rejection-sampled compliant examples — a direct, "
                    f"low-variance signal. {reason}, so this trains locally via LoRA (mlx-lm)."
                ),
            )
        if cloud_backend:
            return TechniqueSpec(
                name=TechniqueName.LORA_SFT,
                backend=cloud_backend,
                rationale=(
                    f"{reason}, so training runs in the cloud instead: LoRA-SFT on rejection-sampled "
                    f"compliant examples. This run {_cloud_backend_note(cloud_backend)}."
                ),
            )
        return TechniqueSpec(
            name=TechniqueName.LORA_SFT,
            backend=Backend.MLX_LOCAL,
            rationale=f"{reason} — {_cloud_backend_note(None)}; attempting local LoRA-SFT anyway, may be slow.",
        )

    if goal.objective_type == ObjectiveType.BENCHMARK_IMPROVEMENT:
        fits, reason = fits_locally_for_training(model_ref)
        if fits:
            return TechniqueSpec(
                name=TechniqueName.LORA_SFT,
                backend=Backend.MLX_LOCAL,
                rationale=(
                    "Benchmark performance is improved by LoRA-SFT on task-matched instruction/"
                    f"solution pairs (retrieved/curated for the named benchmark). {reason}, so this "
                    "trains locally."
                ),
            )
        if cloud_backend:
            return TechniqueSpec(
                name=TechniqueName.LORA_SFT,
                backend=cloud_backend,
                rationale=f"{reason}, so LoRA-SFT {_cloud_backend_note(cloud_backend)} with more capacity.",
            )
        return TechniqueSpec(
            name=TechniqueName.LORA_SFT,
            backend=Backend.MLX_LOCAL,
            rationale=f"{reason} — {_cloud_backend_note(None)}; attempting local LoRA-SFT anyway.",
        )

    # custom / fallback
    fits, reason = fits_locally_for_training(model_ref)
    backend = Backend.MLX_LOCAL if fits else (cloud_backend or Backend.MLX_LOCAL)
    return TechniqueSpec(
        name=TechniqueName.LORA_SFT,
        backend=backend,
        rationale=f"Custom goal defaults to general-purpose LoRA-SFT. {reason}.",
    )
