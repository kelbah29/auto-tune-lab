"""TechniqueSpec -> default, user-editable HyperparamConfig."""
from __future__ import annotations

from app.core.schemas import Backend, HyperparamConfig, TechniqueName, TechniqueSpec


def default_hyperparams(spec: TechniqueSpec, dataset_size_hint: int = 200) -> HyperparamConfig:
    if spec.name == TechniqueName.ABLITERATION:
        # Not gradient-based: fields below are informational for the UI; heretic_abliteration.py
        # reads `extra` for its own knobs (trials, direction layer search).
        return HyperparamConfig(
            learning_rate=0.0,
            epochs=0,
            iters=0,
            batch_size=8,
            lora_rank=0,
            lora_alpha=0,
            lora_dropout=0.0,
            max_seq_length=512,
            num_layers=0,
            warmup_steps=0,
            extra={"trials": 15, "direction_layer": "auto", "device": "auto"},
        )

    if spec.name == TechniqueName.DPO:
        return HyperparamConfig(
            learning_rate=5e-6,
            epochs=1,
            iters=max(50, dataset_size_hint),
            batch_size=4,
            lora_rank=16,
            lora_alpha=32,
            lora_dropout=0.0,
            max_seq_length=1024,
            num_layers=16,
            warmup_steps=10,
            extra={"beta": 0.1},
        )

    # LoRA / QLoRA SFT default. Kept modest relative to dataset size — an earlier local
    # smoke test at 3x/dataset iters drove train loss to ~0 (severe overfitting on a
    # ~25-example set, val loss rising) while held-out violation rate only dropped
    # 90%->70%; the model memorized examples instead of generalizing the rule.
    iters = max(60, dataset_size_hint * 2)
    return HyperparamConfig(
        learning_rate=1e-4,
        epochs=1,
        iters=iters,
        batch_size=4,
        lora_rank=16 if spec.backend != Backend.TINKER else 32,
        lora_alpha=32 if spec.backend != Backend.TINKER else 64,
        lora_dropout=0.05,
        max_seq_length=1024,
        num_layers=16 if spec.backend != Backend.TINKER else 0,
        warmup_steps=max(5, iters // 20),
    )


# Schema hints for the frontend to render an appropriate editable form per technique.
EDITABLE_FIELDS = {
    TechniqueName.LORA_SFT: [
        "learning_rate", "iters", "batch_size", "lora_rank", "lora_alpha",
        "lora_dropout", "max_seq_length", "num_layers", "warmup_steps",
    ],
    TechniqueName.QLORA_SFT: [
        "learning_rate", "iters", "batch_size", "lora_rank", "lora_alpha",
        "lora_dropout", "max_seq_length", "num_layers", "warmup_steps",
    ],
    TechniqueName.DPO: [
        "learning_rate", "iters", "batch_size", "lora_rank", "lora_alpha", "max_seq_length",
    ],
    TechniqueName.ABLITERATION: ["batch_size", "max_seq_length"],
}
