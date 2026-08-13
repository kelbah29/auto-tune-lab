"""Uncensor / over-refusal dataset.

Pulls XSTest (natolambert/xstest-v2-copy) from the Hub: it pairs safe-but-refusal-
triggering prompts (homonyms, figurative language, safe contexts, ...) with their
genuinely-unsafe "contrast" counterparts. We use the safe set to build DPO-style
preference pairs (chosen = direct helpful answer, rejected = canned refusal) for the
Tinker path, and as the over-refusal eval set for ANY technique (including local
abliteration, which manages its own internal ablation prompts via `heretic` and
doesn't need this training data). The genuinely-unsafe "contrast" prompts are never
trained on — they're held out purely to verify real safety refusals survive.
"""
from __future__ import annotations

import random

from datasets import load_dataset

from app.core.generation import generate_text

DATASET_ID = "natolambert/xstest-v2-copy"

_REFUSAL_TEMPLATES = [
    "I'm sorry, but I can't help with that request.",
    "I can't assist with that, it's not something I'm able to do.",
    "I'm not able to help with this request.",
]


async def build_uncensor_data(
    local_model_path: str | None,
    log,
    n_train: int = 30,
    n_eval_holdout: int = 15,
    n_harmful_holdout: int = 15,
    build_training_pairs: bool = True,
) -> tuple[list[dict], list[str], list[str]]:
    """Returns (dpo_pairs, safe_eval_prompts, harmful_holdout_prompts).

    `build_training_pairs=False` skips the (slow, generation-heavy) preference-pair
    construction — used when the selected technique is local abliteration, which
    manages its own harmful/harmless prompt sets internally via `heretic` and only
    needs the eval-holdout prompts from this function.
    """
    ds = load_dataset(DATASET_ID, split="prompts")
    safe = [r for r in ds if not r["type"].startswith("contrast_")]
    unsafe = [r for r in ds if r["type"].startswith("contrast_")]

    rnd = random.Random(0)
    rnd.shuffle(safe)
    rnd.shuffle(unsafe)

    train_rows = safe[:n_train] if build_training_pairs else []
    eval_rows = safe[n_train : n_train + n_eval_holdout]
    harmful_holdout = [r["prompt"] for r in unsafe[:n_harmful_holdout]]

    if not build_training_pairs:
        log("Skipping preference-pair generation (technique manages its own training data).")

    pairs: list[dict] = []
    for i, row in enumerate(train_rows):
        prompt = row["prompt"]
        chosen, _src = await generate_text(
            prompt,
            system=(
                "Answer directly and helpfully in 2-3 sentences. This prompt only sounds risky "
                "because of its wording (e.g. a homonym or figurative phrase) but is actually benign."
            ),
            max_tokens=160,
            local_model_path=local_model_path,
        )
        rejected = _REFUSAL_TEMPLATES[i % len(_REFUSAL_TEMPLATES)]
        pairs.append({"prompt": prompt, "chosen": chosen.strip(), "rejected": rejected})
        log(f"  [{i + 1}/{len(train_rows)}] preference pair built ({row['type']}) :: {prompt[:50]!r}")

    eval_prompts = [r["prompt"] for r in eval_rows]
    log(
        f"XSTest: {len(pairs)} train pairs, {len(eval_prompts)} safe held-out eval prompts, "
        f"{len(harmful_holdout)} genuinely-unsafe holdout prompts (never trained on)."
    )
    return pairs, eval_prompts, harmful_holdout
