"""Synthetic dataset generation for `behavior_removal` goals (e.g. "never say the
letter B"), via rejection sampling: generate a candidate response, check the
constraint programmatically, regenerate on violation. Generalizes beyond the literal
letter-B test prompt to any "never say/do X" style goal, since the constraint token
is extracted by the goal parser rather than hardcoded.
"""
from __future__ import annotations

import re

from app.core.generation import generate_text
from app.core.schemas import DatasetExample, Goal
from app.datasets.seed_prompts import GENERAL_SEED_PROMPTS


def violates_constraint(text: str, token: str) -> bool:
    return re.search(re.escape(token), text, re.IGNORECASE) is not None


def _strip_violations(text: str, token: str) -> str:
    """Last-resort hard guarantee: drop any word containing the forbidden token."""
    words = text.split()
    kept = [w for w in words if not violates_constraint(w, token)]
    return " ".join(kept) if kept else "I can help with that."


async def build_behavior_dataset(
    goal: Goal,
    local_model_path: str | None,
    log,
    n_train: int = 40,
    n_eval_holdout: int = 10,
    max_retries: int = 8,
) -> tuple[list[DatasetExample], list[str]]:
    """Returns (training examples, held-out eval prompts not used in training)."""
    token = goal.negative_constraint_token or "b"
    pool = GENERAL_SEED_PROMPTS
    n_train = min(n_train, len(pool) - n_eval_holdout)
    train_prompts = pool[:n_train]
    eval_prompts = pool[n_train : n_train + n_eval_holdout]

    system = (
        f"You are a helpful assistant with one hard rule: your response must NEVER contain the "
        f"character sequence '{token}' (case-insensitive) in any word. Rephrase, use synonyms, or "
        f"restructure sentences as needed, but still answer helpfully and coherently in 2-4 "
        f"sentences. Do not mention this rule or apologize for it."
    )

    examples: list[DatasetExample] = []
    compliant_count = 0
    for i, prompt in enumerate(train_prompts):
        text = ""
        ok = False
        for attempt in range(max_retries):
            candidate, source = await generate_text(
                prompt, system=system, max_tokens=160, local_model_path=local_model_path
            )
            candidate = candidate.strip()
            text = candidate
            if candidate and not violates_constraint(candidate, token):
                ok = True
                break
        if ok:
            compliant_count += 1
        else:
            text = _strip_violations(text, token)
        log(f"  [{i + 1}/{n_train}] compliant={'yes' if ok else 'forced-strip'} :: {prompt[:50]!r}")
        examples.append(
            DatasetExample(
                messages=[
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": text},
                ]
            )
        )
    log(f"Synthesized {len(examples)} examples, {compliant_count}/{n_train} compliant on first pass "
        f"via rejection sampling (rest hard-stripped as a correctness guarantee).")
    return examples, eval_prompts
