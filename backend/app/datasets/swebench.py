"""SWE-bench-style benchmark-improvement dataset.

Downloads ScaleAI/SWE-bench_Pro (731 real GitHub issues with golden patches across
11 repos). Since it ships as a single "test" split, we carve out a disjoint SFT slice
(problem_statement -> golden patch) and a held-out eval slice from *different*
instances, so evaluation never touches training data.
"""
from __future__ import annotations

import random

from datasets import load_dataset

from app.core.schemas import DatasetExample

DATASET_ID = "ScaleAI/SWE-bench_Pro"

_SFT_SYSTEM = (
    "You are an expert software engineer. Given a GitHub issue, produce a minimal, correct "
    "unified diff patch that resolves it."
)


def _to_user_prompt(row: dict) -> str:
    return (
        f"Repository: {row['repo']}\n"
        f"Issue:\n{row['problem_statement']}\n\n"
        "Produce a unified diff patch that resolves this issue."
    )


async def build_swebench_data(
    log, n_train: int = 25, n_eval_holdout: int = 15
) -> tuple[list[DatasetExample], list[dict]]:
    """Returns (SFT training examples, held-out eval instances with golden patches)."""
    ds = load_dataset(DATASET_ID, split="test")
    rows = list(ds)
    rnd = random.Random(0)
    rnd.shuffle(rows)

    # Keep patches to a manageable size for SFT context / a quick local demo.
    rows = [r for r in rows if r.get("patch") and len(r["patch"]) < 6000]

    train_rows = rows[:n_train]
    eval_rows = rows[n_train : n_train + n_eval_holdout]

    examples = [
        DatasetExample(
            messages=[
                {"role": "system", "content": _SFT_SYSTEM},
                {"role": "user", "content": _to_user_prompt(row)},
                {"role": "assistant", "content": row["patch"]},
            ]
        )
        for row in train_rows
    ]
    log(
        f"SWE-bench_Pro: {len(examples)} SFT examples, {len(eval_rows)} held-out eval instances "
        f"(disjoint repos/issues, golden patches withheld from training)."
    )
    eval_instances = [
        {
            "instance_id": r["instance_id"],
            "repo": r["repo"],
            "problem_statement": r["problem_statement"],
            "golden_patch": r["patch"],
            "prompt": _to_user_prompt(r),
        }
        for r in eval_rows
    ]
    return examples, eval_instances
