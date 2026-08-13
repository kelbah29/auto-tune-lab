"""Dispatches dataset resolution by objective_type and persists the result as
mlx-lm-compatible JSONL (chat format) under runs/<id>/dataset/, plus an eval-holdout
file consumed later by the matching evaluator.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from app.core.run_store import run_dir
from app.core.schemas import DatasetSpec, Goal, ObjectiveType, TechniqueName
from app.datasets import hf_search, swebench, synth_behavior, xstest_uncensor
from app.datasets.seed_prompts import GENERAL_SEED_PROMPTS

LogFn = Callable[[str], None]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _split_train_val(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    if not rows:
        return [], []
    n_val = max(1, len(rows) // 6)
    return rows[n_val:], rows[:n_val]


async def resolve_dataset(
    run_id: str,
    goal: Goal,
    local_model_path: str | None,
    log: LogFn,
    technique_name: TechniqueName | None = None,
) -> DatasetSpec:
    d = run_dir(run_id) / "dataset"
    d.mkdir(exist_ok=True)

    if goal.objective_type == ObjectiveType.BEHAVIOR_REMOVAL:
        examples, eval_prompts = await synth_behavior.build_behavior_dataset(
            goal, local_model_path, log
        )
        train_rows, val_rows = _split_train_val([ex.model_dump() for ex in examples])
        _write_jsonl(d / "train.jsonl", train_rows)
        _write_jsonl(d / "valid.jsonl", val_rows)
        (d / "eval_holdout.json").write_text(json.dumps({"prompts": eval_prompts}, indent=2))
        return DatasetSpec(
            source="synthetic",
            description=(
                f"Rejection-sampled SFT examples enforcing: never produce "
                f"'{goal.negative_constraint_token}'"
            ),
            num_train=len(train_rows),
            num_val=len(val_rows),
            num_eval_holdout=len(eval_prompts),
            train_path=str(d / "train.jsonl"),
            val_path=str(d / "valid.jsonl"),
            eval_holdout_path=str(d / "eval_holdout.json"),
        )

    if goal.objective_type == ObjectiveType.UNCENSOR:
        pairs, eval_prompts, harmful_holdout = await xstest_uncensor.build_uncensor_data(
            local_model_path, log, build_training_pairs=(technique_name != TechniqueName.ABLITERATION)
        )
        _write_jsonl(d / "dpo_pairs.jsonl", pairs)
        (d / "eval_holdout.json").write_text(
            json.dumps({"safe_prompts": eval_prompts, "harmful_prompts": harmful_holdout}, indent=2)
        )
        # SFT-style chosen-only view, used as a fallback if the DPO path isn't available.
        chosen_rows = [
            {
                "messages": [
                    {"role": "user", "content": p["prompt"]},
                    {"role": "assistant", "content": p["chosen"]},
                ]
            }
            for p in pairs
        ]
        train_rows, val_rows = _split_train_val(chosen_rows)
        _write_jsonl(d / "train.jsonl", train_rows)
        _write_jsonl(d / "valid.jsonl", val_rows)
        return DatasetSpec(
            source="huggingface_hub",
            hf_dataset_id=xstest_uncensor.DATASET_ID,
            description=(
                "XSTest safe-but-refusal-triggering prompts as DPO preference pairs, plus a "
                "genuinely-unsafe holdout used only to verify safety refusals are retained"
            ),
            num_train=len(pairs),
            num_val=0,
            num_eval_holdout=len(eval_prompts) + len(harmful_holdout),
            train_path=str(d / "train.jsonl"),
            val_path=str(d / "valid.jsonl"),
            eval_holdout_path=str(d / "eval_holdout.json"),
        )

    if goal.objective_type == ObjectiveType.BENCHMARK_IMPROVEMENT:
        examples, eval_instances = await swebench.build_swebench_data(log)
        train_rows, val_rows = _split_train_val([ex.model_dump() for ex in examples])
        _write_jsonl(d / "train.jsonl", train_rows)
        _write_jsonl(d / "valid.jsonl", val_rows)
        (d / "eval_holdout.json").write_text(json.dumps({"instances": eval_instances}, indent=2))
        return DatasetSpec(
            source="huggingface_hub",
            hf_dataset_id=swebench.DATASET_ID,
            description="SWE-bench_Pro problem_statement -> golden patch SFT pairs, disjoint eval holdout",
            num_train=len(train_rows),
            num_val=len(val_rows),
            num_eval_holdout=len(eval_instances),
            train_path=str(d / "train.jsonl"),
            val_path=str(d / "valid.jsonl"),
            eval_holdout_path=str(d / "eval_holdout.json"),
        )

    # custom / fallback
    examples, eval_prompts, dataset_id = await hf_search.build_generic_dataset(
        goal.behavior_description or goal.raw_prompt, log
    )
    train_rows, val_rows = _split_train_val([ex.model_dump() for ex in examples])
    _write_jsonl(d / "train.jsonl", train_rows)
    _write_jsonl(d / "valid.jsonl", val_rows)
    eval_prompts = eval_prompts or GENERAL_SEED_PROMPTS[:10]
    (d / "eval_holdout.json").write_text(json.dumps({"prompts": eval_prompts}, indent=2))
    return DatasetSpec(
        source="huggingface_hub" if dataset_id else "curated_builtin",
        hf_dataset_id=dataset_id,
        description=(
            f"Custom-goal dataset from Hub search ({dataset_id})"
            if dataset_id
            else "No matching Hub dataset found; used built-in seed prompts as eval-only fallback"
        ),
        num_train=len(train_rows),
        num_val=len(val_rows),
        num_eval_holdout=len(eval_prompts),
        train_path=str(d / "train.jsonl"),
        val_path=str(d / "valid.jsonl"),
        eval_holdout_path=str(d / "eval_holdout.json"),
    )
