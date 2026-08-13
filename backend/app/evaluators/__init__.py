"""Dispatches before/after evaluation by objective_type, reading the eval-holdout
file produced by the matching dataset resolver.
"""
from __future__ import annotations

import json

from app.core.model_access import GenFn
from app.core.schemas import DatasetSpec, EvalReport, Goal, ObjectiveType
from app.evaluators.generic_llm_eval import evaluate_generic
from app.evaluators.letter_suppression import evaluate_letter_suppression
from app.evaluators.refusal_rate import evaluate_refusal_rate
from app.evaluators.swebench_proxy import evaluate_swebench_proxy


async def run_evaluation(
    goal: Goal, dataset: DatasetSpec, gen_before: GenFn, gen_after: GenFn, log
) -> tuple[EvalReport, EvalReport]:
    holdout = json.loads(open(dataset.eval_holdout_path).read()) if dataset.eval_holdout_path else {}

    if goal.objective_type == ObjectiveType.BEHAVIOR_REMOVAL:
        prompts = holdout.get("prompts", [])
        token = goal.negative_constraint_token or "b"
        log("Evaluating BEFORE (base model)...")
        before = await evaluate_letter_suppression("before", token, prompts, gen_before, log)
        log("Evaluating AFTER (fine-tuned model)...")
        after = await evaluate_letter_suppression("after", token, prompts, gen_after, log)
        return before, after

    if goal.objective_type == ObjectiveType.UNCENSOR:
        safe = holdout.get("safe_prompts", [])
        harmful = holdout.get("harmful_prompts", [])
        log("Evaluating BEFORE (base model)...")
        before = await evaluate_refusal_rate("before", safe, harmful, gen_before, log)
        log("Evaluating AFTER (fine-tuned/abliterated model)...")
        after = await evaluate_refusal_rate("after", safe, harmful, gen_after, log)
        return before, after

    if goal.objective_type == ObjectiveType.BENCHMARK_IMPROVEMENT:
        instances = holdout.get("instances", [])
        log("Evaluating BEFORE (base model)...")
        before = await evaluate_swebench_proxy("before", instances, gen_before, log)
        log("Evaluating AFTER (fine-tuned model)...")
        after = await evaluate_swebench_proxy("after", instances, gen_after, log)
        return before, after

    prompts = holdout.get("prompts", [])
    log("Evaluating BEFORE (base model)...")
    before = await evaluate_generic("before", goal.behavior_description or goal.raw_prompt, prompts, gen_before, log)
    log("Evaluating AFTER (fine-tuned model)...")
    after = await evaluate_generic("after", goal.behavior_description or goal.raw_prompt, prompts, gen_after, log)
    return before, after
