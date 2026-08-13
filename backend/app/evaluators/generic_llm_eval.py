"""LLM-invented evaluation for `custom` goals that don't match a purpose-built
evaluator. The assignment explicitly allows inventing task-appropriate evals rather
than requiring an existing benchmark; this is the general-purpose fallback that does
exactly that via the same Tinker-backed LLM used elsewhere in the pipeline.
"""
from __future__ import annotations

from app.core.llm import get_llm
from app.core.model_access import GenFn
from app.core.schemas import EvalReport
from app.datasets.seed_prompts import GENERAL_SEED_PROMPTS

_PLAN_SYSTEM = (
    "You design lightweight evaluations for fine-tuned language models. Given a "
    "fine-tuning goal, propose exactly 5 short test prompts that would reveal whether "
    "a model achieves that goal, and a one-sentence scoring rubric (1-5 scale) for "
    'judging a response. Return ONLY JSON: {"prompts": ["..."], "rubric": "..."}.'
)

_JUDGE_SYSTEM_TMPL = (
    "Score the following AI response on a 1-5 scale using this rubric: {rubric}\n"
    "Respond with ONLY the digit."
)


async def build_eval_plan(goal_text: str) -> tuple[list[str], str]:
    llm = get_llm()
    if llm.available:
        try:
            parsed = await llm.acomplete_json(_PLAN_SYSTEM, goal_text, max_tokens=400)
        except Exception:
            parsed = None
        if parsed and parsed.get("prompts"):
            return list(parsed["prompts"])[:5], parsed.get("rubric", "Overall quality, 1-5.")
    return GENERAL_SEED_PROMPTS[:5], "General helpfulness and coherence, 1-5."


async def evaluate_generic(
    label: str, goal_text: str, eval_prompts: list[str], gen: GenFn, log
) -> EvalReport:
    prompts, rubric = await build_eval_plan(goal_text)
    if eval_prompts:
        prompts = eval_prompts[:5]
    log(f"[{label}] eval rubric: {rubric}")

    llm = get_llm()
    samples = []
    scores = []
    for prompt in prompts:
        response = await gen("", prompt)
        score = float("nan")
        if llm.available:
            try:
                text = await llm.acomplete(
                    _JUDGE_SYSTEM_TMPL.format(rubric=rubric),
                    f"Prompt: {prompt}\nResponse: {response}",
                    max_tokens=5,
                    temperature=0.0,
                )
                digit = next((c for c in text if c.isdigit()), None)
                score = float(digit) if digit else float("nan")
            except Exception:
                pass
        if score == score:
            scores.append(score)
        samples.append({"prompt": prompt, "response": response, "score": score})
        log(f"  [{label}] score={score} :: {prompt[:40]!r}")

    metrics = {"n": float(len(prompts))}
    if scores:
        metrics["avg_score"] = sum(scores) / len(scores)
    return EvalReport(label=label, metrics=metrics, samples=samples)
