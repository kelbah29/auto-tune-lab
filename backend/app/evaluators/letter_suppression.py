"""Before/after evaluator for `behavior_removal` goals: forbidden-token occurrence
rate over held-out prompts, plus an LLM-judge coherence score so "learned to avoid
the token" can be told apart from "degenerated into gibberish to dodge it".
"""
from __future__ import annotations

from app.core.llm import get_llm
from app.core.model_access import GenFn
from app.core.schemas import EvalReport
from app.datasets.synth_behavior import violates_constraint

_JUDGE_SYSTEM = (
    "Rate how coherent and helpful this AI response is on a 1-5 scale (5 = perfectly "
    "coherent and helpful, 1 = incoherent or unhelpful). Respond with ONLY the digit."
)


async def _judge_coherence(prompt: str, response: str) -> float:
    llm = get_llm()
    # Free-tier Colab allows only one concurrent GPU session, and gen() here already
    # holds it (reusing the in-training-session model — see model_access.colab_generator).
    # Provisioning a separate judge session mid-loop would tear that session down via
    # stop_all_sessions_sync(), killing the eval that's still in progress. Coherence is
    # a supplementary diagnostic (violation_rate is the metric that matters), so skip it
    # rather than risk the run.
    if not llm.available or llm._backend == "colab":
        return float("nan")
    try:
        text = await llm.acomplete(
            _JUDGE_SYSTEM, f"Question: {prompt}\nResponse: {response}", max_tokens=5, temperature=0.0
        )
        digit = next((c for c in text if c.isdigit()), None)
        return float(digit) if digit else float("nan")
    except Exception:
        return float("nan")


async def evaluate_letter_suppression(
    label: str, token: str, eval_prompts: list[str], gen: GenFn, log
) -> EvalReport:
    violations = 0
    samples = []
    coherence_scores = []
    for prompt in eval_prompts:
        response = await gen("", prompt)
        bad = violates_constraint(response, token)
        if bad:
            violations += 1
        score = await _judge_coherence(prompt, response)
        if score == score:  # not NaN
            coherence_scores.append(score)
        samples.append(
            {"prompt": prompt, "response": response, "violates": bad, "coherence": score}
        )
        log(f"  [{label}] {'VIOLATES' if bad else 'ok'} :: {prompt[:40]!r} -> {response[:60]!r}")

    n = len(eval_prompts)
    metrics = {
        "violation_rate": violations / n if n else 0.0,
        "violations": float(violations),
        "total": float(n),
    }
    if coherence_scores:
        metrics["avg_coherence"] = sum(coherence_scores) / len(coherence_scores)
    return EvalReport(label=label, metrics=metrics, samples=samples)
