"""Natural language -> Goal.

A deterministic rule-based pass always runs first (fast, offline, no external
dependency) and is the source of truth for model-name resolution. When a Tinker
API key is available, an LLM pass additionally proposes `behavior_description`,
`reasoning`, and cross-checks the rule-based fields, but its model-name guesses are
always re-resolved through the same alias/HF-search pipeline rather than trusted
blindly — an LLM hallucinating a plausible-but-wrong HF repo id must not silently
break the pipeline.
"""
from __future__ import annotations

import re

from huggingface_hub import HfApi

from app.core.llm import get_llm
from app.core.schemas import Goal, ObjectiveType

# Shorthand model names (as people actually type them) -> canonical HF repo id.
MODEL_ALIASES: dict[str, str] = {
    "qwen3.5-4b": "Qwen/Qwen3.5-4B",
    "qwen35-4b": "Qwen/Qwen3.5-4B",
    "qwen3.5-4b-instruct": "Qwen/Qwen3.5-4B",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen3.5-9b": "Qwen/Qwen3.5-9B",
    "qwen2.5-coder-7b-instruct": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "qwen2.5-coder-7b": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "qwen2.5-coder-1.5b-instruct": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "qwen2.5-coder-0.5b-instruct": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    "qwen2.5-0.5b-instruct": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5-1.5b-instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-3b-instruct": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
    "llama-3.2-1b-instruct": "meta-llama/Llama-3.2-1B-Instruct",
    "llama-3.2-3b-instruct": "meta-llama/Llama-3.2-3B-Instruct",
    "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
}

_UNCENSOR_KEYWORDS = [
    "uncensor", "jailbreak", "de-censor", "decensor", "unalign", "unaligned",
    "remove refusal", "remove refusals", "remove restrictions", "bypass safety",
    "stop refusing", "over-refus", "overrefus", "abliterat",
]
_BEHAVIOR_REMOVAL_PATTERNS = [
    r"\b(ability|able)\s+to\s+(say|use|write|output|produce|generate)\b",
    r"\bnever\s+(say|use|write|output|produce|mention)\b",
    r"\b(stop|prevent|block)\s+.*\b(saying|using|writing|outputting)\b",
    r"\bwithout\s+using\s+the\s+letter\b",
    r"\bavoid\s+(the\s+)?(letter|word)\b",
    r"\bcan'?t\s+say\b",
]
_BENCHMARK_KEYWORDS = [
    "benchmark", "swe-bench", "swebench", "mmlu", "humaneval", "gsm8k", "hellaswag",
    "truthfulqa", "arc-challenge", "bbh", "math benchmark", "pass@1", "passrate",
]
_IMPROVE_KEYWORDS = ["improve", "increase", "raise", "boost", "maximize", "enhance", "better"]

_LETTER_RE = re.compile(r"letter\s+['\"]?([A-Za-z])['\"]?\b", re.IGNORECASE)
_WORD_RE = re.compile(r"(?:word|token)\s+['\"]([^'\"]+)['\"]", re.IGNORECASE)
_MODEL_TOKEN_RE = re.compile(
    r"\b([A-Za-z][A-Za-z0-9]*[-\.][A-Za-z0-9][A-Za-z0-9\.\-]*[Bb](?:-[A-Za-z0-9\-]+)?)\b"
)
_BENCHMARK_NAME_RE = re.compile(
    r"\b(SWE-?bench(?:[-_ ]?Pro|[-_ ]?Verified)?|MMLU|HumanEval|GSM8K|HellaSwag|TruthfulQA|"
    r"ARC-Challenge|BBH|MATH)\b",
    re.IGNORECASE,
)


def _normalize(name: str) -> str:
    return re.sub(r"[\s_]+", "", name.strip().lower())


def resolve_model_name(candidate: str) -> tuple[str | None, float]:
    """Returns (hf_repo_id, confidence) for a shorthand model name."""
    key = _normalize(candidate)
    if key in MODEL_ALIASES:
        return MODEL_ALIASES[key], 1.0
    # already looks like an HF repo id ("org/name")
    if "/" in candidate and " " not in candidate:
        return candidate.strip(), 0.9
    # fuzzy alias match (substring)
    for alias, repo_id in MODEL_ALIASES.items():
        if alias in key or key in alias:
            return repo_id, 0.7
    # last resort: search the HF Hub
    try:
        api = HfApi()
        results = list(api.list_models(search=candidate, limit=5))
        if results:
            return results[0].id, 0.5
    except Exception:
        pass
    return None, 0.0


def _detect_objective(text: str) -> ObjectiveType:
    low = text.lower()
    if any(k in low for k in _UNCENSOR_KEYWORDS):
        return ObjectiveType.UNCENSOR
    if any(re.search(p, low) for p in _BEHAVIOR_REMOVAL_PATTERNS):
        return ObjectiveType.BEHAVIOR_REMOVAL
    if any(k in low for k in _BENCHMARK_KEYWORDS) and any(k in low for k in _IMPROVE_KEYWORDS):
        return ObjectiveType.BENCHMARK_IMPROVEMENT
    if any(k in low for k in _BENCHMARK_KEYWORDS):
        return ObjectiveType.BENCHMARK_IMPROVEMENT
    return ObjectiveType.CUSTOM


def _extract_model_candidate(text: str) -> str | None:
    m = _MODEL_TOKEN_RE.search(text)
    return m.group(1) if m else None


def _rule_based_pass(full_text: str) -> Goal:
    objective = _detect_objective(full_text)

    model_candidate = _extract_model_candidate(full_text)
    hf_id, model_conf = (None, 0.0)
    if model_candidate:
        hf_id, model_conf = resolve_model_name(model_candidate)

    constraint_token = None
    if objective == ObjectiveType.BEHAVIOR_REMOVAL:
        lm = _LETTER_RE.search(full_text)
        wm = _WORD_RE.search(full_text)
        if lm:
            constraint_token = lm.group(1)
        elif wm:
            constraint_token = wm.group(1)

    benchmark = None
    bm = _BENCHMARK_NAME_RE.search(full_text)
    if bm:
        benchmark = bm.group(1)

    confidence = 0.4 + 0.4 * model_conf
    if objective == ObjectiveType.BEHAVIOR_REMOVAL and constraint_token:
        confidence += 0.2
    if objective == ObjectiveType.BENCHMARK_IMPROVEMENT and benchmark:
        confidence += 0.1
    if objective == ObjectiveType.CUSTOM:
        confidence -= 0.1
    confidence = max(0.0, min(1.0, confidence))

    questions = []
    if hf_id is None:
        questions.append(
            "Which exact model should I fine-tune? Please give a Hugging Face repo id "
            "(e.g. Qwen/Qwen2.5-0.5B-Instruct) or a common name like 'Qwen3.5-4B'."
        )
    if objective == ObjectiveType.BEHAVIOR_REMOVAL and not constraint_token:
        questions.append(
            "What exact word/letter/behavior should the model stop producing?"
        )
    if objective == ObjectiveType.CUSTOM:
        questions.append(
            "What's the specific goal — remove a behavior, uncensor/reduce refusals, "
            "or improve performance on a task/benchmark? A one-line description helps."
        )

    return Goal(
        raw_prompt=full_text,
        target_model_alias=model_candidate,
        target_model_hf_id=hf_id,
        objective_type=objective,
        behavior_description=full_text.strip(),
        target_benchmark=benchmark,
        negative_constraint_token=constraint_token,
        confidence=confidence,
        clarification_questions=questions,
        reasoning=(
            f"Rule-based parse: objective={objective.value}, model={hf_id or 'unresolved'} "
            f"(matched token '{model_candidate}')"
            if model_candidate
            else f"Rule-based parse: objective={objective.value}, no model token found"
        ),
    )


_LLM_SYSTEM_PROMPT = """You are the goal-parsing module of an autonomous LLM fine-tuning platform.
Given a user's natural language fine-tuning request, extract structured fields and return ONLY a
JSON object with keys: target_model (string, the model name as the user wrote it), objective_type
(one of "behavior_removal", "uncensor", "benchmark_improvement", "custom"), behavior_description
(a clear one-sentence restatement of exactly what should change about the model's behavior),
target_benchmark (string or null), negative_constraint_token (a single word/letter the model should
stop producing, or null if not applicable), confidence (0-1 float, your confidence that you understood
the goal unambiguously), reasoning (one sentence explaining your classification)."""


async def parse_goal(raw_prompt: str, follow_up_prompts: list[str] | None = None) -> Goal:
    follow_up_prompts = follow_up_prompts or []
    full_text = "\n".join([raw_prompt, *follow_up_prompts]).strip()

    goal = _rule_based_pass(full_text)
    goal.follow_up_prompts = follow_up_prompts

    llm = get_llm()
    if llm.available:
        try:
            parsed = await llm.acomplete_json(_LLM_SYSTEM_PROMPT, full_text, max_tokens=400)
        except Exception:
            parsed = None
        if parsed:
            try:
                llm_objective = ObjectiveType(parsed.get("objective_type"))
            except (ValueError, TypeError):
                llm_objective = None
            if llm_objective:
                goal.objective_type = llm_objective
            if parsed.get("behavior_description"):
                goal.behavior_description = str(parsed["behavior_description"])
            if parsed.get("target_benchmark"):
                goal.target_benchmark = str(parsed["target_benchmark"])
            if parsed.get("negative_constraint_token"):
                goal.negative_constraint_token = str(parsed["negative_constraint_token"])
            if parsed.get("reasoning"):
                goal.reasoning = f"{goal.reasoning} | LLM: {parsed['reasoning']}"
            # Re-resolve the LLM's model guess through the trusted pipeline rather than
            # trusting a raw string it produced.
            llm_model = parsed.get("target_model")
            if llm_model and not goal.target_model_hf_id:
                hf_id, conf = resolve_model_name(str(llm_model))
                if hf_id:
                    goal.target_model_hf_id = hf_id
                    goal.target_model_alias = str(llm_model)
                    goal.confidence = max(goal.confidence, 0.4 + 0.4 * conf)
            if isinstance(parsed.get("confidence"), (int, float)):
                goal.confidence = max(goal.confidence, min(1.0, float(parsed["confidence"])) * 0.9)

    # Recompute whether clarification is still needed after the LLM pass.
    goal.clarification_questions = [
        q for q in goal.clarification_questions
        if not (goal.target_model_hf_id and "exact model" in q)
        and not (goal.negative_constraint_token and "letter/behavior" in q)
    ]
    return goal
