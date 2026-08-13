"""Unified text-generation helper used for dataset synthesis and LLM-judging.

Prefers Tinker (when TINKER_API_KEY is set) since it can sample from a large,
capable instruct model regardless of what's being fine-tuned. Falls back to local
mlx-lm generation (Apple Silicon) using the run's own target/base model so the
platform still produces real, non-templated text end-to-end with zero external key
— this is what powers the local smoke test.
"""
from __future__ import annotations

import asyncio
from functools import lru_cache

from app.core.llm import get_llm


@lru_cache(maxsize=2)
def _load_local(model_path: str):
    from mlx_lm import load

    return load(model_path)


async def _local_generate(model_path: str, system: str, user: str, max_tokens: int) -> str:
    def _run() -> str:
        from mlx_lm import generate as mlx_generate
        from mlx_lm.sample_utils import make_sampler

        model, tokenizer = _load_local(model_path)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return mlx_generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=0.7),
            verbose=False,
        )

    return await asyncio.to_thread(_run)


async def generate_text(
    user: str, system: str = "", max_tokens: int = 256, local_model_path: str | None = None
) -> tuple[str, str]:
    """Returns (text, source) where source is 'tinker', 'local:<model>', or
    'local:<model>(tinker_failed)' if Tinker was configured but errored at call time
    (e.g. billing block, rate limit) — a live key is not a guarantee of a live call,
    so this degrades the same way the "no key" case does rather than crashing the run.
    """
    llm = get_llm()
    if llm.available:
        try:
            text = await llm.acomplete(system, user, max_tokens=max_tokens)
            return text, "tinker"
        except Exception:
            if not local_model_path:
                raise
    if local_model_path:
        text = await _local_generate(local_model_path, system, user, max_tokens=max_tokens)
        source = "local:" + local_model_path + ("(tinker_failed)" if llm.available else "")
        return text, source
    raise RuntimeError("No generation backend available (no TINKER_API_KEY and no local model given)")
