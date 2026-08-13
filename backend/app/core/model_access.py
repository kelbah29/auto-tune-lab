"""Builds unified before/after text-generation callables regardless of which trainer
backend produced the fine-tuned model, so evaluators stay backend-agnostic: they just
call `gen(system, user) -> text` twice (before/after) and compare outputs.
"""
from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Awaitable, Callable

GenFn = Callable[[str, str], Awaitable[str]]


@lru_cache(maxsize=4)
def _load_mlx(model_path: str, adapter_path: str | None):
    from mlx_lm import load

    return load(model_path, adapter_path=adapter_path) if adapter_path else load(model_path)


@lru_cache(maxsize=2)
def _load_hf(model_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype="auto").to(device)
    return model, tokenizer, device


def mlx_generator(model_path: str, adapter_path: str | None = None, max_tokens: int = 150) -> GenFn:
    async def _gen(system: str, user: str) -> str:
        def _run() -> str:
            from mlx_lm import generate
            from mlx_lm.sample_utils import make_sampler

            model, tokenizer = _load_mlx(model_path, adapter_path)
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": user})
            prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            return generate(
                model, tokenizer, prompt=prompt, max_tokens=max_tokens,
                sampler=make_sampler(temp=0.4), verbose=False,
            )

        return await asyncio.to_thread(_run)

    return _gen


def hf_generator(model_path: str, max_tokens: int = 150) -> GenFn:
    async def _gen(system: str, user: str) -> str:
        def _run() -> str:
            import torch

            model, tokenizer, device = _load_hf(model_path)
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": user})
            inputs = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
            ).to(device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.4,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            new_tokens = out[0][inputs["input_ids"].shape[1]:]
            return tokenizer.decode(new_tokens, skip_special_tokens=True)

        return await asyncio.to_thread(_run)

    return _gen


def tinker_generator(sampling_client, tokenizer, max_tokens: int = 150) -> GenFn:
    async def _gen(system: str, user: str) -> str:
        from tinker import types

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        token_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        prompt = types.ModelInput.from_ints(tokens=token_ids)
        params = types.SamplingParams(max_tokens=max_tokens, temperature=0.4)
        result = await sampling_client.sample_async(prompt=prompt, num_samples=1, sampling_params=params)
        return tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True)

    return _gen


def colab_generator(session: str, use_adapter: bool, max_tokens: int = 150) -> GenFn:
    """Reuses the model already resident in the Colab session's kernel (see
    app/trainers/colab_trainer.py) rather than reloading weights — `use_adapter=False`
    temporarily disables the trained LoRA adapter via peft's `disable_adapter()` context
    manager to get base-model-equivalent output from the same loaded model.
    """
    from app.trainers.colab_trainer import _colab, _GENERATE_SCRIPT_TEMPLATE, _extract_marker

    async def _gen(system: str, user: str) -> str:
        script = _GENERATE_SCRIPT_TEMPLATE.format(
            use_adapter=use_adapter, system=system, user=user, max_tokens=max_tokens
        )

        async def _call() -> str:
            import os
            import tempfile

            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
                f.write(script)
                path = f.name
            try:
                r = await _colab(["exec", "-s", session, "-f", path, "--timeout", "90"], timeout=90)
            finally:
                os.remove(path)
            if r.returncode != 0:
                raise RuntimeError(f"Colab generate failed: {r.stderr[-2000:]}")
            return _extract_marker(r.stdout, "AUTOTUNELAB_GEN_RESULT=")

        async def _session_alive() -> bool:
            r = await _colab(["sessions"], timeout=30)
            return f"[{session}]" in r.stdout

        # Free-tier Colab shares GPU capacity, so an occasional call queues long enough to
        # time out even generously — that's noise, retry. But if the VM itself died mid-run,
        # there's no session to reprovision here (the trained adapter only exists in that
        # kernel's memory — see colab_trainer.py) so fail fast with a clear cause instead of
        # blindly burning 3x the full timeout against a session that's provably gone.
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                return await _call()
            except Exception as e:  # noqa: BLE001
                last_error = e
                if attempt < 2:
                    if not await _session_alive():
                        raise RuntimeError(
                            f"Colab training session {session!r} died mid-eval (VM was reclaimed "
                            "or hit a session duration limit) — the trained adapter lived only in "
                            "that session's memory and can't be recovered without retraining."
                        ) from e
                    await asyncio.sleep(5)
        raise last_error

    return _gen


def hf_zerogpu_generator(
    base_model: str, job_id: str | None, use_adapter: bool, max_tokens: int = 150
) -> GenFn:
    """job_id/use_adapter select whether to hit the base model or the job's saved LoRA
    adapter on the ZeroGPU Space — see zerogpu_space/app.py's `generate` endpoint.
    """
    import os

    async def _gen(system: str, user: str) -> str:
        def _call() -> str:
            from gradio_client import Client

            space_id = os.environ["HF_ZEROGPU_SPACE_ID"]
            client = Client(space_id, hf_token=os.environ.get("HF_TOKEN"))
            return client.predict(
                job_id or "", base_model, use_adapter, system, user, max_tokens, api_name="/generate"
            )

        return await asyncio.to_thread(_call)

    return _gen
