"""Unified LLM access: prefers Tinker when TINKER_API_KEY is set, otherwise falls back to
the free HF ZeroGPU Space when HF_TOKEN+HF_ZEROGPU_SPACE_ID are set, otherwise callers fall
back to rule-based logic. This is the single "brain" used for goal parsing, synthetic
dataset generation, and LLM-as-judge evaluation, so the platform needs only one of those two
external keys rather than a separate OpenAI/Anthropic key on top.
"""
from __future__ import annotations

import asyncio
import json
import os
import re

# Qwen3-8B (bf16, ~16GB) sits right at a free Colab T4's VRAM ceiling and is slow to
# cold-download; Qwen2.5-3B-Instruct loads several times faster and leaves headroom, while
# still being a perfectly capable judge/dataset-synthesis model for this platform's purposes.
DEFAULT_JUDGE_MODEL = os.environ.get("TINKER_JUDGE_MODEL", "Qwen/Qwen2.5-3B-Instruct")


_COLAB_JUDGE_INIT_SCRIPT = """
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
_judge_tok = AutoTokenizer.from_pretrained({model!r})
if _judge_tok.pad_token is None:
    _judge_tok.pad_token = _judge_tok.eos_token
_judge_model = AutoModelForCausalLM.from_pretrained({model!r}, dtype=torch.bfloat16).to("cuda")
_judge_model.eval()
print("AUTOTUNELAB_JUDGE_READY=true")
"""

_COLAB_COMPLETE_SCRIPT = """
import json, torch
_messages = []
if {system!r}:
    _messages.append({{"role": "system", "content": {system!r}}})
_messages.append({{"role": "user", "content": {user!r}}})
_inputs = _judge_tok.apply_chat_template(_messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda")
with torch.no_grad():
    _out = _judge_model.generate(**_inputs, max_new_tokens={max_tokens}, do_sample=True, temperature={temperature}, pad_token_id=_judge_tok.pad_token_id)
_text = _judge_tok.decode(_out[0][_inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("AUTOTUNELAB_GEN_RESULT=" + json.dumps(_text))
"""


class LLM:
    def __init__(self, model: str = DEFAULT_JUDGE_MODEL):
        self.model = model
        self._tinker_key = os.environ.get("TINKER_API_KEY")
        self._hf_token = os.environ.get("HF_TOKEN")
        self._zerogpu_space = os.environ.get("HF_ZEROGPU_SPACE_ID")
        self._use_colab = os.environ.get("AUTOTUNELAB_USE_COLAB", "").lower() in ("1", "true", "yes")
        self._service_client = None
        self._sampling_client = None
        self._tokenizer = None
        self._colab_session: str | None = None
        self._colab_lock = asyncio.Lock()

    @property
    def _backend(self) -> str | None:
        if os.environ.get("AUTOTUNELAB_LOCAL_ONLY", "").lower() in ("1", "true", "yes"):
            return None
        if self._use_colab:
            return "colab"
        if self._tinker_key:
            return "tinker"
        if self._hf_token and self._zerogpu_space:
            return "hf_zerogpu"
        return None

    @property
    def available(self) -> bool:
        return self._backend is not None

    def _ensure_tinker_clients(self) -> None:
        if self._sampling_client is not None:
            return
        import tinker

        self._service_client = tinker.ServiceClient()
        self._sampling_client = self._service_client.create_sampling_client(base_model=self.model)
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model)

    async def _acomplete_tinker(self, system: str, user: str, max_tokens: int, temperature: float) -> str:
        self._ensure_tinker_clients()
        from tinker import types

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        token_ids = self._tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        prompt = types.ModelInput.from_ints(tokens=token_ids)
        params = types.SamplingParams(max_tokens=max_tokens, temperature=temperature)
        result = await self._sampling_client.sample_async(
            prompt=prompt, num_samples=1, sampling_params=params
        )
        return self._tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True)

    async def _acomplete_zerogpu(self, system: str, user: str, max_tokens: int) -> str:
        from gradio_client import Client

        def _call() -> str:
            client = Client(self._zerogpu_space, hf_token=self._hf_token)
            return client.predict(
                "", self.model, False, system, user, max_tokens, api_name="/generate"
            )

        return await asyncio.to_thread(_call)

    async def _ensure_colab_judge_session(self) -> str:
        """Lazily creates ONE persistent session that keeps the judge model loaded in its
        kernel for the life of this LLM instance, instead of a fresh `colab run` VM (and
        fresh multi-GB model download) per completion — dataset synthesis alone makes one
        call per training example, so that would be both slow and wasteful.
        """
        import tempfile
        import uuid

        from app.trainers.colab_trainer import _colab, _extract_marker, stop_all_sessions_sync, stop_session_sync

        async with self._colab_lock:
            if self._colab_session is not None:
                return self._colab_session
            session = "atl_judge_" + uuid.uuid4().hex[:10]

            # Free tier allows only ONE concurrent session — a stale one from a previous
            # process (crashed/restarted backend, so this instance's own cache was lost)
            # would otherwise block creation with a TooManyAssignmentsError.
            await asyncio.to_thread(stop_all_sessions_sync)
            await asyncio.sleep(3)  # let the server-side unassign actually propagate

            r = await _colab(["new", "-s", session, "--gpu", "T4"], timeout=180)
            if r.returncode != 0:
                raise RuntimeError(f"Failed to provision Colab judge session: {r.stderr[-2000:]}")

            init_script = _COLAB_JUDGE_INIT_SCRIPT.format(model=self.model)
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
                f.write(init_script)
                path = f.name
            try:
                r = await _colab(["exec", "-s", session, "-f", path, "--timeout", "480"], timeout=480)
                if r.returncode != 0 or "AUTOTUNELAB_JUDGE_READY=true" not in r.stdout:
                    raise RuntimeError(
                        f"Failed to load judge model on Colab: {r.stderr[-2000:] or r.stdout[-2000:]}"
                    )
            except BaseException:
                # Free tier allows only ONE concurrent session — an orphaned half-initialized
                # session here would permanently block every future attempt, so always release
                # it before propagating, not just on the happy path.
                await asyncio.to_thread(stop_session_sync, session)
                raise
            finally:
                os.remove(path)

            self._colab_session = session
            return session

    async def _colab_session_alive(self, session: str) -> bool:
        from app.trainers.colab_trainer import _colab

        try:
            r = await _colab(["sessions"], timeout=30)
            return f"[{session}]" in r.stdout
        except Exception:  # noqa: BLE001
            return False

    async def _acomplete_colab(self, system: str, user: str, max_tokens: int, temperature: float) -> str:
        import tempfile

        from app.trainers.colab_trainer import _colab, _extract_marker

        script = _COLAB_COMPLETE_SCRIPT.format(
            system=system, user=user, max_tokens=max_tokens, temperature=temperature
        )

        async def _call(session: str) -> str:
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
                f.write(script)
                path = f.name
            try:
                r = await _colab(["exec", "-s", session, "-f", path, "--timeout", "90"], timeout=90)
            finally:
                os.remove(path)
            if r.returncode != 0:
                raise RuntimeError(f"Colab completion failed: {r.stderr[-2000:]}")
            return _extract_marker(r.stdout, "AUTOTUNELAB_GEN_RESULT=")

        # Free tier's GPU VM can die mid-run (session duration limits, infra reclaiming it)
        # independently of anything the code does — retrying against a session that's
        # actually dead just burns the full timeout 3x for nothing, so after the first
        # failure, check liveness and reprovision immediately instead of blindly retrying
        # the same dead session (this is what turned one bad call into a ~70min stall before
        # this fix: 4 rejection-sampling attempts x 3 blind retries x a long per-call timeout).
        last_error: Exception | None = None
        for attempt in range(3):
            session = await self._ensure_colab_judge_session()
            try:
                return await _call(session)
            except Exception as e:  # noqa: BLE001
                last_error = e
                if attempt < 2:
                    if not await self._colab_session_alive(session):
                        self._colab_session = None  # force reprovision on the next loop iteration
                    else:
                        await asyncio.sleep(5)  # transient queueing noise — same session is fine
        raise last_error

    async def acomplete(
        self, system: str, user: str, max_tokens: int = 512, temperature: float = 0.4
    ) -> str:
        backend = self._backend
        if backend == "colab":
            return await self._acomplete_colab(system, user, max_tokens, temperature)
        if backend == "tinker":
            return await self._acomplete_tinker(system, user, max_tokens, temperature)
        if backend == "hf_zerogpu":
            return await self._acomplete_zerogpu(system, user, max_tokens)
        raise RuntimeError(
            "No LLM backend configured (set AUTOTUNELAB_USE_COLAB=1, TINKER_API_KEY, or "
            "HF_TOKEN+HF_ZEROGPU_SPACE_ID)"
        )

    def complete(self, system: str, user: str, max_tokens: int = 512, temperature: float = 0.4) -> str:
        """Sync convenience wrapper — only safe to call outside a running event loop."""
        return asyncio.run(self.acomplete(system, user, max_tokens=max_tokens, temperature=temperature))

    async def acomplete_json(self, system: str, user: str, max_tokens: int = 512) -> dict | None:
        raw = await self.acomplete(system, user, max_tokens=max_tokens, temperature=0.2)
        return extract_json(raw)


def extract_json(text: str) -> dict | None:
    """Best-effort JSON object extraction from LLM output (handles ```json fences, prose)."""
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fence.group(1) if fence else None
    if candidate is None:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        candidate = brace.group(0) if brace else None
    if candidate is None:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


_default_llm: LLM | None = None


def get_llm() -> LLM:
    global _default_llm
    if _default_llm is None:
        _default_llm = LLM()
    return _default_llm


def invalidate_colab_judge_session() -> None:
    """Call after something else (e.g. ColabTrainer, which needs the free tier's one
    concurrent-session slot for itself) has stopped the judge session out from under this
    singleton, so the next acomplete() call re-provisions instead of talking to a dead one.
    """
    if _default_llm is not None:
        _default_llm._colab_session = None
