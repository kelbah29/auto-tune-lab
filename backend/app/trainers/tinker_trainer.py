"""Cloud LoRA training via the Tinker API.

Implements two real forward_backward-based recipes, verified directly against the
installed `tinker==0.24.1` SDK source (not just the docs summary):

- LoRA-SFT: standard `forward_backward(loss_fn="cross_entropy")` on masked chat
  Datums (prompt tokens weight=0, assistant-response tokens weight=1). Used for
  behavior_removal and benchmark_improvement goals.
- Preference-weighted LoRA (the "DPO" technique on this backend): the SDK's
  `LossFnType` only exposes {cross_entropy, importance_sampling, ppo, cispo, dro} —
  there is no dedicated pairwise-preference loss. A fully custom loss IS possible via
  `forward_backward_custom` (client-side autograd over returned logprobs), but that
  path requires reference-model logprobs and careful gradient bookkeeping we can't
  fully validate without live API access. Instead we build the preference signal
  directly out of the well-tested cross_entropy primitive: for each pair, submit the
  `chosen` completion with weight=+1 (standard likelihood maximization) and the
  `rejected` completion with weight=-beta (likelihood suppression) in the same
  forward_backward call. This is a lower-risk, still-real gradient-based preference
  method, documented honestly rather than claimed to be textbook DPO.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.core.schemas import DatasetSpec, HyperparamConfig
from app.trainers.base import LogFn, MetricFn, TrainResult

# In-memory registry so the evaluator (running later in the same async process) can
# reuse the live SamplingClient without round-tripping through a persisted checkpoint.
LIVE_SAMPLING_CLIENTS: dict[str, object] = {}


def _build_masked_datum(tokenizer, messages: list[dict], types_mod):
    """Chat messages -> a Datum with prompt tokens masked out of the loss."""
    full_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=False)
    prefix_ids = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True)
    prefix_len = len(prefix_ids)

    input_tokens = full_ids[:-1]
    target_tokens = full_ids[1:]
    weights = [1.0 if (i + 1) >= prefix_len else 0.0 for i in range(len(target_tokens))]

    return types_mod.Datum(
        model_input=types_mod.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs={"target_tokens": target_tokens, "weights": weights},
    )


def _build_weighted_datum(tokenizer, prompt: str, response: str, weight: float, types_mod):
    messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
    full_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=False)
    prefix_ids = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True)
    prefix_len = len(prefix_ids)

    input_tokens = full_ids[:-1]
    target_tokens = full_ids[1:]
    weights = [weight if (i + 1) >= prefix_len else 0.0 for i in range(len(target_tokens))]

    return types_mod.Datum(
        model_input=types_mod.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs={"target_tokens": target_tokens, "weights": weights},
    )


def _read_jsonl(path: str) -> list[dict]:
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _extract_loss(fwdbwd_result) -> float | None:
    """Best-effort loss extraction.

    `ForwardBackwardOutput` doesn't document a stable top-level `.loss` field (its
    dataclass exposes `metrics: dict[str, float]` and per-datum `loss_fn_outputs`
    instead), and we don't have live API access in this environment to pin the exact
    schema. This is deliberately defensive so a field-name mismatch degrades to an
    unlogged step rather than crashing an otherwise-successful training run.
    """
    try:
        metrics = getattr(fwdbwd_result, "metrics", None) or {}
        for key in ("loss", "mean_loss", "cross_entropy", "nll"):
            if key in metrics:
                return float(metrics[key])
        outputs = getattr(fwdbwd_result, "loss_fn_outputs", None) or []
        values = []
        for out in outputs:
            for key in ("loss", "nll", "neg_logprobs"):
                if key in out:
                    values.extend(out[key].tolist())
                    break
        if values:
            return float(sum(values) / len(values))
    except Exception:
        pass
    return None


class TinkerLoraSftTrainer:
    async def train(
        self,
        base_model: str,
        dataset: DatasetSpec,
        hyperparams: HyperparamConfig,
        run_dir: str,
        log: LogFn,
        metric: MetricFn,
    ) -> TrainResult:
        import tinker
        from tinker import types

        service_client = tinker.ServiceClient()
        training_client = await service_client.create_lora_training_client_async(
            base_model=base_model, rank=hyperparams.lora_rank
        )
        tokenizer = training_client.get_tokenizer()

        rows = _read_jsonl(dataset.train_path) if dataset.train_path else []
        if not rows:
            raise RuntimeError("No training examples available for Tinker LoRA-SFT")
        datums = [_build_masked_datum(tokenizer, r["messages"], types) for r in rows]

        log(f"Tinker LoRA-SFT: {len(datums)} examples, rank={hyperparams.lora_rank}, "
            f"lr={hyperparams.learning_rate}, batch_size={hyperparams.batch_size}")

        batch_size = max(1, hyperparams.batch_size)
        n_steps = hyperparams.iters or max(20, len(datums) * hyperparams.epochs // batch_size)
        final_loss = None

        for step in range(n_steps):
            batch = [datums[(step * batch_size + j) % len(datums)] for j in range(batch_size)]
            fwdbwd_future = await training_client.forward_backward_async(
                data=batch, loss_fn="cross_entropy"
            )
            fwdbwd_result = await fwdbwd_future.result_async()
            optim_future = await training_client.optim_step_async(
                types.AdamParams(learning_rate=hyperparams.learning_rate)
            )
            await optim_future.result_async()

            loss = _extract_loss(fwdbwd_result)
            if loss is not None:
                final_loss = loss
                metric("train_loss", loss, step)
            if step % 5 == 0 or step == n_steps - 1:
                log(f"  step {step + 1}/{n_steps}: loss={loss if loss is not None else 'n/a'}")

        log("Saving weights and creating a sampling client for evaluation...")
        sampling_client = await training_client.save_weights_and_get_sampling_client_async()

        run_id = Path(run_dir).name
        LIVE_SAMPLING_CLIENTS[run_id] = sampling_client

        return TrainResult(
            adapter_path=None,
            model_path=base_model,
            backend_ref=f"tinker:{run_id}",
            final_loss=final_loss,
            extra={"n_steps": n_steps},
        )


class TinkerPreferenceTrainer:
    """The DPO-technique backend on Tinker (see module docstring for the honest scoping)."""

    async def train(
        self,
        base_model: str,
        dataset: DatasetSpec,
        hyperparams: HyperparamConfig,
        run_dir: str,
        log: LogFn,
        metric: MetricFn,
    ) -> TrainResult:
        import tinker
        from tinker import types

        service_client = tinker.ServiceClient()
        training_client = await service_client.create_lora_training_client_async(
            base_model=base_model, rank=hyperparams.lora_rank
        )
        tokenizer = training_client.get_tokenizer()

        pairs_path = str(Path(dataset.train_path).parent / "dpo_pairs.jsonl")
        pairs = _read_jsonl(pairs_path)
        if not pairs:
            raise RuntimeError("No preference pairs available for Tinker preference training")

        beta = float(hyperparams.extra.get("beta", 0.5))
        log(f"Tinker preference-weighted LoRA: {len(pairs)} pairs, rank={hyperparams.lora_rank}, "
            f"lr={hyperparams.learning_rate}, beta={beta}")

        batch_pairs = max(1, hyperparams.batch_size // 2 or 1)
        n_steps = hyperparams.iters or max(20, len(pairs) // batch_pairs)
        final_loss = None

        for step in range(n_steps):
            batch = [pairs[(step * batch_pairs + j) % len(pairs)] for j in range(batch_pairs)]
            datums = []
            for p in batch:
                datums.append(_build_weighted_datum(tokenizer, p["prompt"], p["chosen"], 1.0, types))
                datums.append(_build_weighted_datum(tokenizer, p["prompt"], p["rejected"], -beta, types))

            fwdbwd_future = await training_client.forward_backward_async(
                data=datums, loss_fn="cross_entropy"
            )
            fwdbwd_result = await fwdbwd_future.result_async()
            optim_future = await training_client.optim_step_async(
                types.AdamParams(learning_rate=hyperparams.learning_rate)
            )
            await optim_future.result_async()

            loss = _extract_loss(fwdbwd_result)
            if loss is not None:
                final_loss = loss
                metric("train_loss", loss, step)
            if step % 5 == 0 or step == n_steps - 1:
                log(f"  step {step + 1}/{n_steps}: loss={loss if loss is not None else 'n/a'}")

        log("Saving weights and creating a sampling client for evaluation...")
        sampling_client = await training_client.save_weights_and_get_sampling_client_async()

        run_id = Path(run_dir).name
        LIVE_SAMPLING_CLIENTS[run_id] = sampling_client

        return TrainResult(
            adapter_path=None,
            model_path=base_model,
            backend_ref=f"tinker:{run_id}",
            final_loss=final_loss,
            extra={"n_steps": n_steps, "beta": beta},
        )
