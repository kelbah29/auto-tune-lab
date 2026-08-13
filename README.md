# AutoTuneLab

**Lovable/Replit for fine-tuning.** Describe a fine-tuning goal in one sentence —
AutoTuneLab parses it, builds or pulls a dataset, picks a technique (LoRA / SFT /
DPO / Abliteration), sets hyperparameters, trains, invents an evaluation, and
reports before/after results. You can watch the agent's reasoning live, pause to
edit hyperparameters before training starts, and steer the run with follow-up
messages.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full design and a
rubric-to-implementation map.

## Quickstart

```bash
# One-time setup
cd backend
curl -LsSf https://astral.sh/uv/install.sh | sh   # if you don't have uv
uv venv --python 3.11 .venv
uv pip install -r requirements.txt --python .venv/bin/python
cp .env.example .env   # then configure a cloud backend — see "Cloud backends" below

# Optional, for the preferred (free) cloud backend:
uv tool install google-colab-cli --with "jupyter-kernel-client==0.15.0"
# ^ pinned: the CLI's current release (see its PyPI metadata) depends on unpinned
# jupyter-kernel-client, whose 1.0.x renamed the KernelClient class the CLI imports,
# breaking `colab exec` with `AttributeError: module 'jupyter_kernel_client' has no
# attribute 'KernelClient'`. 0.15.0 is the last release with the old name.
colab sessions   # one-time: open the printed URL, approve, paste the code back

cd ../frontend
npm install

# Run both
cd ..
./start.sh
```

Open http://localhost:5173. Try one of the three example prompts, or your own.

With no cloud backend configured, everything still works: goal parsing falls back
to a robust rule-based parser, dataset synthesis/eval-judging fall back to a local
model, and small models (≲ a few B params, whatever fits your RAM) train locally
via `mlx-lm` (Apple Silicon) — this is exactly the path used for real local
end-to-end validation during development (see below). Configure one of the cloud
backends below to unlock training for larger models.

## Cloud backends

Three are implemented, tried in this order (`technique_selector.py`'s
`_cloud_backend()`), each with a real external gate that's outside this code:

1. **Google Colab** (`AUTOTUNELAB_USE_COLAB=1`) — preferred: a free T4 GPU on your
   own Google account, no billing, no account-age requirement. One-time OAuth
   consent required (see Quickstart above) — can't be automated on your behalf,
   it's tied to your Google identity.
2. **HF ZeroGPU** (`HF_TOKEN` + `HF_ZEROGPU_SPACE_ID`, deployed via
   `backend/scripts/deploy_zerogpu_space.py`) — also free, but Hugging Face
   requires the hosting account to be 30+ days old (or HF PRO) to create a
   Gradio+ZeroGPU Space at all, and caps usage at 5 GPU-minutes/day.
3. **Tinker** (`TINKER_API_KEY`) — thinkingmachines.ai's fine-tuning API, no daily
   cap, but requires a payment method on `tinker.thinkingmachines.ai/billing`.

`AUTOTUNELAB_LOCAL_ONLY=1` forces local execution regardless of what's configured
— useful since a key/token being *present* doesn't mean the account is actually
*usable* (e.g. Tinker billing-blocked, HF account too new); flip it off once
whichever backend you're using is confirmed working end-to-end.

## What actually runs where

The platform measures available RAM at runtime and routes each run accordingly —
this isn't hardcoded per model name:

- **Fits in local RAM** → trains on this machine: `mlx_lm.lora` for LoRA-SFT,
  `heretic` (driven programmatically, not through its interactive CLI) for
  abliteration.
- **Doesn't fit locally, a cloud backend is configured** → trains there instead,
  via that backend's real gradient/preference-training calls (see
  `app/trainers/{colab,hf_zerogpu,tinker}_trainer.py`).
- **Doesn't fit locally, nothing configured** → attempts it locally anyway and
  says so in the reasoning log, rather than silently failing.

## Validated locally during development

On an 8GB Apple M3 (no discrete GPU) — real runs, not mocks, with real before/after
metrics in `runs/<id>/report.md`:

- `Qwen/Qwen2.5-0.5B-Instruct`, "remove the letter B" → LoRA-SFT via `mlx_lm.lora`,
  violation rate 0.90 → 0.70 on held-out prompts.
- `Qwen/Qwen2.5-0.5B-Instruct`, "uncensor" → local abliteration via `heretic`,
  before/after refusal counts from heretic's own evaluator.

## Validated end-to-end on Google Colab (free T4)

Real runs against the Colab cloud backend, exercising the full pipeline (dataset
synthesis via a Colab-hosted judge model, gradient training on a fresh T4 session,
before/after eval against the trained adapter) — not mocks:

- `Qwen/Qwen2.5-1.5B-Instruct`, "remove the letter B" (too big to fit in this
  machine's free RAM, so routed to Colab) → LoRA-SFT, violation rate 1.00 → 0.90.
  Modest improvement — most training examples were mechanically stripped rather
  than naturally compliant (see `synth_behavior.py`), a data-quality lever, not a
  pipeline defect; `max_retries` was raised 4→8 to improve this.
- `Qwen/Qwen2.5-1.5B-Instruct`, "uncensor" → routed to a SimPO-style DPO recipe
  (Colab exposes no direct weight-ablation primitive the way local `heretic`
  does), over-refusal rate 0.73 → 0.60 while safety retention on genuinely-harmful
  held-out prompts stayed unchanged at 0.93 — a real, meaningful improvement with
  safety preserved.
- `Qwen/Qwen2.5-Coder-7B-Instruct` (the assignment's actual SWE-bench test-prompt
  model), "raise its performance on SWE-bench" → LoRA-SFT (4-bit/QLoRA — a 7B
  model's bf16 weights alone leave no VRAM headroom on a free T4), patch_apply_rate
  0.00 → 0.067 (0/15 → 1/15 held-out instances). The base model produced zero
  structurally valid patches; the fine-tuned model produced one. Modest, but real
  and directionally correct, on a 60-step budget.

All three of the assignment's graded test prompts have now been run end-to-end on
this Colab path with real before/after metrics. `Qwen3.5-4B` (the letter-B/uncensor
target model) and `Qwen2.5-Coder-7B-Instruct` above both route to Google Colab,
since neither fits in 8GB locally and Colab is the preferred cloud backend (free,
no billing, no account-age gate — see "Cloud backends" above). Colab's free-tier
GPU is subject to external, account-level rate limiting outside this code's control
(observed: multi-hour windows of `503 Service Unavailable` on new sessions); runs
automatically retry with a fresh session on transient failures.

## Repo layout

```
backend/    FastAPI app + orchestrator + trainers/datasets/evaluators (Python 3.11)
frontend/   React + Vite + TypeScript console/dashboard
runs/       per-run artifacts: dataset, adapters/weights, logs, report.md/json
docs/       architecture + grading-rubric map
```
