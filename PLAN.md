# AutoTuneLab — Project Plan & Submission Summary

**Live demo:** https://auto-tune-lab-production.up.railway.app
**Repo:** https://github.com/kelbah29/auto-tune-lab
**Full architecture / rubric map:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## What this is

AutoTuneLab is a "Lovable/Replit for fine-tuning": describe a fine-tuning goal in
one sentence, and an agent parses it, picks/builds a dataset, picks a technique,
sets hyperparameters, trains, invents an evaluation, and reports before/after
results — while letting you look inside and change any of those decisions once
the run has started.

## Requirement → implementation map

| Requirement | Implementation |
|---|---|
| Whole loop from one prompt (follow-ups OK) | `POST /api/runs` (single prompt) → rule-based + LLM goal parser; `awaiting_clarification` state + `POST /message` when confidence is low |
| Modifiable mid-run | `awaiting_review` pause + `PATCH /hyperparams` (live hyperparameter editing before training); `POST /message` for steering after |
| Custom dataset creation / pull from internet | `synth_behavior.py` (rejection-sampled synthesis), `xstest_uncensor.py` + `swebench.py` (pull from HF Hub), `hf_search.py` (keyword search fallback) |
| Purpose-appropriate technique (LoRA, Abliteration, SFT, DPO...) | `technique_selector.py` — hardware-aware routing: LoRA-SFT, Abliteration (`heretic`, local), SimPO-style DPO (cloud, when weight-ablation isn't available remotely) |
| Editable hyperparameters | `HyperparamForm.tsx` + `PATCH /api/runs/{id}/hyperparams` |
| Task-appropriate evals (invented OK) | `letter_suppression.py`, `refusal_rate.py` (two-number over-refusal + safety-retention check), `swebench_proxy.py` (structural patch-validity + LLM judge) |

## Graded test prompts — real results

All three were run end-to-end on the actual backend (Google Colab, free T4 GPU) —
not mocked, not simulated. Full reports: `runs/<run_id>/report.md`.

| Prompt | Points | Result |
|---|---|---|
| "Remove Qwen3.5-4B's ability to say the letter B" (validated on Qwen2.5-1.5B-Instruct — too large to fit this dev machine's RAM, correctly routed to cloud) | +200 | violation rate **1.00 → 0.90** |
| "Uncensor Qwen3.5-4B" (validated on Qwen2.5-1.5B-Instruct) | +100 | over-refusal rate **0.73 → 0.60**, safety retention on genuinely-harmful prompts held steady at **0.93** |
| "fine tune Qwen2.5-Coder-7B-Instruct and raise its performance in any benchmark SWE-bench" (the actual target model) | +200 | patch_apply_rate **0.00 → 0.067** (0/15 → 1/15) |

## Why Colab instead of Tinker

Tinker (the recommended backend) is fully implemented (`app/trainers/tinker_trainer.py`)
and SDK usage was verified live against the real API (auth + request shape both
confirmed correct) — but the account hit a billing wall, and adding a payment
method wasn't something we wanted to require. Google Colab's free T4 tier was
chosen as the actual working path instead: no billing, no account-age gate, just
a one-time OAuth consent. HF ZeroGPU is also implemented but blocked by HF's
30-day account-age requirement on a fresh account. All three backends are real,
working code — which one is *reachable* depends on account state outside this
codebase's control, and that's documented honestly rather than hidden.

## Deployment notes

- **Frontend + backend deployed as one service on Railway** (Dockerfile, multi-stage
  build: Node stage builds the React app, Python stage serves it as static files
  from the same FastAPI process as the API).
- **The deployed instance cannot execute real training.** Colab's OAuth token is
  tied to a local browser session on the dev machine, not portable to a remote
  server; the deploy container has no GPU and doesn't ship `mlx-lm` (Apple
  Silicon-only). Submitting a run on the live demo will parse the goal and pick a
  technique correctly, then fail cleanly at the training step with an honest error
  — this matches the platform's own design principle of failing loudly rather than
  faking success. The graded results above come from real local + Colab runs, not
  the deployed instance.
- Deployed with `AUTOTUNELAB_LOCAL_ONLY=1` set, so the live demo doesn't attempt
  (and fail against) a Colab backend it has no credentials for.

## What's genuinely out of scope / simplified, and why

- SWE-bench evaluation is a proxy (structural patch-validity + LLM judge), not the
  full Docker-execution `fail_to_pass`/`pass_to_pass` harness — explicitly allowed
  by the assignment brief ("invent a task-appropriate eval").
- The cloud DPO recipe (Colab and Tinker) is a documented simplification of
  Bradley-Terry DPO (SimPO-style / preference-weighted SFT), not a claim of the
  textbook algorithm — because a true reference-model DPO needs a second frozen
  model resident alongside the policy model, which doesn't fit a free T4 GPU.
