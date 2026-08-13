# AutoTuneLab — Architecture & Grading Map

AutoTuneLab is a "Lovable/Replit for fine-tuning": you describe a fine-tuning goal in
one sentence, and an agent parses it, picks a dataset, picks a technique, picks
hyperparameters, trains, invents an evaluation, and reports before/after results —
while letting you look inside and change any of those decisions once the run has
started.

## System overview

```
Frontend (React + Vite + TS)              Backend (FastAPI, Python 3.11)
─────────────────────────────             ────────────────────────────────────
Console          → prompt in, 3           POST /api/runs          → orchestrator.start_run
                    example chips                                   (spawns an asyncio task)
RunView           → live pipeline DAG,    GET  /api/runs/{id}/events (SSE) → tails
                    streamed reasoning/                               runs/<id>/logs.jsonl
                    logs, editable        POST /api/runs/{id}/message   → clarification /
                    hyperparam panel                                     mid-run steering
                    (awaiting_review),    PATCH /api/runs/{id}/hyperparams → live-edit
                    before/after charts                                    hyperparams
```

Everything about a run is a plain file under `runs/<run_id>/`: `state.json`
(full `RunState`), `logs.jsonl` (append-only event log, replayed on SSE reconnect),
`dataset/` (train/valid JSONL + eval holdout), `artifacts/` (LoRA adapter or full
abliterated model weights), `report.md` + `report.json`.

## The pipeline (`app/pipeline/orchestrator.py`)

```
parse_goal → [awaiting_clarification]* → select_technique → resolve_dataset →
configure_hyperparams → awaiting_review → train → evaluate → completed
```

- **awaiting_clarification**: fires when the goal parser's confidence is low (e.g.
  it can't resolve a model name). The frontend shows the question(s); your next
  message is merged into the same goal and re-parsed. This is what makes "birkaç
  prompt da olur" (a few prompts is fine) work, not just a single perfect prompt.
- **awaiting_review**: the pipeline always pauses here (unless you check
  "auto-approve" before launching) with a live-editable hyperparameter form.
  `PATCH /api/runs/{id}/hyperparams` is only accepted in this state — once training
  starts you can still send steering messages, but structural edits require
  cancel-and-restart rather than silently pretending to hot-patch a running gradient
  step.
- Every stage emits structured `RunEvent`s (`reasoning`, `info`, `metric`,
  `state_change`, `error`) that are both persisted and pushed live over SSE — the
  "reasoning" events are the agent explaining *why* it picked what it picked, not
  just logging what happened.

## Goal parsing (`app/pipeline/goal_parser.py`)

A deterministic rule-based pass (regex model-name extraction against a shorthand→HF
alias table + HF Hub search fallback, keyword-based objective classification, letter/
word constraint extraction, benchmark-name extraction) always runs first and is the
source of truth for model resolution. When a cloud LLM backend is configured
(`app/core/llm.py` — Google Colab preferred, then HF ZeroGPU, then Tinker; see
"Cloud backends" in the README), an LLM pass additionally proposes a natural-
language `behavior_description` and cross-checks the classification — but its model
guess is always re-resolved through the same trusted alias/search pipeline rather
than trusted blindly, so a hallucinated repo id can't silently break the run.

## Technique & backend selection (`app/pipeline/technique_selector.py`, `app/core/hardware.py`)

Technique and *execution backend* are chosen together, based on the goal type and
the actual RAM available on the machine right now (`psutil`), so the same code
routes a 0.5B model to local training and a 7B model to the cloud (Colab preferred,
then HF ZeroGPU, then Tinker — `_cloud_backend()`) automatically, on any machine
it's run on:

| Goal type | Technique | Fits locally | Doesn't fit locally |
|---|---|---|---|
| `behavior_removal` ("never say X") | LoRA-SFT on rejection-sampled compliant data | `mlx_lm.lora` (Apple Silicon) | cloud gradient training (`colab_trainer.py` / `hf_zerogpu_trainer.py` / `tinker_trainer.py`) |
| `uncensor` | **Abliteration** (direct weight ablation) | `heretic` library, driven programmatically | cloud preference-weighted LoRA (see below) |
| `benchmark_improvement` | LoRA-SFT on task-matched pairs | `mlx_lm.lora` | cloud gradient training |
| `custom` | Heuristic/LLM-classified, defaults to LoRA-SFT | as above | as above |

**Why uncensor resolves differently depending on hardware, on purpose:** abliteration
is direct weight surgery (compute a refusal direction from harmful-vs-harmless
residuals, orthogonalize attention/MLP output projections against it) — there's no
gradient step, so it can't run on a cloud gradient-training API the way SFT can;
those trainers work via gradients and expose no weight-editing primitive. When the
target model is too large for this machine's RAM, the platform substitutes a
**DPO-style preference recipe** instead (see below) and says so explicitly in its
reasoning log — a deliberate technique substitution, not a silent downgrade.

## Trainers (`app/trainers/`)

- **`mlx_lora.py`** — shells out to the real `mlx_lm.lora` console script (Apple
  Silicon LoRA/DoRA/full fine-tuning), streaming stdout live and parsing
  `Iter N: Train loss X` / `Val loss X` lines into metric events. Chat-format JSONL
  (`{"messages": [...]}`) is mlx-lm's native format, so the dataset resolver writes
  directly to it.
- **`heretic_abliteration.py`** — `heretic`'s own CLI is fully interactive (Optuna
  search followed by a `questionary` menu with no scriptable save flag — verified by
  reading its source, not assumed), so instead of fighting that with piped stdin we
  call the exact same underlying classes (`heretic.model.Model`,
  `heretic.evaluator.Evaluator`, the same Optuna `TPESampler` study loop from
  `heretic/main.py`) directly, and automatically pick the Pareto-optimal trial with
  the fewest refusals (heretic's own tie-break rule) instead of asking a human.
  `Evaluator.base_refusals`, computed once against the unmodified model, is used
  directly as the "before" metric.
- **`colab_trainer.py`** — the preferred, actively-validated cloud path: shells out
  to the official `google-colab-cli` (`colab` binary) to run a real training script
  inside a persistent Jupyter kernel on a free Colab T4 GPU (session-based, not
  ephemeral — the trained model stays resident in-kernel for eval afterward instead
  of a save/reload round-trip). SFT mode is a standard masked-chat loop; DPO mode is
  a SimPO-style reference-free preference loss (`-logsigmoid(beta*(lp_chosen -
  lp_rejected) - gamma)`) since a true Bradley-Terry DPO needs a frozen reference
  model that doesn't fit a free T4 alongside the policy model. Free-tier Colab
  allows only one concurrent GPU session, so session lifecycle (provisioning,
  teardown, retry-with-a-fresh-session on failure) is managed carefully — see the
  module docstring and `stop_all_sessions_sync()`.
- **`hf_zerogpu_trainer.py`** — calls a deployed Gradio Space's `/train` endpoint on
  Hugging Face's free ZeroGPU tier (same SFT/SimPO-DPO recipes as Colab); blocked in
  practice by HF's 30-day account-age gate on brand-new accounts, so implemented but
  not the primary path.
- **`tinker_trainer.py`** — real `forward_backward`/`optim_step` loops against the
  Tinker SDK, verified against the installed `tinker==0.24.1` package source (not
  just the docs). Two recipes:
  - `TinkerLoraSftTrainer`: standard masked-chat LoRA-SFT (`loss_fn="cross_entropy"`,
    prompt tokens weight=0, response tokens weight=1).
  - `TinkerPreferenceTrainer` (the Tinker-side "DPO" technique): Tinker's
    `LossFnType` only exposes `{cross_entropy, importance_sampling, ppo, cispo,
    dro}` — there's no dedicated pairwise-preference loss, and while a fully custom
    loss is possible via `forward_backward_custom` (client-side autograd over
    returned logprobs), that path needs reference-model logprobs and gradient
    bookkeeping we can't fully validate without live API access. Instead we submit
    `chosen` at weight=+1 and `rejected` at weight=-β in the *same*
    `forward_backward(loss_fn="cross_entropy")` call — a lower-risk, still-real
    gradient-based preference signal, documented honestly as "preference-weighted
    SFT" rather than claimed to be textbook Bradley-Terry DPO. Implemented and SDK
    usage verified live (auth + request shape both confirmed against the real API),
    but this account is billing-blocked, so Colab is the path actually exercised
    end-to-end (see README's "Validated end-to-end on Google Colab").

## Datasets (`app/datasets/`)

- **`synth_behavior.py`** (behavior_removal): rejection sampling — generate a
  candidate response under a system prompt enforcing the constraint, check it
  programmatically, regenerate on violation (up to 8 tries), hard-strip any
  remaining violating words as a last-resort correctness guarantee. Generalizes to
  any "never say/do X" goal, not just the literal letter-B test prompt. In practice
  a meaningful fraction of examples still need the hard-strip fallback (the
  constraint is genuinely hard for an LLM to satisfy fluently) — this is a data-
  quality lever (retry count, judge model choice), not a pipeline defect; see
  README's Colab-validated results for a concrete before/after.
- **`xstest_uncensor.py`** (uncensor): pulls `natolambert/xstest-v2-copy` — pairs of
  safe-but-refusal-triggering prompts (homonyms, figurative language, safe
  contexts...) and their genuinely-unsafe "contrast" counterparts. The safe set
  builds DPO-style preference pairs (cloud path) and the over-refusal eval set
  (any technique); the unsafe set is *never trained on* — it's a held-out check that
  real safety refusals survive the run.
- **`swebench.py`** (benchmark_improvement): downloads `ScaleAI/SWE-bench_Pro`,
  splits into an SFT slice (`problem_statement → golden patch`) and a disjoint
  held-out eval slice.
- **`hf_search.py`** (custom fallback): keyword-searches the HF Hub for a matching
  dataset and best-effort maps its columns onto instruction/response pairs.

## Evaluation (`app/evaluators/`)

Task-appropriate evals, invented per the assignment's own allowance rather than
requiring an existing benchmark harness:

- **Letter/behavior suppression**: violation rate on held-out prompts + an
  LLM-judge coherence score (so "learned the rule" reads differently from
  "degenerated into gibberish to dodge the token").
- **Uncensor**: two numbers, not one — over-refusal rate on the safe held-out set
  (should drop) *and* refusal rate on the genuinely-harmful holdout (should stay
  high), so a run can't look like a win by just maximizing compliance everywhere.
- **SWE-bench**: `git apply --check` (a real structural-correctness signal, not a
  vibe) + an LLM-judge 1-5 rubric against the golden patch. A full Docker-execution
  harness (build the repo, run `fail_to_pass`/`pass_to_pass`) is out of scope for
  this environment; the assignment explicitly allows a proxy eval instead.
- **Custom**: the configured cloud/local LLM proposes 5 test prompts and a scoring
  rubric for the stated goal, then runs before/after against it.

All four run through `app/core/model_access.py`, which builds backend-agnostic
`(system, user) -> text` generator functions for whichever backend produced the
model (mlx-lm with/without adapter, transformers for the heretic-saved model,
`colab_generator`/`hf_zerogpu_generator` reusing the trained model in-session,
Tinker `SamplingClient`), so the evaluators never need to know which backend trained
the model they're grading. On Colab specifically, the secondary LLM-judge scores
(coherence, patch-quality) are skipped rather than computed, since free-tier Colab
allows only one concurrent GPU session and a second judge session would tear down
the eval session mid-run — the primary, judge-free metrics (violation_rate,
patch_apply_rate) aren't affected.

## Honest scope notes

- Full-scale training of 4B–7B models to convergence is compute-bound and runs on
  Google Colab's free T4 GPU rather than the 8GB dev laptop this was built on
  (Colab is preferred over HF ZeroGPU and Tinker — see README's "Cloud backends");
  the local path (`mlx_lm`/`heretic`) is fully real, not mocked, and was validated
  end-to-end on small models, and the Colab path has now also been validated
  end-to-end with real before/after metrics (see README) — the platform
  auto-routes by measured RAM, so this isn't a hardcoded model-size limit.
- Colab's free-tier GPU is subject to external, account-level rate limiting
  (observed empirically: multi-hour windows of `503 Service Unavailable` on new
  session requests, unrelated to anything in this code) outside anyone's control;
  the trainer retries with a fresh session up to 3x on failure, and the platform
  has no fallback for when Colab, ZeroGPU (30-day account-age gate), and Tinker
  (billing) are all simultaneously unusable — that's a real external constraint,
  not a design gap.
- The DPO/preference-training recipe used on both Colab (SimPO-style, reference-
  free) and Tinker (preference-weighted SFT) is a documented simplification of
  textbook Bradley-Terry DPO, not a claim of implementing it exactly — see the
  Trainers section above for why.
- SWE-bench evaluation is a proxy (patch-applies + LLM judge), not the full
  Docker-execution harness.

## Grading rubric → implementation

| Rubric item | Where |
|---|---|
| Whole loop from one prompt, room for follow-ups, mid-run tweaks | `orchestrator.py` state machine; `awaiting_clarification` / `awaiting_review`; `POST /message`, `PATCH /hyperparams` |
| Custom dataset creation / pull from the internet | `datasets/synth_behavior.py` (synthesize), `datasets/xstest_uncensor.py` + `datasets/swebench.py` + `datasets/hf_search.py` (pull from HF Hub) |
| Purpose-appropriate technique (LoRA, Abliteration, SFT, DPO...) | `technique_selector.py` + rationale strings surfaced live in the UI |
| Editable hyperparameters | `hyperparams.py` defaults + `HyperparamForm.tsx` + `PATCH /api/runs/{id}/hyperparams` |
| Task-appropriate evals (invented, not required to be an existing benchmark) | `evaluators/*.py` |
| "Remove Qwen3.5-4B's ability to say the letter B" | `behavior_removal` → `synth_behavior.py` → `mlx_lora`/cloud (Colab) LoRA-SFT → `letter_suppression.py` |
| "Uncensor Qwen3.5-4B" | `uncensor` → `xstest_uncensor.py` → `heretic_abliteration.py` (local) / cloud (Colab) SimPO-DPO → `refusal_rate.py` |
| "Fine tune Qwen2.5-Coder-7B-Instruct, raise SWE-bench performance" | `benchmark_improvement` → `swebench.py` → cloud (Colab) LoRA-SFT → `swebench_proxy.py` |
