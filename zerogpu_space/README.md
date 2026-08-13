---
title: AutoTuneLab ZeroGPU Trainer
emoji: 🧪
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.31.0
app_file: app.py
pinned: false
short_description: Free ZeroGPU LoRA/SimPO trainer backend for AutoTuneLab, no billing required.
---

# AutoTuneLab ZeroGPU Trainer

Headless training/generation backend called by AutoTuneLab's `HFZeroGPUTrainer` over the
Gradio API (`gradio_client`), used when a target model doesn't fit in local RAM and the user
doesn't want to add billing to a paid provider (e.g. Tinker). Free HF accounts get 5 minutes
of ZeroGPU time per day — see `train`'s `max_seconds` hyperparameter, which self-limits each
job to fit that budget in a single GPU attach (repeated model reloads across many small calls
would burn the budget on I/O instead of training).

Two API endpoints:
- `train(job_id, mode, base_model, dataset_jsonl, hp_json)` — `mode` is `"sft"` (next-token
  loss on chat examples) or `"dpo"` (SimPO-style reference-free preference loss on
  prompt/chosen/rejected triples — no reference model, to fit the VRAM/time budget). Saves a
  LoRA adapter under `/tmp/autotunelab_jobs/<job_id>/adapter`.
- `generate(job_id, base_model, use_adapter, system, user, max_tokens)` — generates from the
  base model, optionally with the job's saved adapter applied, for before/after eval.
