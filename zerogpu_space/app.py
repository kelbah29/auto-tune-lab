"""AutoTuneLab's free ZeroGPU trainer/generator backend — see README.md for the API contract
and the rationale for the SimPO-style (reference-free) preference loss used for "dpo" mode.

Design constraint driving everything here: free-tier ZeroGPU gives 5 minutes of GPU time per
day (https://huggingface.co/docs/hub/spaces-zerogpu). A 4B-7B model reload alone can eat
double-digit seconds, so `train` does the *entire* job (load, LoRA-wrap, train N steps, save)
inside a single `@spaces.GPU` call rather than many small chunked calls that would each pay
the reload cost again.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import gradio as gr
import spaces
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

JOBS_DIR = Path("/tmp/autotunelab_jobs")
JOBS_DIR.mkdir(exist_ok=True)

DEFAULT_HP = {
    "learning_rate": 1e-4,
    "lora_rank": 8,
    "lora_alpha": 16,
    "max_seq_length": 512,
    "target_steps": 30,
    "batch_size": 2,
    "max_seconds": 240,
    "simpo_beta": 2.0,
    "simpo_gamma": 0.5,
}


def _tokenizer_for(base_model: str):
    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _sequence_logprob(model, tokenizer, prompt: str, response: str, max_len: int, device: str) -> torch.Tensor:
    """Average per-token log-prob of `response` conditioned on `prompt`, batch size 1."""
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True
    )
    full_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}], tokenize=True
    )
    full_ids = full_ids[:max_len]
    prompt_len = min(len(prompt_ids), len(full_ids))
    input_ids = torch.tensor([full_ids], device=device)
    out = model(input_ids=input_ids)
    logits = out.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    logprobs = F.log_softmax(logits.float(), dim=-1)
    token_logprobs = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    response_mask_start = max(prompt_len - 1, 0)
    response_logprobs = token_logprobs[:, response_mask_start:]
    if response_logprobs.shape[1] == 0:
        return torch.tensor(0.0, device=device)
    return response_logprobs.mean()


def _get_train_duration(job_id, mode, base_model, dataset_jsonl, hp_json):
    try:
        hp = {**DEFAULT_HP, **json.loads(hp_json)}
    except Exception:
        hp = DEFAULT_HP
    return min(float(hp.get("max_seconds", 240)) + 30.0, 280.0)


@spaces.GPU(duration=_get_train_duration)
def _train_gpu(job_id: str, mode: str, base_model: str, dataset_jsonl: str, hp_json: str) -> str:
    t0 = time.time()
    hp = {**DEFAULT_HP, **json.loads(hp_json)}
    max_seconds = float(hp["max_seconds"])
    target_steps = int(hp["target_steps"])
    max_seq_len = int(hp["max_seq_length"])
    batch_size = int(hp["batch_size"])

    tokenizer = _tokenizer_for(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16).to("cuda")
    lora_cfg = LoraConfig(
        r=int(hp["lora_rank"]), lora_alpha=int(hp["lora_alpha"]), lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.train()
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(hp["learning_rate"]))

    examples = [json.loads(line) for line in dataset_jsonl.strip().splitlines() if line.strip()]
    if not examples:
        raise ValueError("Empty dataset passed to ZeroGPU trainer")

    step, last_loss = 0, None

    if mode == "sft":
        texts = [tokenizer.apply_chat_template(ex["messages"], tokenize=False) for ex in examples]
        while step < target_steps and (time.time() - t0) < max_seconds:
            batch = [texts[(step * batch_size + i) % len(texts)] for i in range(batch_size)]
            enc = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_len
            ).to("cuda")
            labels = enc["input_ids"].clone()
            labels[enc["attention_mask"] == 0] = -100
            loss = model(**enc, labels=labels).loss
            loss.backward()
            optim.step()
            optim.zero_grad()
            last_loss = float(loss.item())
            step += 1

    elif mode == "dpo":
        # Reference-free SimPO-style loss (Meng et al., 2024): reward = beta * avg_logprob,
        # loss = -log_sigmoid(reward_chosen - reward_rejected - gamma). No reference model
        # needed (halves VRAM vs. classic DPO), which matters under ZeroGPU's tight budget.
        beta, gamma = float(hp["simpo_beta"]), float(hp["simpo_gamma"])
        while step < target_steps and (time.time() - t0) < max_seconds:
            ex = examples[step % len(examples)]
            lp_chosen = _sequence_logprob(model, tokenizer, ex["prompt"], ex["chosen"], max_seq_len, "cuda")
            lp_rejected = _sequence_logprob(model, tokenizer, ex["prompt"], ex["rejected"], max_seq_len, "cuda")
            reward_gap = beta * (lp_chosen - lp_rejected) - gamma
            loss = -F.logsigmoid(reward_gap)
            loss.backward()
            optim.step()
            optim.zero_grad()
            last_loss = float(loss.item())
            step += 1
    else:
        raise ValueError(f"Unknown mode {mode!r}, expected 'sft' or 'dpo'")

    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(job_dir / "adapter"))

    del model, optim
    torch.cuda.empty_cache()

    return json.dumps({
        "step": step, "target_steps": target_steps, "loss": last_loss,
        "done": step >= target_steps, "elapsed": time.time() - t0,
        "adapter_path": str(job_dir / "adapter"),
    })


def train(job_id: str, mode: str, base_model: str, dataset_jsonl: str, hp_json: str) -> str:
    return _train_gpu(job_id, mode, base_model, dataset_jsonl, hp_json)


@spaces.GPU(duration=60)
def _generate_gpu(job_id: str, base_model: str, use_adapter: bool, system: str, user: str, max_tokens: int) -> str:
    tokenizer = _tokenizer_for(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16).to("cuda")
    if use_adapter:
        adapter_path = str(JOBS_DIR / job_id / "adapter")
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to("cuda")
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=int(max_tokens), do_sample=True, temperature=0.4,
            pad_token_id=tokenizer.pad_token_id,
        )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    del model
    torch.cuda.empty_cache()
    return text


def generate(job_id: str, base_model: str, use_adapter: bool, system: str, user: str, max_tokens: float) -> str:
    return _generate_gpu(job_id, base_model, bool(use_adapter), system, user, int(max_tokens))


train_iface = gr.Interface(
    fn=train,
    inputs=[
        gr.Textbox(label="job_id"), gr.Textbox(label="mode (sft|dpo)"), gr.Textbox(label="base_model"),
        gr.Textbox(label="dataset_jsonl", lines=10), gr.Textbox(label="hp_json"),
    ],
    outputs=gr.Textbox(label="result_json"),
    api_name="train",
)

generate_iface = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="job_id"), gr.Textbox(label="base_model"), gr.Checkbox(label="use_adapter"),
        gr.Textbox(label="system"), gr.Textbox(label="user"), gr.Number(label="max_tokens", value=150),
    ],
    outputs=gr.Textbox(label="text"),
    api_name="generate",
)

demo = gr.TabbedInterface([train_iface, generate_iface], ["train", "generate"])

if __name__ == "__main__":
    demo.queue().launch()
