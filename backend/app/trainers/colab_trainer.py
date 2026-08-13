"""LoRA-SFT / SimPO training via the official `google-colab-cli` (colab CLI), used as the
preferred cloud backend: genuinely free (a personal Google account, no billing, no HF-style
account-age gate), unlike Tinker (billing-blocked) and HF ZeroGPU (needs a 30-day-old HF
account) which this project also has integrations for — see technique_selector.py's
_cloud_backend() for the preference order.

One-time setup required from the user (can't be done on their behalf — it's an OAuth
consent tied to their own Google identity): run `colab sessions` once in a terminal, open
the printed URL, approve, paste the code back. After that the CLI works non-interactively.

Unlike ZeroGPU (each @spaces.GPU call is an isolated attach/detach), a `colab` session is a
persistent Jupyter kernel: variables set in one `colab exec` call are still there in the
next one on the same session. So `train()` keeps the trained (LoRA-wrapped) model resident
in the kernel's memory for the whole run, and generation later just reuses it directly
in-process, toggling the adapter on/off, instead of saving/reloading weights from disk.
"""
from __future__ import annotations

import asyncio
import json
import os
import shlex
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

from app.core.schemas import DatasetSpec, HyperparamConfig
from app.trainers.base import LogFn, MetricFn, TrainResult

COLAB_BIN = os.environ.get("COLAB_CLI_PATH", "colab")

_TRAIN_SCRIPT_TEMPLATE = '''
import json, threading, time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

_hp = {hp_json}
_examples = {examples_json}
_mode = {mode!r}
_base_model = {base_model!r}

# The colab CLI's kernel-comms channel silently returns whatever output it has
# collected so far if the kernel goes ~25-30s without producing any stdout, even
# though a much larger --timeout was requested (observed empirically: 3/3 runs cut
# off at 25-26s, right as GPU-bound work with no print statements started, e.g. LoRA
# setup + early training steps before the first step%5 progress line). A background
# heartbeat keeps stdout flowing so no quiet stretch ever gets that long.
def _heartbeat():
    while True:
        time.sleep(8)
        print("AUTOTUNELAB_PROGRESS=heartbeat", flush=True)

threading.Thread(target=_heartbeat, daemon=True).start()

_t0 = time.time()
print("AUTOTUNELAB_PROGRESS=loading_tokenizer", flush=True)
tokenizer = AutoTokenizer.from_pretrained(_base_model, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("AUTOTUNELAB_PROGRESS=loading_model", flush=True)
# A free T4 has ~14.5GB usable VRAM. A 7B model's bf16 weights alone (~14.3GB) leave
# effectively zero headroom for anything else, crashing on the very first forward
# pass (observed: CUDA OOM inside RMSNorm before a single training step). 4-bit
# quantization (QLoRA-style) cuts the base model to ~4-5GB, leaving real room for
# LoRA training regardless of model size — applied unconditionally since it costs
# little for the smaller models this also runs (1.5B) and removes the size cliff.
_bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
)
_base = AutoModelForCausalLM.from_pretrained(
    _base_model, quantization_config=_bnb_config, local_files_only=True, device_map={{"": 0}},
)
# prepare_model_for_kbit_training() upcasts every non-4bit param (norms, and for a
# large-vocab model like this one, the multi-GB embedding/lm_head) to fp32 for
# training stability — expensive, and unnecessary here since we don't use gradient
# checkpointing (the other thing that helper sets up). Skipping it verified stable
# via a live memory diagnostic: steady-state ~5.7GB, peak ~9.8GB on a 14.56GB T4.
print("AUTOTUNELAB_PROGRESS=model_loaded", flush=True)
_lora_cfg = LoraConfig(
    r=int(_hp["lora_rank"]), lora_alpha=int(_hp["lora_alpha"]), lora_dropout=0.05,
    bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(_base, _lora_cfg)
model.train()
optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(_hp["learning_rate"]))

def _seq_logprob(prompt, response, max_len):
    # apply_chat_template(tokenize=True) doesn't reliably return a plain list[int] across
    # transformers versions (observed: "Could not infer dtype of tokenizers.Encoding" when
    # fed straight into torch.tensor) — templating to text then tokenizing separately is
    # version-independent.
    prompt_text = tokenizer.apply_chat_template([{{"role": "user", "content": prompt}}], add_generation_prompt=True, tokenize=False)
    full_text = tokenizer.apply_chat_template(
        [{{"role": "user", "content": prompt}}, {{"role": "assistant", "content": response}}], tokenize=False
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"][:max_len]
    prompt_len = min(len(prompt_ids), len(full_ids))
    input_ids = torch.tensor([full_ids], device="cuda")
    logits = model(input_ids=input_ids).logits[:, :-1, :]
    targets = input_ids[:, 1:]
    logprobs = F.log_softmax(logits.float(), dim=-1)
    tok_lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    resp_lp = tok_lp[:, max(prompt_len - 1, 0):]
    return resp_lp.mean() if resp_lp.shape[1] > 0 else torch.tensor(0.0, device="cuda")

step, last_loss = 0, None
target_steps = int(_hp["target_steps"])
max_seconds = float(_hp["max_seconds"])
max_seq_len = int(_hp["max_seq_length"])
batch_size = int(_hp["batch_size"])

if _mode == "sft":
    texts = [tokenizer.apply_chat_template(ex["messages"], tokenize=False) for ex in _examples]
    while step < target_steps and (time.time() - _t0) < max_seconds:
        batch = [texts[(step * batch_size + i) % len(texts)] for i in range(batch_size)]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_len).to("cuda")
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        loss = model(**enc, labels=labels).loss
        loss.backward()
        optim.step()
        optim.zero_grad()
        last_loss = float(loss.item())
        step += 1
        print(f"AUTOTUNELAB_PROGRESS=step_{{step}}_loss_{{last_loss:.4f}}", flush=True)
else:
    beta, gamma = float(_hp["simpo_beta"]), float(_hp["simpo_gamma"])
    while step < target_steps and (time.time() - _t0) < max_seconds:
        ex = _examples[step % len(_examples)]
        lp_c = _seq_logprob(ex["prompt"], ex["chosen"], max_seq_len)
        lp_r = _seq_logprob(ex["prompt"], ex["rejected"], max_seq_len)
        loss = -F.logsigmoid(beta * (lp_c - lp_r) - gamma)
        loss.backward()
        optim.step()
        optim.zero_grad()
        last_loss = float(loss.item())
        step += 1
        print(f"AUTOTUNELAB_PROGRESS=step_{{step}}_loss_{{last_loss:.4f}}", flush=True)

model.eval()
print("AUTOTUNELAB_RESULT_JSON=" + json.dumps({{
    "step": step, "target_steps": target_steps, "loss": last_loss,
    "done": step >= target_steps, "elapsed": time.time() - _t0,
}}))
'''

_GENERATE_SCRIPT_TEMPLATE = """
import json, contextlib, torch
_use_adapter = {use_adapter}
_messages = []
if {system!r}:
    _messages.append({{"role": "system", "content": {system!r}}})
_messages.append({{"role": "user", "content": {user!r}}})
_inputs = tokenizer.apply_chat_template(
    _messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
).to("cuda")
_ctx = contextlib.nullcontext() if _use_adapter else model.disable_adapter()
with torch.no_grad(), _ctx:
    _out = model.generate(
        **_inputs, max_new_tokens={max_tokens}, do_sample=True, temperature=0.4,
        pad_token_id=tokenizer.pad_token_id,
    )
_text = tokenizer.decode(_out[0][_inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("AUTOTUNELAB_GEN_RESULT=" + json.dumps(_text))
"""


@dataclass
class _ProcResult:
    returncode: int
    stdout: str
    stderr: str


def _extract_marker(stdout: str, marker: str):
    for line in stdout.splitlines():
        if line.startswith(marker):
            return json.loads(line[len(marker):])
    raise RuntimeError(f"Colab exec output missing {marker!r} marker; full stdout:\n{stdout}")


async def _colab(args: list[str], timeout: float, log: LogFn | None = None) -> _ProcResult:
    """Runs `colab <args>` with a real, enforced timeout.

    `subprocess.run(..., timeout=X)` run inside `asyncio.to_thread` was observed to NOT
    reliably kill the child on timeout (a long-running `colab exec` kept running well past
    its configured timeout, confirmed via `ps` while the orchestrator sat "stuck"). Using
    `asyncio.create_subprocess_exec` + `asyncio.wait_for` here instead, with an explicit
    `proc.kill()` in the except block, actually terminates the process on timeout.
    """
    proc = await asyncio.create_subprocess_exec(
        COLAB_BIN, *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout + 30)
        result = _ProcResult(proc.returncode, stdout_b.decode(errors="replace"), stderr_b.decode(errors="replace"))
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        result = _ProcResult(-1, "", f"colab {' '.join(args)} timed out after {timeout + 30}s and was killed")
    if result.returncode != 0 and log:
        log(f"colab {' '.join(shlex.quote(a) for a in args)} -> exit {result.returncode}\n{result.stderr[-2000:]}")
    return result


def stop_session_sync(session: str) -> None:
    try:
        subprocess.run([COLAB_BIN, "stop", "-s", session], capture_output=True, text=True, timeout=60)
    except Exception:  # noqa: BLE001
        pass


def stop_all_sessions_sync() -> None:
    """Free-tier Colab allows only ONE concurrent GPU session, so a lingering session (e.g.
    the LLM-judge session from app/core/llm.py, used for dataset synthesis) blocks creating
    a new one for training with a TooManyAssignmentsError. Called before every `colab new`
    here so training always wins the single slot; the judge session just gets recreated
    (paying its cold-load cost again) the next time dataset synthesis needs it.

    Goes straight at the server-side assignment list via the CLI's own Client class rather
    than `colab sessions`/`colab stop -s <name>`, which only know about sessions this CLI
    process itself created — a session whose local record was lost (e.g. this backend process
    got killed/restarted mid-run) still holds the one server-side slot with no local name to
    stop it by, and `colab sessions` prints it as an unparseable `[?]` line in that case.
    """
    try:
        import shutil

        resolved_bin = shutil.which(COLAB_BIN)
        if resolved_bin is None:
            return
        colab_python = str(Path(resolved_bin).resolve().parent / "python")
        script = (
            "from colab_cli.auth import AuthProvider, get_credentials\n"
            "from colab_cli.client import Client, Prod\n"
            "import os\n"
            "creds = get_credentials(os.path.expanduser('~/.colab-cli-oauth-config.json'), provider=AuthProvider.OAUTH2)\n"
            "client = Client(Prod(), creds)\n"
            "for a in client.list_assignments():\n"
            "    client.unassign(a.endpoint)\n"
        )
        r = subprocess.run([colab_python, "-c", script], capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            print(f"[stop_all_sessions_sync] non-zero exit: {r.stderr[-1000:]}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[stop_all_sessions_sync] failed: {e}", flush=True)


class ColabTrainer:
    def __init__(self, mode: str) -> None:
        self.mode = mode  # "sft" or "dpo" (SimPO-style, see module docstring)

    async def train(
        self,
        base_model: str,
        dataset: DatasetSpec,
        hyperparams: HyperparamConfig,
        run_dir: str,
        log: LogFn,
        metric: MetricFn,
    ) -> TrainResult:
        # dataset.train_path always points at the SFT-style {"messages": [...]} view
        # (resolver.py writes a chosen-only fallback there for every objective type);
        # the {"prompt", "chosen", "rejected"} preference pairs this mode's training
        # script actually needs live in a sibling dpo_pairs.jsonl — see the identical
        # derivation in tinker_trainer.py's preference trainer.
        examples_path = (
            Path(dataset.train_path).parent / "dpo_pairs.jsonl"
            if self.mode == "dpo"
            else Path(dataset.train_path)
        )
        examples = [json.loads(line) for line in examples_path.read_text().splitlines() if line.strip()]
        if self.mode == "dpo" and not examples:
            raise RuntimeError(f"No preference pairs available at {examples_path}")
        hp = {
            "learning_rate": hyperparams.learning_rate,
            "lora_rank": hyperparams.lora_rank,
            "lora_alpha": hyperparams.lora_alpha,
            # Qwen2.5-Coder's 152K vocab makes the lm_head's own logits tensor
            # (batch*seq_len*vocab_size, kept full-precision even under 4-bit
            # quantization) a real memory cost on its own — a full 1024-token
            # sequence produces a ~300MB logits tensor per example, which was
            # confirmed to be exactly what tipped a T4 over into OOM (matches the
            # reported allocation size almost exactly: 1*1024*152064*2 bytes ≈
            # 297MB). Capping at 512 halves that worst case.
            "max_seq_length": min(hyperparams.max_seq_length, 512),
            "target_steps": hyperparams.iters or int(hyperparams.extra.get("target_steps", 30)),
            # A free T4 (~14.5GB usable VRAM) OOMs on larger models (e.g. 7B, 152K-vocab)
            # even with 4-bit quantization: bitsandbytes' dequant path materializes full
            # dense weight tensors transiently during forward, and that overhead scales
            # with batch size. batch_size=1 cuts activation/dequant memory ~4x with no
            # correctness cost (fewer examples per step, not a different training
            # algorithm) — verified via a live memory diagnostic against this exact
            # model/config (batch_size=4 OOM'd at ~14.5GB in use before a single step
            # completed).
            "batch_size": min(hyperparams.batch_size, 1),
            "max_seconds": hyperparams.extra.get("max_seconds", 600),
            "simpo_beta": hyperparams.extra.get("simpo_beta", 2.0),
            "simpo_gamma": hyperparams.extra.get("simpo_gamma", 0.5),
        }

        # Free-tier Colab VMs have been observed dying mid-run unpredictably (session
        # duration limits / infra reclaiming the box) — not something retrying a single call
        # fixes, since the whole VM is gone. So the unit of retry here is the whole
        # provision-install-warmup-train sequence with a brand new session each attempt.
        last_error: Exception | None = None
        for attempt in range(3):
            session = "atl_" + uuid.uuid4().hex[:10]
            try:
                return await self._attempt_train(session, base_model, hp, examples, log, metric)
            except Exception as e:  # noqa: BLE001
                last_error = e
                stop_session_sync(session)
                if attempt < 2:
                    log(f"Colab training attempt {attempt + 1}/3 failed ({e}); retrying with a fresh session...")
        raise last_error

    async def _attempt_train(
        self, session: str, base_model: str, hp: dict, examples: list, log: LogFn, metric: MetricFn
    ) -> TrainResult:
        train_timeout = float(hp["max_seconds"]) + 300  # + model load overhead

        # Free tier allows only ONE concurrent GPU session — release whatever's holding it
        # (e.g. the LLM-judge session used for dataset synthesis right before this) so
        # training can actually acquire the slot.
        await asyncio.to_thread(stop_all_sessions_sync)
        from app.core.llm import invalidate_colab_judge_session

        invalidate_colab_judge_session()
        await asyncio.sleep(3)  # let the server-side unassign actually propagate

        log(f"Provisioning Colab T4 session {session}...")
        r = await _colab(["new", "-s", session, "--gpu", "T4"], timeout=180, log=log)
        if r.returncode != 0:
            raise RuntimeError(f"Failed to provision Colab session: {r.stderr[-2000:]}")

        log("Installing peft/accelerate on the VM (torch/transformers ship with Colab's image)...")
        # torchao ships preinstalled on the Colab image at a version older than what
        # peft>=0.13's LoRA dispatch eagerly version-checks (even though we never touch
        # torchao/quantization ourselves) — get_peft_model() raises ImportError otherwise.
        r = await _colab(
            ["install", "-s", session, "peft>=0.13", "accelerate>=0.34", "torchao>=0.16.0", "bitsandbytes>=0.43"],
            timeout=360, log=log,
        )
        if r.returncode != 0:
            raise RuntimeError(f"Failed to install packages on Colab VM: {r.stderr[-2000:]}")

        # Warm the local HF cache in its own dedicated call so the training script below —
        # which needs local_files_only=True to avoid a separate loading hang, see below —
        # always finds the weights already on disk.
        warmup_script = (
            f"from huggingface_hub import snapshot_download\n"
            f"snapshot_download({base_model!r})\n"
            f"print('AUTOTUNELAB_WARMUP_OK=true')\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(warmup_script)
            warmup_path = f.name
        try:
            log(f"Pre-downloading {base_model} to the VM's HF cache...")
            # 480s was tuned against ~1.5B-3GB models; a 7B model is ~5x the download
            # (~15GB) and can genuinely take longer, not just hang — give it real room.
            r = await _colab(["exec", "-s", session, "-f", warmup_path, "--timeout", "900"], timeout=900, log=log)
            if r.returncode != 0 or "AUTOTUNELAB_WARMUP_OK=true" not in r.stdout:
                raise RuntimeError(f"Failed to pre-download {base_model} on Colab: {r.stderr[-2000:] or r.stdout[-2000:]}")
        finally:
            os.unlink(warmup_path)

        script = _TRAIN_SCRIPT_TEMPLATE.format(
            hp_json=json.dumps(hp), examples_json=json.dumps(examples), mode=self.mode, base_model=base_model,
        )
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(script)
            script_path = f.name

        log(
            f"Training on Colab (mode={self.mode}, ~{hp['target_steps']} steps, "
            f"{hp['max_seconds']}s budget)..."
        )
        try:
            r = await _colab(
                ["exec", "-s", session, "-f", script_path, "--timeout", str(train_timeout)],
                timeout=train_timeout, log=log,
            )
        finally:
            os.unlink(script_path)

        if r.returncode != 0:
            raise RuntimeError(f"Colab training failed: {r.stderr[-2000:]}")

        try:
            result = _extract_marker(r.stdout, "AUTOTUNELAB_RESULT_JSON=")
        except RuntimeError as e:
            # The colab CLI exits 0 even when the executed cell itself raised — it only
            # writes the traceback to stderr (see execution.py's display_output), so a
            # missing-marker error with an empty stdout tail is silent about the real
            # cause unless stderr is surfaced too.
            raise RuntimeError(f"{e}\n--- stderr ---\n{r.stderr[-2000:]}") from e
        log(
            f"Colab training returned: step={result['step']}/{result['target_steps']} "
            f"loss={result['loss']} elapsed={result['elapsed']:.1f}s done={result['done']}"
        )
        if not result["done"]:
            log(
                "Did not reach target_steps within the time budget — adapter reflects "
                "partial training. Raise max_seconds or lower target_steps to fit.",
                kind="info",
            )
        metric("loss", result["loss"] or 0.0, result["step"])

        return TrainResult(
            adapter_path=session,  # the trained model stays resident in this Colab kernel
            model_path=base_model,
            backend_ref=f"colab:{base_model}:{session}",
            final_loss=result["loss"],
            extra={"colab_done": result["done"], "colab_step": result["step"]},
        )
