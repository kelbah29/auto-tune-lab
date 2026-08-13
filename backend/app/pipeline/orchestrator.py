"""Async pipeline orchestrator: the state machine driving

    parse_goal -> [awaiting_clarification]* -> select_technique -> resolve_dataset ->
    configure_hyperparams -> awaiting_review -> train -> evaluate -> completed

with SSE event emission (via an in-memory pub/sub `EventBus`, backed by the
append-only `runs/<id>/logs.jsonl` for replay on reconnect) and support for
follow-up messages (clarification answers, mid-run steering) and config edits
while paused at `awaiting_review`.
"""
from __future__ import annotations

import asyncio
import traceback

from app.core import run_store
from app.core.cancellation import clear_cancel, request_cancel
from app.core.model_access import (
    GenFn,
    colab_generator,
    hf_generator,
    hf_zerogpu_generator,
    mlx_generator,
    tinker_generator,
)
from app.core.schemas import Backend, Goal, RunEvent, RunState, RunStatus
from app.datasets.resolver import resolve_dataset
from app.evaluators import run_evaluation
from app.pipeline.goal_parser import parse_goal
from app.pipeline.hyperparams import default_hyperparams
from app.pipeline.technique_selector import select_technique
from app.trainers import get_trainer
from app.trainers.tinker_trainer import LIVE_SAMPLING_CLIENTS

APPROVE_SENTINEL = "__APPROVE__"


class EventBus:
    def __init__(self) -> None:
        self._subscribers: dict[str, list[asyncio.Queue]] = {}

    def subscribe(self, run_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.setdefault(run_id, []).append(q)
        return q

    def unsubscribe(self, run_id: str, q: asyncio.Queue) -> None:
        subs = self._subscribers.get(run_id, [])
        if q in subs:
            subs.remove(q)

    def publish(self, run_id: str, event: RunEvent) -> None:
        for q in self._subscribers.get(run_id, []):
            q.put_nowait(event)


BUS = EventBus()
RUN_TASKS: dict[str, asyncio.Task] = {}
_INBOXES: dict[str, asyncio.Queue] = {}


def _inbox(run_id: str) -> asyncio.Queue:
    return _INBOXES.setdefault(run_id, asyncio.Queue())


def _log_fn(run_id: str, stage: str):
    def _log(message: str, kind: str = "log", data: dict | None = None) -> None:
        event = RunEvent(stage=stage, kind=kind, message=message, data=data or {})
        run_store.append_event(run_id, event)
        BUS.publish(run_id, event)

    return _log


def _metric_fn(run_id: str, stage: str):
    def _metric(name: str, value: float, step: int) -> None:
        event = RunEvent(
            stage=stage, kind="metric", message=f"{name}={value}",
            data={"name": name, "value": value, "step": step},
        )
        run_store.append_event(run_id, event)
        BUS.publish(run_id, event)

    return _metric


def _set_status(state: RunState, status: RunStatus, run_id: str, stage: str) -> None:
    state.status = status
    state.touch()
    run_store.save_state(state)
    _log_fn(run_id, stage)(f"Status -> {status.value}", kind="state_change", data={"status": status.value})


async def start_run(raw_prompt: str, auto_approve: bool = False) -> RunState:
    state = RunState(goal=Goal(raw_prompt=raw_prompt), auto_approve=auto_approve)
    run_store.save_state(state)
    _inbox(state.run_id)
    task = asyncio.create_task(_run_pipeline(state.run_id))
    RUN_TASKS[state.run_id] = task
    return state


async def add_message(run_id: str, message: str) -> None:
    await _inbox(run_id).put(message)


async def approve_review(run_id: str) -> None:
    await add_message(run_id, APPROVE_SENTINEL)


async def _wait_for_message(run_id: str) -> str:
    return await _inbox(run_id).get()


def patch_hyperparams(run_id: str, updates: dict) -> RunState:
    state = run_store.load_state(run_id)
    if state.status != RunStatus.AWAITING_REVIEW:
        raise ValueError(
            f"Cannot edit hyperparameters while status={state.status.value}. "
            "Cancel and restart, or wait for the next awaiting_review pause."
        )
    if state.hyperparams is None:
        raise ValueError("No hyperparameters to edit yet.")
    current = state.hyperparams.model_dump()
    extra = {**current.get("extra", {}), **updates.pop("extra", {})}
    current.update(updates)
    current["extra"] = extra
    state.hyperparams = state.hyperparams.model_validate(current)
    state.touch()
    run_store.save_state(state)
    _log_fn(run_id, "review")(f"Hyperparameters updated: {updates}", kind="info")
    return state


def cancel_run(run_id: str) -> None:
    request_cancel(run_id)  # stops thread-pool-bound work (e.g. heretic) that Task.cancel() can't reach
    task = RUN_TASKS.get(run_id)
    if task and not task.done():
        task.cancel()
    state = run_store.load_state(run_id)
    state.status = RunStatus.CANCELLED
    state.touch()
    run_store.save_state(state)
    _log_fn(run_id, "pipeline")("Run cancelled.", kind="state_change", data={"status": "cancelled"})


async def _build_generators(state: RunState, train_result) -> tuple[GenFn, GenFn]:
    backend = state.technique.backend
    base_model = state.goal.target_model_hf_id

    if backend == Backend.MLX_LOCAL:
        return (
            mlx_generator(base_model, adapter_path=None),
            mlx_generator(base_model, adapter_path=train_result.adapter_path),
        )
    if backend == Backend.HERETIC_LOCAL:
        return hf_generator(base_model), hf_generator(train_result.model_path)
    if backend == Backend.TINKER:
        import tinker
        from transformers import AutoTokenizer

        service_client = tinker.ServiceClient()
        base_sampling = service_client.create_sampling_client(base_model=base_model)
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        after_sampling = LIVE_SAMPLING_CLIENTS.get(state.run_id)
        if after_sampling is None:
            raise RuntimeError("No live Tinker sampling client found for this run")
        return (
            tinker_generator(base_sampling, tokenizer),
            tinker_generator(after_sampling, tokenizer),
        )
    if backend == Backend.HF_ZEROGPU:
        job_id = train_result.adapter_path
        return (
            hf_zerogpu_generator(base_model, job_id=None, use_adapter=False),
            hf_zerogpu_generator(base_model, job_id=job_id, use_adapter=True),
        )
    if backend == Backend.COLAB:
        session = train_result.adapter_path
        return (
            colab_generator(session, use_adapter=False),
            colab_generator(session, use_adapter=True),
        )
    raise ValueError(f"Unknown backend {backend}")


def _write_report(state: RunState) -> None:
    lines = [
        f"# AutoTuneLab run {state.run_id}",
        "",
        f"**Goal:** {state.goal.raw_prompt}",
        f"**Resolved model:** {state.goal.target_model_hf_id}",
        f"**Objective type:** {state.goal.objective_type.value if state.goal.objective_type else 'n/a'}",
        "",
        "## Technique",
        f"- Name: {state.technique.name.value}",
        f"- Backend: {state.technique.backend.value}",
        f"- Rationale: {state.technique.rationale}",
        "",
        "## Dataset",
        f"- {state.dataset.description}" if state.dataset else "- n/a",
        (
            f"- train={state.dataset.num_train} val={state.dataset.num_val} "
            f"eval_holdout={state.dataset.num_eval_holdout}"
            if state.dataset
            else ""
        ),
        "",
        "## Hyperparameters",
        f"```json\n{state.hyperparams.model_dump_json(indent=2) if state.hyperparams else '{}'}\n```",
        "",
        "## Evaluation: before -> after",
    ]
    if state.eval_before and state.eval_after:
        keys = sorted(set(state.eval_before.metrics) | set(state.eval_after.metrics))
        for k in keys:
            b = state.eval_before.metrics.get(k)
            a = state.eval_after.metrics.get(k)
            lines.append(f"- **{k}**: {b} -> {a}")
    lines.append("")
    run_store.write_text(state.run_id, "report.md", "\n".join(lines))
    run_store.write_json(
        state.run_id,
        "report.json",
        {
            "run_id": state.run_id,
            "goal": state.goal.model_dump(),
            "technique": state.technique.model_dump() if state.technique else None,
            "dataset": state.dataset.model_dump() if state.dataset else None,
            "hyperparams": state.hyperparams.model_dump() if state.hyperparams else None,
            "eval_before": state.eval_before.model_dump() if state.eval_before else None,
            "eval_after": state.eval_after.model_dump() if state.eval_after else None,
        },
    )


async def _run_pipeline(run_id: str) -> None:
    state = run_store.load_state(run_id)
    log = _log_fn(run_id, "pipeline")
    try:
        # 1. Parse goal (+ clarification loop)
        _set_status(state, RunStatus.PARSING_GOAL, run_id, "goal")
        goal_log = _log_fn(run_id, "goal")
        goal_log(f"Parsing goal: {state.goal.raw_prompt!r}", kind="reasoning")
        state.goal = await parse_goal(state.goal.raw_prompt)
        run_store.save_state(state)
        goal_log(state.goal.reasoning, kind="reasoning")

        while state.goal.clarification_questions and state.goal.confidence < 0.6:
            _set_status(state, RunStatus.AWAITING_CLARIFICATION, run_id, "goal")
            for q in state.goal.clarification_questions:
                goal_log(q, kind="info")
            answer = await _wait_for_message(run_id)
            if answer == APPROVE_SENTINEL:
                break
            goal_log(f"Follow-up: {answer!r}", kind="info")
            state.goal = await parse_goal(state.goal.raw_prompt, [*state.goal.follow_up_prompts, answer])
            run_store.save_state(state)
            goal_log(state.goal.reasoning, kind="reasoning")

        if not state.goal.target_model_hf_id:
            raise RuntimeError("Could not resolve a target model; cannot continue.")

        # 2. Technique selection
        _set_status(state, RunStatus.SELECTING_TECHNIQUE, run_id, "technique")
        tech_log = _log_fn(run_id, "technique")
        state.technique = select_technique(state.goal)
        run_store.save_state(state)
        tech_log(f"Technique: {state.technique.name.value} on backend {state.technique.backend.value}", kind="reasoning")
        tech_log(state.technique.rationale, kind="reasoning")

        # 3. Dataset resolution
        _set_status(state, RunStatus.RESOLVING_DATASET, run_id, "dataset")
        dataset_log = _log_fn(run_id, "dataset")
        local_model_path = (
            state.goal.target_model_hf_id
            if state.technique.backend in (Backend.MLX_LOCAL, Backend.HERETIC_LOCAL)
            else None
        )
        state.dataset = await resolve_dataset(
            run_id, state.goal, local_model_path, dataset_log, technique_name=state.technique.name
        )
        run_store.save_state(state)
        dataset_log(f"Dataset ready: {state.dataset.description}", kind="reasoning")

        # 4. Hyperparameters
        _set_status(state, RunStatus.CONFIGURING_HYPERPARAMS, run_id, "hyperparams")
        hp_log = _log_fn(run_id, "hyperparams")
        state.hyperparams = default_hyperparams(state.technique, dataset_size_hint=state.dataset.num_train or 20)
        run_store.save_state(state)
        hp_log(f"Default hyperparameters: {state.hyperparams.model_dump()}", kind="info")

        # 5. Pause for review unless auto_approve
        if not state.auto_approve:
            _set_status(state, RunStatus.AWAITING_REVIEW, run_id, "review")
            _log_fn(run_id, "review")(
                "Paused for review — edit hyperparameters or approve to continue.", kind="info"
            )
            await _wait_for_message(run_id)
            state = run_store.load_state(run_id)  # pick up any hyperparameter PATCHes

        # 6. Train
        _set_status(state, RunStatus.TRAINING, run_id, "training")
        train_log = _log_fn(run_id, "training")
        train_metric = _metric_fn(run_id, "training")
        trainer = get_trainer(state.technique)
        run_directory = str(run_store.run_dir(run_id))
        train_result = await trainer.train(
            state.goal.target_model_hf_id, state.dataset, state.hyperparams, run_directory, train_log, train_metric
        )
        state.adapter_path = train_result.adapter_path
        run_store.save_state(state)
        train_log(
            f"Training complete. backend_ref={train_result.backend_ref} final_loss={train_result.final_loss}",
            kind="info",
        )

        # 7. Evaluate
        _set_status(state, RunStatus.EVALUATING, run_id, "eval")
        eval_log = _log_fn(run_id, "eval")
        gen_before, gen_after = await _build_generators(state, train_result)
        before, after = await run_evaluation(state.goal, state.dataset, gen_before, gen_after, eval_log)
        state.eval_before, state.eval_after = before, after
        run_store.save_state(state)
        eval_log(f"Before: {before.metrics}", kind="metric")
        eval_log(f"After: {after.metrics}", kind="metric")

        # 8. Report + complete
        _write_report(state)
        _set_status(state, RunStatus.COMPLETED, run_id, "done")
        log("Run completed.", kind="info")

    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        log(f"FAILED: {e}\n{tb}", kind="error")
        state = run_store.load_state(run_id)
        state.error = str(e)
        state.status = RunStatus.FAILED
        state.touch()
        run_store.save_state(state)
    finally:
        clear_cancel(run_id)
        if state.technique and state.technique.backend == Backend.COLAB and state.adapter_path:
            from app.trainers.colab_trainer import stop_session_sync

            log(f"Releasing Colab session {state.adapter_path}...")
            await asyncio.to_thread(stop_session_sync, state.adapter_path)
