from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from app.core import run_store
from app.core.schemas import RunState
from app.pipeline import orchestrator

router = APIRouter(prefix="/api/runs", tags=["runs"])


class StartRunRequest(BaseModel):
    prompt: str
    auto_approve: bool = False


class MessageRequest(BaseModel):
    text: str


@router.post("", response_model=RunState)
async def create_run(req: StartRunRequest) -> RunState:
    return await orchestrator.start_run(req.prompt, auto_approve=req.auto_approve)


@router.get("", response_model=list[RunState])
async def list_runs() -> list[RunState]:
    out = []
    for rid in run_store.list_runs():
        try:
            out.append(run_store.load_state(rid))
        except Exception:
            continue
    return out


@router.get("/{run_id}", response_model=RunState)
async def get_run(run_id: str) -> RunState:
    try:
        return run_store.load_state(run_id)
    except FileNotFoundError:
        raise HTTPException(404, "run not found")


@router.post("/{run_id}/message")
async def post_message(run_id: str, req: MessageRequest) -> dict:
    await orchestrator.add_message(run_id, req.text)
    return {"ok": True}


@router.post("/{run_id}/approve")
async def approve(run_id: str) -> dict:
    await orchestrator.approve_review(run_id)
    return {"ok": True}


@router.patch("/{run_id}/hyperparams", response_model=RunState)
async def patch_hyperparams(run_id: str, updates: dict) -> RunState:
    try:
        return orchestrator.patch_hyperparams(run_id, updates)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/{run_id}/cancel")
async def cancel(run_id: str) -> dict:
    orchestrator.cancel_run(run_id)
    return {"ok": True}


@router.get("/{run_id}/events")
async def stream_events(run_id: str) -> EventSourceResponse:
    async def event_generator():
        # Subscribe before replaying history so nothing published in the gap is lost
        # (worst case a duplicated line, never a missing one).
        q = orchestrator.BUS.subscribe(run_id)
        try:
            seen = set()
            for event in run_store.read_events(run_id):
                key = (event.ts, event.message)
                seen.add(key)
                yield {"event": "log", "data": event.model_dump_json()}
            while True:
                event = await q.get()
                key = (event.ts, event.message)
                if key in seen:
                    seen.discard(key)
                    continue
                yield {"event": "log", "data": event.model_dump_json()}
        finally:
            orchestrator.BUS.unsubscribe(run_id, q)

    return EventSourceResponse(event_generator())
