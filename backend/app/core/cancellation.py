"""Cross-thread cancellation signals for long-running synchronous work.

`asyncio.Task.cancel()` only raises `CancelledError` at the next `await` point in a
coroutine — it can't interrupt code already dispatched to a thread pool via
`asyncio.to_thread` (e.g. heretic's Optuna study, which is a tight synchronous loop).
Trainers that offload heavy sync work to a thread should poll
`get_cancel_event(run_id).is_set()` at safe checkpoints (e.g. between trials) so
`orchestrator.cancel_run()` actually stops the work instead of just relabeling it.
"""
from __future__ import annotations

import threading

_EVENTS: dict[str, threading.Event] = {}


def get_cancel_event(run_id: str) -> threading.Event:
    return _EVENTS.setdefault(run_id, threading.Event())


def request_cancel(run_id: str) -> None:
    get_cancel_event(run_id).set()


def clear_cancel(run_id: str) -> None:
    _EVENTS.pop(run_id, None)
