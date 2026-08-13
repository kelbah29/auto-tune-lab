"""Filesystem-backed persistence for runs: runs/<id>/state.json + logs.jsonl + artifacts/."""
from __future__ import annotations

import json
import threading
from pathlib import Path

from app.core.schemas import RunEvent, RunState

RUNS_ROOT = Path(__file__).resolve().parents[3] / "runs"
RUNS_ROOT.mkdir(parents=True, exist_ok=True)

_locks: dict[str, threading.Lock] = {}


def _lock_for(run_id: str) -> threading.Lock:
    return _locks.setdefault(run_id, threading.Lock())


def run_dir(run_id: str) -> Path:
    d = RUNS_ROOT / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "dataset").mkdir(exist_ok=True)
    (d / "artifacts").mkdir(exist_ok=True)
    return d


def save_state(state: RunState) -> None:
    d = run_dir(state.run_id)
    with _lock_for(state.run_id):
        (d / "state.json").write_text(state.model_dump_json(indent=2))


def load_state(run_id: str) -> RunState:
    d = run_dir(run_id)
    path = d / "state.json"
    if not path.exists():
        raise FileNotFoundError(run_id)
    return RunState.model_validate_json(path.read_text())


def append_event(run_id: str, event: RunEvent) -> None:
    d = run_dir(run_id)
    with _lock_for(run_id):
        with (d / "logs.jsonl").open("a") as f:
            f.write(event.model_dump_json() + "\n")


def read_events(run_id: str) -> list[RunEvent]:
    d = run_dir(run_id)
    path = d / "logs.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(RunEvent.model_validate_json(line))
    return out


def list_runs() -> list[str]:
    if not RUNS_ROOT.exists():
        return []
    return sorted(
        [p.name for p in RUNS_ROOT.iterdir() if p.is_dir() and (p / "state.json").exists()],
        key=lambda rid: (RUNS_ROOT / rid / "state.json").stat().st_mtime,
        reverse=True,
    )


def write_json(run_id: str, relative_path: str, data) -> str:
    d = run_dir(run_id)
    path = d / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))
    return str(path)


def write_text(run_id: str, relative_path: str, text: str) -> str:
    d = run_dir(run_id)
    path = d / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return str(path)
