"""Generic fallback dataset resolution for `custom` goals: search the HF Hub for a
relevant dataset by keyword, download it, and best-effort map its columns onto an
instruction/response pair. Used when the goal doesn't match a purpose-built
resolver (behavior_removal / uncensor / benchmark_improvement).
"""
from __future__ import annotations

import re

from datasets import load_dataset
from huggingface_hub import HfApi

from app.core.schemas import DatasetExample

_INSTRUCTION_COLS = ["instruction", "prompt", "question", "input", "user"]
_RESPONSE_COLS = ["output", "response", "answer", "completion", "chosen", "assistant"]

_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "for", "on", "in", "with", "its",
    "improve", "increase", "raise", "boost", "performance", "fine", "tune", "model",
}


def _keywords(text: str) -> str:
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]+", text.lower())
    kept = [w for w in words if w not in _STOPWORDS and len(w) > 2]
    return " ".join(kept[:6]) or text[:40]


def search_candidate_datasets(goal_text: str, limit: int = 5) -> list[str]:
    api = HfApi()
    query = _keywords(goal_text)
    results = list(api.list_datasets(search=query, limit=limit, sort="trending"))
    return [r.id for r in results]


def _map_columns(columns: list[str]) -> tuple[str | None, str | None]:
    cols_lower = {c.lower(): c for c in columns}
    instr = next((cols_lower[c] for c in _INSTRUCTION_COLS if c in cols_lower), None)
    resp = next((cols_lower[c] for c in _RESPONSE_COLS if c in cols_lower), None)
    return instr, resp


async def build_generic_dataset(
    goal_text: str, log, n_train: int = 40, n_eval_holdout: int = 10
) -> tuple[list[DatasetExample], list[str], str | None]:
    """Returns (training examples, held-out eval prompts, dataset id used or None)."""
    candidates = search_candidate_datasets(goal_text)
    log(f"HF Hub search for {_keywords(goal_text)!r} -> candidates: {candidates}")

    for dataset_id in candidates:
        try:
            ds = load_dataset(dataset_id, split="train", streaming=True)
            rows = list(ds.take(n_train + n_eval_holdout))
        except Exception as e:  # noqa: BLE001
            log(f"  skip {dataset_id}: {e}")
            continue
        if not rows:
            continue
        instr_col, resp_col = _map_columns(list(rows[0].keys()))
        if not instr_col or not resp_col:
            log(f"  skip {dataset_id}: no recognizable instruction/response columns")
            continue

        train_rows, eval_rows = rows[:n_train], rows[n_train : n_train + n_eval_holdout]
        examples = [
            DatasetExample(
                messages=[
                    {"role": "user", "content": str(r[instr_col])},
                    {"role": "assistant", "content": str(r[resp_col])},
                ]
            )
            for r in train_rows
            if r.get(instr_col) and r.get(resp_col)
        ]
        eval_prompts = [str(r[instr_col]) for r in eval_rows if r.get(instr_col)]
        if examples:
            log(f"Using {dataset_id}: {len(examples)} train, {len(eval_prompts)} eval "
                f"(columns {instr_col!r} -> {resp_col!r})")
            return examples, eval_prompts, dataset_id

    log("No suitable Hub dataset found; falling back to built-in seed prompts.")
    from app.datasets.seed_prompts import GENERAL_SEED_PROMPTS

    return [], GENERAL_SEED_PROMPTS[:n_eval_holdout], None
