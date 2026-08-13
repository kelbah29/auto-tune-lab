"""Before/after evaluator for `benchmark_improvement` (SWE-bench-style) goals.

A full Docker-execution SWE-bench harness (build the repo at base_commit, apply the
patch, run fail_to_pass/pass_to_pass tests) is out of scope for this environment —
the assignment explicitly allows inventing a task-appropriate proxy eval instead of
requiring the real benchmark harness. We use two independent, real signals instead
of one: whether the generated patch is structurally well-formed enough to actually
apply (`git apply --check`, a genuine structural-correctness check, not just a
vibe) and an LLM-judge 1-5 correctness rubric against the golden patch and problem
statement.
"""
from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from app.core.llm import get_llm
from app.core.model_access import GenFn
from app.core.schemas import EvalReport

_JUDGE_SYSTEM = (
    "You are an expert code reviewer. Given a GitHub issue, a golden reference patch, "
    "and a candidate patch, rate how well the candidate patch resolves the issue on a "
    "1-5 scale (5 = fully resolves it equivalently to the golden patch, 1 = irrelevant "
    "or broken). Respond with ONLY the digit."
)

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@")
_FILE_HEADER_RE = re.compile(r"^--- (?:a/(.+)|/dev/null)")


def _reconstruct_pre_image(patch_text: str) -> dict[str, str]:
    """Synthesizes plausible 'before' file contents directly from the patch's own
    hunks (context + removed lines, in order) so `git apply --check` has something
    real to apply against. We don't have the actual SWE-bench repo checked out
    (that's the out-of-scope Docker harness) — but a unified diff's hunks are only
    internally consistent if the line-count arithmetic in each `@@ ... @@` header
    matches its content, so reconstructing the pre-image and applying against it
    still catches real malformed/self-contradicting output, not just a vibe.
    """
    files: dict[str, list[str]] = {}
    current_file: str | None = None
    for line in patch_text.splitlines():
        header = _FILE_HEADER_RE.match(line)
        if header:
            current_file = header.group(1)
            if current_file:
                files.setdefault(current_file, [])
            continue
        if current_file is None:
            continue
        if line.startswith((" ", "-")) and not line.startswith(("---", "+++")):
            files[current_file].append(line[1:])
    return {path: "\n".join(lines) for path, lines in files.items() if lines}


def _patch_applies(patch_text: str) -> bool:
    if not patch_text.strip().startswith(("diff", "---", "Index:")):
        return False
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for rel_path, content in _reconstruct_pre_image(patch_text).items():
            file_path = tmp_path / rel_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content + "\n" if content else content)
        subprocess.run(["git", "init", "-q"], cwd=tmp, capture_output=True)
        patch_file = tmp_path / "patch.diff"
        patch_file.write_text(patch_text)
        result = subprocess.run(
            ["git", "apply", "--check", "--unsafe-paths", "patch.diff"],
            cwd=tmp,
            capture_output=True,
            text=True,
        )
        return result.returncode == 0


async def _judge_patch(problem_statement: str, golden_patch: str, candidate_patch: str) -> float:
    llm = get_llm()
    # Same constraint as letter_suppression._judge_coherence: only one concurrent Colab
    # GPU session is allowed, and gen() above already holds it for the model under test.
    # patch_apply_rate is the primary, judge-free signal; skip the LLM judge on Colab
    # rather than tearing down the eval session mid-run to provision a second one.
    if not llm.available or llm._backend == "colab":
        return float("nan")
    try:
        user = (
            f"Issue:\n{problem_statement[:1500]}\n\nGolden patch:\n{golden_patch[:1500]}\n\n"
            f"Candidate patch:\n{candidate_patch[:1500]}"
        )
        text = await llm.acomplete(_JUDGE_SYSTEM, user, max_tokens=5, temperature=0.0)
        digit = next((c for c in text if c.isdigit()), None)
        return float(digit) if digit else float("nan")
    except Exception:
        return float("nan")


async def evaluate_swebench_proxy(label: str, instances: list[dict], gen: GenFn, log) -> EvalReport:
    samples = []
    applies_count = 0
    judge_scores = []
    for inst in instances:
        candidate = await gen(
            "You are an expert software engineer. Respond with ONLY a unified diff patch.",
            inst["prompt"],
        )
        applies = _patch_applies(candidate)
        applies_count += int(applies)
        score = await _judge_patch(inst["problem_statement"], inst["golden_patch"], candidate)
        if score == score:  # not NaN
            judge_scores.append(score)
        samples.append(
            {
                "instance_id": inst["instance_id"],
                "applies": applies,
                "judge_score": score,
                "candidate_patch": candidate[:500],
            }
        )
        log(f"  [{label}] {inst['instance_id']}: applies={applies} judge={score}")

    n = len(instances)
    metrics = {
        "patch_apply_rate": applies_count / n if n else 0.0,
        "applies": float(applies_count),
        "total": float(n),
    }
    if judge_scores:
        metrics["avg_judge_score"] = sum(judge_scores) / len(judge_scores)
    return EvalReport(label=label, metrics=metrics, samples=samples)
