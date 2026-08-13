"""Local abliteration (directional weight ablation) via the `heretic` library.

heretic's own CLI has no non-interactive/scriptable save path — after the Optuna
search it always drops into a `questionary` menu asking which trial to keep and
where to save it (see heretic/main.py). Rather than fight that with piped stdin, we
call the exact same underlying classes (`Model`, `Evaluator`, the same Optuna study
loop) main.py uses, but pick the best Pareto-optimal trial automatically (fewest
refusals, matching heretic's own tie-breaking) and save directly — same algorithm,
zero interactivity. This is what implements the assignment's Heretic resource.
"""
from __future__ import annotations

import asyncio
import threading
import warnings
from os.path import commonprefix
from pathlib import Path

from app.core.cancellation import get_cancel_event
from app.core.schemas import DatasetSpec, HyperparamConfig
from app.trainers.base import LogFn, MetricFn, TrainResult


def _pick_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _run_heretic_sync(
    base_model: str, out_dir: str, n_trials: int, log: LogFn, cancel_event: threading.Event
) -> dict:
    import optuna
    import torch
    import torch.nn.functional as F
    from optuna import Trial
    from optuna.exceptions import ExperimentalWarning
    from optuna.samplers import TPESampler
    from optuna.study import StudyDirection

    from heretic.config import Settings
    from heretic.evaluator import Evaluator
    from heretic.model import AbliterationParameters, Model
    from heretic.utils import empty_cache, load_prompts

    warnings.filterwarnings("ignore", category=ExperimentalWarning)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    torch.set_grad_enabled(False)

    device = _pick_device()
    n_trials = max(3, n_trials)
    settings = Settings(
        _cli_parse_args=False,
        model=base_model,
        device_map=device,
        n_trials=n_trials,
        n_startup_trials=max(1, n_trials // 3),
        batch_size=4,  # skip the auto-benchmark search; small and safe on 8GB RAM
    )

    log(f"Loading {base_model} for abliteration on device={device}...")
    model = Model(settings)
    if model.model is None:
        raise RuntimeError("heretic failed to load the model with any supported dtype")

    log(f"Loading prompt sets: good={settings.good_prompts.dataset}, bad={settings.bad_prompts.dataset}")
    good_prompts = load_prompts(settings.good_prompts)
    bad_prompts = load_prompts(settings.bad_prompts)
    log(f"* {len(good_prompts)} good, {len(bad_prompts)} bad prompts loaded")

    responses = model.get_responses_batched(good_prompts[:50] + bad_prompts[:50])
    model.response_prefix = commonprefix(responses).rstrip(" ")
    if model.response_prefix.startswith("<think>"):
        model.response_prefix = "<think></think>"

    evaluator = Evaluator(settings, model)
    log(f"Baseline (pre-abliteration) refusals: {evaluator.base_refusals}/{len(evaluator.bad_prompts)}")

    log("Computing per-layer refusal directions from good/bad prompt residuals...")
    good_residuals = model.get_residuals_batched(good_prompts)
    bad_residuals = model.get_residuals_batched(bad_prompts)
    refusal_directions = F.normalize(
        bad_residuals.mean(dim=0) - good_residuals.mean(dim=0), p=2, dim=1
    )
    del good_residuals, bad_residuals
    empty_cache()

    trial_index = 0

    def objective(trial: Trial):
        nonlocal trial_index
        if cancel_event.is_set():
            log("Cancellation requested — stopping the Optuna study.")
            trial.study.stop()
            raise optuna.TrialPruned()
        trial_index += 1
        trial.set_user_attr("index", trial_index)

        direction_scope = trial.suggest_categorical("direction_scope", ["global", "per layer"])
        direction_index = trial.suggest_float(
            "direction_index", 0.4 * (len(model.get_layers()) - 1), 0.9 * (len(model.get_layers()) - 1)
        )
        if direction_scope == "per layer":
            direction_index = None

        parameters = {}
        for component in model.get_abliterable_components():
            max_weight = trial.suggest_float(f"{component}.max_weight", 0.8, 1.5)
            max_weight_position = trial.suggest_float(
                f"{component}.max_weight_position",
                0.6 * (len(model.get_layers()) - 1),
                len(model.get_layers()) - 1,
            )
            min_weight = trial.suggest_float(f"{component}.min_weight", 0.0, 1.0)
            min_weight_distance = trial.suggest_float(
                f"{component}.min_weight_distance", 1.0, 0.6 * (len(model.get_layers()) - 1)
            )
            parameters[component] = AbliterationParameters(
                max_weight=max_weight,
                max_weight_position=max_weight_position,
                min_weight=min_weight * max_weight,
                min_weight_distance=min_weight_distance,
            )

        trial.set_user_attr("direction_index", direction_index)
        trial.set_user_attr("parameters", parameters)

        log(f"Trial {trial_index}/{n_trials}: reload + abliterate + evaluate...")
        model.reload_model()
        model.abliterate(refusal_directions, direction_index, parameters)
        score, kl_divergence, refusals = evaluator.get_score()
        trial.set_user_attr("kl_divergence", kl_divergence)
        trial.set_user_attr("refusals", refusals)
        log(f"  -> refusals={refusals}/{len(evaluator.bad_prompts)} kl_divergence={kl_divergence:.4f}")
        return score

    study = optuna.create_study(
        sampler=TPESampler(n_startup_trials=settings.n_startup_trials, n_ei_candidates=128, multivariate=True),
        directions=[StudyDirection.MINIMIZE, StudyDirection.MINIMIZE],
    )
    study.optimize(objective, n_trials=n_trials)

    if not study.best_trials:
        raise RuntimeError("heretic optimization produced no viable (Pareto-optimal) trials")

    chosen = sorted(study.best_trials, key=lambda t: t.user_attrs["refusals"])[0]
    log(
        f"Chosen trial {chosen.user_attrs['index']}: refusals={chosen.user_attrs['refusals']}/"
        f"{len(evaluator.bad_prompts)}, kl_divergence={chosen.user_attrs['kl_divergence']:.4f}"
    )

    model.reload_model()
    model.abliterate(refusal_directions, chosen.user_attrs["direction_index"], chosen.user_attrs["parameters"])
    _score, final_kl, final_refusals = evaluator.get_score()

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.model.save_pretrained(out_dir)
    model.tokenizer.save_pretrained(out_dir)
    log(f"Abliterated model saved to {out_dir}")

    return {
        "base_refusals": evaluator.base_refusals,
        "final_refusals": final_refusals,
        "total_bad_prompts": len(evaluator.bad_prompts),
        "total_good_prompts": len(evaluator.good_prompts),
        "kl_divergence": final_kl,
        "n_trials_run": trial_index,
    }


class HereticAbliterationTrainer:
    async def train(
        self,
        base_model: str,
        dataset: DatasetSpec,
        hyperparams: HyperparamConfig,
        run_dir: str,
        log: LogFn,
        metric: MetricFn,
    ) -> TrainResult:
        out_dir = str(Path(run_dir) / "artifacts" / "abliterated_model")
        n_trials = int(hyperparams.extra.get("trials", 15))
        run_id = Path(run_dir).name
        cancel_event = get_cancel_event(run_id)
        result = await asyncio.to_thread(_run_heretic_sync, base_model, out_dir, n_trials, log, cancel_event)
        metric("base_refusals", result["base_refusals"], 0)
        metric("final_refusals", result["final_refusals"], result["n_trials_run"])
        metric("kl_divergence", result["kl_divergence"], result["n_trials_run"])
        return TrainResult(
            adapter_path=None,
            model_path=out_dir,
            backend_ref=f"heretic_local:{base_model}",
            final_loss=None,
            extra=result,
        )
