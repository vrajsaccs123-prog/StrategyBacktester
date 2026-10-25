"""Walk-forward parameter search with a locked holdout segment."""

from __future__ import annotations

import math

import optuna
import pandas as pd

from engine.backtest import run_backtest
from engine.metrics import compute_metrics, objective_value
from engine.models import BacktestSettings, CompletedRun, FoldScore, OptimizationResult, ParamSpec
from engine.service import complete_run
from engine.strategy import Strategy, resolve_params

optuna.logging.set_verbosity(optuna.logging.WARNING)

PENALTY = -1e9


def holdout_start(n: int, frac: float) -> int:
    if not 0.05 <= frac <= 0.5:
        raise ValueError("Holdout fraction must be between 5% and 50%")
    start = int(round(n * (1 - frac)))
    if start < 20 or n - start < 5:
        raise ValueError("Not enough bars for that holdout. Use a longer CSV or a smaller holdout.")
    return start


def walk_forward_bounds(n: int, n_folds: int) -> list[tuple[int, int]]:
    """Validation windows (start, end) with end exclusive. History before each window is warmup."""

    if n_folds < 2:
        raise ValueError("Use at least 2 walk-forward folds")
    segments = n_folds + 1
    if n < segments * 5:
        raise ValueError(f"Need at least {segments * 5} bars before the holdout for {n_folds} folds")

    edges = [int(round(i * n / segments)) for i in range(segments + 1)]
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1
    edges[-1] = n
    folds = []
    for k in range(n_folds):
        val_start = edges[k + 1]
        val_end = edges[k + 2]
        if val_end <= val_start:
            raise ValueError("Could not build walk-forward folds for this series")
        folds.append((val_start, val_end))
    return folds


def _snap_int(low: float, high: float, step: float | None) -> tuple[int, int, int]:
    step_i = max(int(round(step or 1)), 1)
    low_i = int(round(low))
    high_i = int(round(high))
    if high_i < low_i:
        high_i = low_i
    high_i = low_i + ((high_i - low_i) // step_i) * step_i
    return low_i, high_i, step_i


def _snap_float(low: float, high: float, step: float | None) -> tuple[float, float, float | None]:
    low_f = round(float(low), 8)
    high_f = round(float(high), 8)
    if step is None or float(step) <= 0:
        return low_f, high_f, None
    step_f = round(float(step), 8)
    count = int(math.floor((high_f - low_f) / step_f + 1e-8))
    high_f = round(low_f + count * step_f, 8)
    if high_f <= low_f:
        high_f = round(low_f + step_f, 8)
    return low_f, high_f, step_f


def _suggest(trial: optuna.Trial, space: dict[str, ParamSpec]) -> dict:
    suggested = {}
    for name, spec in space.items():
        if not spec.searchable:
            continue
        if spec.kind == "int":
            low, high, step = _snap_int(spec.min, spec.max, spec.step)
            suggested[name] = low if low == high else trial.suggest_int(name, low, high, step=step)
        else:
            low, high, step = _snap_float(spec.min, spec.max, spec.step)
            if math.isclose(low, high):
                suggested[name] = low
            elif step is None:
                suggested[name] = trial.suggest_float(name, low, high)
            else:
                suggested[name] = trial.suggest_float(name, low, high, step=step)
    return suggested


def _score_window(
    bars: pd.DataFrame,
    strategy: Strategy,
    params: dict,
    start: int,
    end: int,
    settings: BacktestSettings,
    objective: str,
    max_dd_cap: float | None,
    scored: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> float:
    frame = bars.iloc[:end].copy()
    scored.append((frame.index[0], frame.index[-1]))
    target = strategy.target_positions(frame, params, start=start)
    segment = frame.iloc[start:]
    result = run_backtest(segment, target, settings)
    metrics = compute_metrics(result)
    return objective_value(metrics, objective, max_dd_cap if objective == "cagr" else None)


def optimize(
    bars: pd.DataFrame,
    strategy: Strategy,
    settings: BacktestSettings,
    objective: str = "sharpe",
    holdout_frac: float = 0.2,
    n_folds: int = 4,
    n_trials: int = 30,
    seed: int = 42,
    max_dd_cap: float | None = None,
    param_space: dict[str, ParamSpec] | None = None,
    progress=None,
) -> OptimizationResult:
    """Search `param_space` on walk-forward validation folds. The holdout is scored once, after selection."""

    if objective not in {"sharpe", "sortino", "calmar", "cagr"}:
        raise ValueError(f"Unknown objective '{objective}'")
    if n_trials < 1:
        raise ValueError("Run at least one trial")

    settings.validate()
    test_loc = holdout_start(len(bars), holdout_frac)
    search = bars.iloc[:test_loc].copy()
    folds = walk_forward_bounds(len(search), n_folds)
    space = dict(param_space or strategy.param_space())
    if not any(spec.searchable for spec in space.values()):
        raise ValueError("Mark at least one parameter with a min and max so it can be searched")

    scored: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def objective_fn(trial: optuna.Trial) -> float:
        params = resolve_params(space, _suggest(trial, space))
        scores = []
        for step, (val_start, val_end) in enumerate(folds):
            try:
                score = _score_window(
                    search, strategy, params, val_start, val_end, settings, objective, max_dd_cap, scored
                )
            except Exception as exc:
                trial.set_user_attr("error", str(exc))
                score = PENALTY
            scores.append(score)
            trial.report(float(pd.Series(scores).median()), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(pd.Series(scores).median())

    def _on_trial(study: optuna.Study, trial: optuna.Trial) -> None:
        if progress is not None:
            finished = sum(item.state.is_finished() for item in study.trials)
            progress(finished, n_trials)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    study.optimize(objective_fn, n_trials=n_trials, callbacks=[_on_trial], show_progress_bar=False)
    if study.best_value <= PENALTY / 10:
        raise RuntimeError(
            "No parameter set produced a finite score. Widen the ranges or check that the rules can trade."
        )

    best = resolve_params(space, study.best_params)
    fold_scores = []
    for number, (val_start, val_end) in enumerate(folds, start=1):
        score = _score_window(
            search, strategy, best, val_start, val_end, settings, objective, max_dd_cap, scored=[]
        )
        fold_scores.append(
            FoldScore(
                fold=number,
                val_start=search.index[val_start],
                val_end=search.index[val_end - 1],
                score=score,
            )
        )

    cap = max_dd_cap if objective == "cagr" else None
    pretest = complete_run(search, strategy, best, settings, start=0)
    test = complete_run(bars, strategy, best, settings, start=test_loc)
    pre_value = objective_value(pretest.metrics, objective, cap)
    test_value = objective_value(test.metrics, objective, cap)
    if pre_value <= PENALTY / 10 or test_value <= PENALTY / 10:
        gap = float("nan")
    else:
        gap = pre_value - test_value

    trial_rows = []
    for trial in study.trials:
        row = {"number": trial.number, "value": trial.value, "state": trial.state.name}
        row.update(trial.params)
        trial_rows.append(row)

    return OptimizationResult(
        best_params=best,
        objective=objective,
        validation_median=float(study.best_value),
        fold_scores=fold_scores,
        trials=pd.DataFrame(trial_rows),
        pretest=pretest,
        test=test,
        gap=gap,
        test_start=bars.index[test_loc],
        scored_windows=scored,
        max_dd_cap=max_dd_cap,
    )
