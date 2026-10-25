"""Choose indicator parameters on the train window, then score a locked test window.

`study_book` is the book backtest. The first 70% of sessions are the train
window. Indicator settings are sampled there and scored on expanding folds.
A random forest reads those scores and proposes a short extra list. A proposal
is kept only after the same folds score it. The last 30% is run once, after
the choice is locked, and it does not change the parameters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from engine.book import indicator_names, run_book
from engine.metrics import compute_metrics
from engine.models import BacktestResult, ParamSpec
from engine.strategy import Strategy, resolve_params

TRAIN_FRACTION = 0.70
PENALTY = -1e9
HOLDS_RATIO = 0.75


@dataclass
class FoldRead:
    fold: int
    start: pd.Timestamp
    end: pd.Timestamp
    score: float


@dataclass
class StudyResult:
    strategy_key: str
    strategy_name: str
    params: dict
    searched: list[str]
    optimized: bool
    split_end: pd.Timestamp
    test_start: pd.Timestamp
    first_date: pd.Timestamp
    last_date: pd.Timestamp
    universe_size: int
    train_result: BacktestResult
    test_result: BacktestResult
    train_metrics: dict
    test_metrics: dict
    train_score: float
    test_score: float
    selection_score: float
    selection_std: float
    verdict: str
    verdict_text: str
    samples: pd.DataFrame
    importances: pd.DataFrame
    oob_r2: float
    folds: list[FoldRead] = field(default_factory=list)
    scored_ends: list[pd.Timestamp] = field(default_factory=list)


def calendar(data: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    dates = None
    for frame in data.values():
        dates = frame.index if dates is None else dates.union(frame.index)
    if dates is None or len(dates) == 0:
        raise ValueError("No bars to study")
    return pd.DatetimeIndex(pd.to_datetime(dates)).sort_values().unique()


def window_dates(study: StudyResult) -> tuple[str, str]:
    """Inclusive session dates of the train set and the test set."""

    def span(start, end) -> str:
        return f"{pd.Timestamp(start):%d %b %Y} to {pd.Timestamp(end):%d %b %Y}"

    return span(study.first_date, study.split_end), span(study.test_start, study.last_date)


def research_split(data: dict[str, pd.DataFrame]) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Last session of the train window, and the first session of the test window."""

    dates = calendar(data)
    cut = int(len(dates) * TRAIN_FRACTION)
    if cut < 40 or len(dates) - cut < 15:
        raise ValueError("Need a longer history for a 70/30 split")
    return pd.Timestamp(dates[cut - 1]), pd.Timestamp(dates[cut])


def fold_windows(dates: pd.DatetimeIndex, n_folds: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Validation blocks on the train calendar. Earlier sessions stay as warmup."""

    if n_folds < 2:
        raise ValueError("Use at least 2 cross-validation folds")
    warm = max(int(len(dates) * 0.40), 15)
    if len(dates) - warm < n_folds * 5:
        raise ValueError("The train window is too short for that many folds")
    rest = len(dates) - warm
    edges = [warm + int(round(i * rest / n_folds)) for i in range(n_folds + 1)]
    edges[0] = warm
    edges[-1] = len(dates)
    for index in range(1, len(edges)):
        if edges[index] <= edges[index - 1]:
            edges[index] = edges[index - 1] + 1
    edges[-1] = len(dates)
    windows = []
    for fold in range(n_folds):
        start = pd.Timestamp(dates[edges[fold]])
        end = pd.Timestamp(dates[edges[fold + 1] - 1])
        if end < start:
            raise ValueError("Could not build cross-validation folds")
        windows.append((start, end))
    return windows


def ratio_score(metrics: dict) -> float:
    """CAGR divided by the absolute max drawdown. Higher is better."""

    cagr = metrics.get("cagr", float("nan"))
    drawdown = metrics.get("max_drawdown", float("nan"))
    if not isinstance(cagr, (int, float)) or not isinstance(drawdown, (int, float)):
        return PENALTY
    if not math.isfinite(cagr) or not math.isfinite(drawdown):
        return PENALTY
    if drawdown < 0:
        return float(cagr) / abs(float(drawdown))
    if cagr > 0:
        return float(cagr) / 1e-4
    if cagr == 0:
        return 0.0
    return PENALTY


def judge(train_score: float, test_score: float) -> tuple[str, str]:
    """Whether the locked test window holds the train result."""

    if not math.isfinite(train_score) or not math.isfinite(test_score) or train_score <= PENALTY / 10:
        return (
            "No read",
            "One window did not produce a finite CAGR relative to its max drawdown.",
        )
    if train_score <= 0 and test_score <= 0:
        return ("Weak", "CAGR does not pay for the drawdown in the train window or the test window.")
    if train_score > 0 >= test_score:
        return (
            "Does not hold",
            "The train window is positive and the test window is not. These parameters do not survive the unseen 30%.",
        )
    if train_score <= 0 < test_score:
        return (
            "Test only",
            "The test window is positive while the train window is not. That is not a confirmed edge.",
        )
    ratio = test_score / train_score
    shown = f"{ratio:.0%}"
    if ratio >= HOLDS_RATIO:
        return (
            "Holds on test",
            f"Test CAGR/|max drawdown| is {shown} of the train value. The choice travels to the unseen 30%.",
        )
    return (
        "Decays on test",
        f"Test CAGR/|max drawdown| is {shown} of the train value. The fit is stronger on the train window.",
    )


def _draw(rng: np.random.Generator, spec: ParamSpec) -> float | int:
    if spec.kind == "int":
        step = max(int(round(spec.step or 1)), 1)
        low = int(round(spec.min))
        high = int(round(spec.max))
        high = low + ((high - low) // step) * step
        values = list(range(low, high + 1, step))
        return int(rng.choice(values))
    low = float(spec.min)
    high = float(spec.max)
    step = spec.step
    if step is None or float(step) <= 0:
        return float(rng.uniform(low, high))
    step_f = float(step)
    count = int(math.floor((high - low) / step_f + 1e-8))
    values = [round(low + index * step_f, 8) for index in range(count + 1)]
    return float(rng.choice(values))


def _key(params: dict, names: list[str]) -> tuple:
    return tuple(params[name] for name in names)


def _snap_below(spec: ParamSpec, ceiling: int):
    """Largest grid value strictly below `ceiling`, or None when the range has none."""

    if spec.min is None or spec.max is None:
        return None
    step = max(int(round(spec.step or 1)), 1)
    low = int(round(float(spec.min)))
    high = int(round(float(spec.max)))
    if ceiling <= low:
        return None
    limit = min(high, ceiling - 1)
    snapped = low + ((limit - low) // step) * step
    if snapped < low or snapped > high or snapped >= ceiling:
        return None
    return spec.coerce(snapped)


def _snap_above(spec: ParamSpec, floor: int):
    """Smallest grid value strictly above `floor`, or None when the range has none."""

    if spec.min is None or spec.max is None:
        return None
    step = max(int(round(spec.step or 1)), 1)
    low = int(round(float(spec.min)))
    high = int(round(float(spec.max)))
    start = max(low, floor + 1)
    if start > high:
        return None
    offset = (start - low + step - 1) // step
    snapped = low + offset * step
    if snapped <= floor or snapped > high:
        return None
    return spec.coerce(snapped)


def repair_lengths(space: dict[str, ParamSpec], params: dict) -> dict | None:
    """Keep a fast length below its slow pair. None means the ranges cannot do that."""

    trial = dict(params)
    for fast_name, slow_name in (("fast", "slow"), ("weekly_fast", "weekly_slow")):
        if fast_name not in trial or slow_name not in trial:
            continue
        fast_spec = space.get(fast_name)
        slow_spec = space.get(slow_name)
        if fast_spec is None or slow_spec is None:
            continue
        fast = int(fast_spec.coerce(trial[fast_name]))
        slow = int(slow_spec.coerce(trial[slow_name]))
        if fast < slow:
            trial[fast_name] = fast_spec.coerce(fast)
            trial[slow_name] = slow_spec.coerce(slow)
            continue
        lowered = _snap_below(fast_spec, slow)
        if lowered is not None:
            trial[fast_name] = lowered
            trial[slow_name] = slow_spec.coerce(slow)
            continue
        raised = _snap_above(slow_spec, fast)
        if raised is not None:
            trial[fast_name] = fast_spec.coerce(fast)
            trial[slow_name] = raised
            continue
        return None
    return trial


def _draw_unseen(rng, space, names: list[str], base: dict, seen: set[tuple], budget: list[int]) -> dict | None:
    """Draw one unseen repaired set. `budget` is a one-item list of attempts still allowed."""

    while budget[0] > 0:
        budget[0] -= 1
        trial = dict(base)
        for name in names:
            trial[name] = _draw(rng, space[name])
        repaired = repair_lengths(space, trial)
        if repaired is None:
            continue
        key = _key(repaired, names)
        if key in seen:
            continue
        seen.add(key)
        return repaired
    return None


def _encode(params: dict, names: list[str]) -> np.ndarray:
    return np.array([float(params[name]) for name in names], dtype=float)


def _finite(score: float) -> bool:
    return math.isfinite(score) and score > PENALTY / 10


class _Ticker:
    def __init__(self, n_symbols: int, progress, n_runs: int):
        self.n_symbols = max(n_symbols, 1)
        self.progress = progress
        self.n_runs = max(n_runs, 1)
        self.done_runs = 0
        self.label = "Running"
        self.label_base = "Running"

    def run(self, strategy, data, params, start, end):
        def inner(done: int, total: int) -> None:
            if self.progress is None:
                return
            step = self.done_runs * total + done
            self.progress(min(step, self.n_runs * total), self.n_runs * total, f"{self.label} · {done} of {total} names")

        try:
            return run_book(strategy, data, params, start=start, end=end, progress=inner)
        finally:
            self.done_runs += 1


def _score_window(strategy, data, params, start, end, ticker: _Ticker) -> float:
    try:
        result = ticker.run(strategy, data, params, start, end)
    except Exception:
        return PENALTY
    return ratio_score(compute_metrics(result))


def _fold_scores(strategy, data, params, windows, ticker: _Ticker, scored_ends: list[pd.Timestamp]) -> list[float]:
    scores = []
    for fold, (start, end) in enumerate(windows, start=1):
        ticker.label = ticker.label_base + f" · fold {fold} of {len(windows)}"
        scores.append(_score_window(strategy, data, params, start, end, ticker))
        scored_ends.append(end)
    return scores


def _summary(scores: list[float]) -> tuple[float, float]:
    array = np.array(scores, dtype=float)
    return float(np.median(array)), float(np.std(array))


def _fit_forest(matrix: np.ndarray, target: np.ndarray, seed: int):
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(
        n_estimators=400,
        min_samples_leaf=1 if len(target) < 12 else 2,
        random_state=seed,
        bootstrap=True,
        oob_score=len(target) >= 10,
        n_jobs=1,
    )
    model.fit(matrix, target)
    return model


def _predict(model, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    trees = np.stack([tree.predict(matrix) for tree in model.estimators_], axis=0)
    return trees.mean(axis=0), trees.std(axis=0)


def _propose(rng, space, names, base, seen: set[tuple], model, n_candidates: int, n_shortlist: int) -> list[dict]:
    pool = []
    budget = [n_candidates * 30]
    while len(pool) < n_candidates:
        trial = _draw_unseen(rng, space, names, base, seen, budget)
        if trial is None:
            break
        pool.append(trial)
    if not pool:
        return []
    matrix = np.vstack([_encode(trial, names) for trial in pool])
    mean, std = _predict(model, matrix)
    order = np.argsort(-(mean - std))
    return [pool[int(index)] for index in order[:n_shortlist]]


def _finish(
    strategy: Strategy,
    data: dict[str, pd.DataFrame],
    params: dict,
    searched: list[str],
    optimized: bool,
    split_end: pd.Timestamp,
    test_start: pd.Timestamp,
    selection_score: float,
    selection_std: float,
    samples: pd.DataFrame,
    importances: pd.DataFrame,
    oob_r2: float,
    folds: list[FoldRead],
    scored_ends: list[pd.Timestamp],
    ticker: _Ticker,
) -> StudyResult:
    dates = calendar(data)
    ticker.label = "Train window"
    train = ticker.run(strategy, data, params, None, split_end)
    ticker.label = "Test window"
    test = ticker.run(strategy, data, params, test_start, None)
    train_metrics = compute_metrics(train)
    test_metrics = compute_metrics(test)
    train_score = ratio_score(train_metrics)
    test_score = ratio_score(test_metrics)
    verdict, text = judge(train_score, test_score)
    return StudyResult(
        strategy_key=strategy.key,
        strategy_name=strategy.name,
        params=params,
        searched=searched,
        optimized=optimized,
        split_end=split_end,
        test_start=test_start,
        first_date=pd.Timestamp(dates[0]),
        last_date=pd.Timestamp(dates[-1]),
        universe_size=len(data),
        train_result=train,
        test_result=test,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        train_score=train_score,
        test_score=test_score,
        selection_score=selection_score,
        selection_std=selection_std,
        verdict=verdict,
        verdict_text=text,
        samples=samples,
        importances=importances,
        oob_r2=oob_r2,
        folds=folds,
        scored_ends=scored_ends,
    )


def book_strategies(found: list) -> list:
    """Book strategies a backtest studies. One-symbol files have no stages, so they are left out."""

    return [item for item in found if getattr(item, "stages", None)]


def backtest_books(on_study=None, progress=None) -> list[StudyResult]:
    """Choose indicator parameters for every book strategy and score the locked test window.

    A strategy file added later is included the next time this runs. The test
    window is not an input to any choice.
    """

    from engine.nse import available, listings, load_universe
    from strategies.registry import discover

    if not available():
        raise FileNotFoundError("The Nifty 500 price file is missing from nse500/daily-ohlcv/adjusted.")
    symbols = [item.symbol for item in listings()]
    data = load_universe(adjusted=True, symbols=symbols)
    if not data:
        raise ValueError("No listed name has daily bars")
    books = book_strategies(discover())
    if not books:
        raise ValueError("No book strategy is loaded")
    studies = []
    errors: list[str] = []
    for index, strategy in enumerate(books, start=1):
        def inner(done: int, total: int, label: str, index: int = index, count: int = len(books), name: str = strategy.name) -> None:
            if progress is not None:
                progress(index, count, name, done, total, label)

        try:
            study = study_book(strategy, data, progress=inner)
        except Exception as exc:
            errors.append(f"{strategy.key}: {exc}")
            continue
        if on_study is not None:
            on_study(study)
        studies.append(study)
    if errors:
        finished = ", ".join(study.strategy_key for study in studies) or "none"
        raise RuntimeError(f"Saved {finished}. " + " ".join(errors))
    return studies


def study_book(
    strategy: Strategy,
    data: dict[str, pd.DataFrame],
    n_samples: int = 24,
    n_folds: int = 4,
    n_shortlist: int = 4,
    n_candidates: int = 300,
    seed: int = 42,
    progress=None,
) -> StudyResult:
    """Choose indicator parameters on the train window, then score both windows once.

    The choice is the strategy's parameter set. Capital, costs, the stop, and
    entry rank stay at the file defaults. The test window is not an input.
    """

    names = indicator_names(strategy)
    if not names:
        return evaluate_book(strategy, data, progress=progress)
    return optimize_book(
        strategy,
        data,
        names,
        n_samples=n_samples,
        n_folds=n_folds,
        n_shortlist=n_shortlist,
        n_candidates=n_candidates,
        seed=seed,
        progress=progress,
    )


def evaluate_book(
    strategy: Strategy,
    data: dict[str, pd.DataFrame],
    params: dict | None = None,
    progress=None,
) -> StudyResult:
    """Score the strategy's own parameters on the train window and the test window."""

    if not data:
        raise ValueError("No bars to study")
    split_end, test_start = research_split(data)
    resolved = resolve_params(strategy.param_space(), params or {})
    ticker = _Ticker(len(data), progress, 2)
    return _finish(
        strategy,
        data,
        resolved,
        [],
        False,
        split_end,
        test_start,
        float("nan"),
        float("nan"),
        pd.DataFrame(),
        pd.DataFrame(columns=["parameter", "importance"]),
        float("nan"),
        [],
        [],
        ticker,
    )


def optimize_book(
    strategy: Strategy,
    data: dict[str, pd.DataFrame],
    searched: list[str],
    n_samples: int = 24,
    n_folds: int = 4,
    n_shortlist: int = 4,
    n_candidates: int = 300,
    seed: int = 42,
    progress=None,
) -> StudyResult:
    """Search `searched` on train folds. The test window is not an input to the choice."""

    if not data:
        raise ValueError("No bars to study")
    if n_samples < 1:
        raise ValueError("Use at least one sample")
    if n_folds < 2:
        raise ValueError("Use at least 2 cross-validation folds")
    space = strategy.param_space()
    names = []
    for name in searched:
        spec = space.get(name)
        if spec is None or not spec.searchable:
            raise ValueError(f"'{name}' has no search range")
        if name not in names:
            names.append(name)
    if not names:
        raise ValueError("Choose at least one parameter with a min and a max")

    split_end, test_start = research_split(data)
    dates = calendar(data)
    train_dates = dates[dates <= split_end]
    windows = fold_windows(train_dates, n_folds)
    base = repair_lengths(space, resolve_params(space, None))
    if base is None:
        raise ValueError("The default lengths cannot keep a fast average below the slow average")
    rng = np.random.default_rng(seed)

    samples = [dict(base)]
    seen = {_key(base, names)}
    budget = [n_samples * 40]
    while len(samples) < n_samples:
        trial = _draw_unseen(rng, space, names, base, seen, budget)
        if trial is None:
            break
        samples.append(trial)

    n_shortlist = max(int(n_shortlist), 0)
    ticker = _Ticker(len(data), progress, len(samples) * n_folds + n_shortlist * n_folds + 2)
    scored_ends: list[pd.Timestamp] = []
    rows = []
    remembered: dict[tuple, tuple[dict, list[float]]] = {}

    for index, trial in enumerate(samples, start=1):
        ticker.label_base = f"Sample {index} of {len(samples)}"
        ticker.label = ticker.label_base
        scores = _fold_scores(strategy, data, trial, windows, ticker, scored_ends)
        median, std = _summary(scores)
        remembered[_key(trial, names)] = (trial, scores)
        row = {"source": "sample", "cv_score": median, "cv_std": std}
        for name in names:
            row[name] = trial[name]
        rows.append(row)

    importances = pd.DataFrame(columns=["parameter", "importance"])
    oob_r2 = float("nan")
    proposals: list[dict] = []
    usable = [row for row in rows if _finite(row["cv_score"])]
    if len(usable) >= 3:
        matrix = np.vstack([_encode(remembered[_key_from_row(row, names)][0], names) for row in usable])
        target = np.array([row["cv_score"] for row in usable], dtype=float)
        model = _fit_forest(matrix, target, seed)
        raw_oob = getattr(model, "oob_score_", float("nan"))
        oob_r2 = float(raw_oob) if raw_oob is not None and math.isfinite(float(raw_oob)) else float("nan")
        importances = (
            pd.DataFrame({"parameter": names, "importance": model.feature_importances_})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        proposals = _propose(rng, space, names, base, seen, model, n_candidates, n_shortlist)

    for index, trial in enumerate(proposals, start=1):
        ticker.label_base = f"Forest proposal {index} of {len(proposals)}"
        ticker.label = ticker.label_base
        scores = _fold_scores(strategy, data, trial, windows, ticker, scored_ends)
        median, std = _summary(scores)
        remembered[_key(trial, names)] = (trial, scores)
        row = {"source": "forest", "cv_score": median, "cv_std": std}
        for name in names:
            row[name] = trial[name]
        rows.append(row)

    ticker.n_runs = ticker.done_runs + 2
    if not rows:
        raise RuntimeError("The search produced no parameter sets")
    ranked = sorted(rows, key=lambda row: (row["cv_score"], -row["cv_std"]), reverse=True)
    best_row = ranked[0]
    if not _finite(best_row["cv_score"]):
        raise RuntimeError("No parameter set produced a finite CAGR / drawdown. Widen the ranges or loosen the filter.")
    winner, winner_scores = remembered[_key_from_row(best_row, names)]
    folds = [
        FoldRead(fold=number, start=start, end=end, score=score)
        for number, ((start, end), score) in enumerate(zip(windows, winner_scores), start=1)
    ]
    return _finish(
        strategy,
        data,
        winner,
        names,
        True,
        split_end,
        test_start,
        best_row["cv_score"],
        best_row["cv_std"],
        pd.DataFrame(rows),
        importances,
        oob_r2,
        folds,
        scored_ends,
        ticker,
    )


def _key_from_row(row: dict, names: list[str]) -> tuple:
    return tuple(row[name] for name in names)


def tagged_trades(study: StudyResult) -> pd.DataFrame:
    """Closed trades from both windows, with a train or test label."""

    frames = []
    for window, result in (("train", study.train_result), ("test", study.test_result)):
        trades = result.trades
        if trades is None or trades.empty:
            continue
        copied = trades.copy()
        copied["window"] = window
        frames.append(copied)
    if not frames:
        return pd.DataFrame(
            columns=[
                "entry_time",
                "exit_time",
                "side",
                "entry_price",
                "exit_price",
                "shares",
                "pnl",
                "return_pct",
                "symbol",
                "reason",
                "window",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def traded_symbols(study: StudyResult) -> list[str]:
    trades = tagged_trades(study)
    if trades.empty or "symbol" not in trades.columns:
        return []
    return sorted(str(symbol) for symbol in trades["symbol"].dropna().unique())
