"""Model comparison tables and figures (BOLT_SPEC.md Section 7).

Produces ``outputs/tables/model_comparison.csv`` with one row per model and mean
+/- std across folds, the per-fold detail behind it, and a matching figure.

Per-fold results are not optional detail. A model that wins on average but fails
on the FTX fold is an important finding, and an aggregate alone would hide it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from bolt.config import BoltConfig
from bolt.evaluate.metrics import (
    aggregate_folds,
    baseline_pr_auc,
    compute_metrics,
    find_best_threshold,
    guard_metrics,
    lead_time_days,
)
from bolt.evaluate.splits import SplitResult, TrainOnlyScaler, walk_forward_splits
from bolt.logging_setup import get_logger
from bolt.models.classical import LogRegModel, MLPModel, RandomForestModel
from bolt.models.gbm import XGBModel
from bolt.models.gru_numpy import NumpyGRU
from bolt.models.lstm_numpy import NumpyLSTM
from bolt.models.rule import VolatilityRule

log = get_logger(__name__)

REPORT_COLUMNS = ["pr_auc", "f1", "precision", "recall", "false_alarm_rate", "brier"]

#: Fraction of each training fold held out, chronologically, to calibrate the
#: decision threshold out of sample.
CALIBRATION_FRACTION = 0.20


def _calibration_split(
    meta: pd.DataFrame, train_idx: np.ndarray, embargo_days: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split a training fold chronologically into (fit, calibrate) positions.

    Returns POSITIONS within ``train_idx``, not global row indices. The same
    embargo that separates train from test also separates fit from calibrate,
    so the threshold is tuned on windows whose labels the fitted model could not
    have seen.
    """
    window_start = pd.to_datetime(meta.loc[train_idx, "window_start"], utc=True)
    order = np.argsort(window_start.to_numpy())
    ordered_dates = window_start.to_numpy()[order]

    cut = int(len(order) * (1.0 - CALIBRATION_FRACTION))
    if cut <= 0 or cut >= len(order):
        return np.arange(len(train_idx)), np.array([], dtype=int)

    boundary = ordered_dates[cut]
    label_end = pd.to_datetime(meta.loc[train_idx, "label_end"], utc=True).to_numpy()[order]
    # Fit rows must resolve before the calibration window opens.
    fit_mask = label_end < (boundary - np.timedelta64(embargo_days, "D"))
    fit_positions = order[fit_mask]
    cal_positions = order[cut:]

    if len(fit_positions) < 50:
        return np.arange(len(train_idx)), np.array([], dtype=int)
    return fit_positions, cal_positions


def model_registry(cfg: BoltConfig) -> dict[str, Callable[[], object]]:
    """Every benchmarked model, constructed from configured parameters only."""
    p = cfg.raw["models"]
    jobs = int(p.get("n_jobs", 1))
    return {
        "lstm":   lambda: NumpyLSTM(**p["lstm"]),
        "gru":    lambda: NumpyGRU(**p["gru"]),
        "xgb":    lambda: XGBModel(**p["xgb"], n_jobs=jobs),
        "rf":     lambda: RandomForestModel(**p["rf"], n_jobs=jobs),
        "logreg": lambda: LogRegModel(**p["logreg"]),
        "mlp":    lambda: MLPModel(**p["mlp"]),
        "rule":   lambda: VolatilityRule(**p["rule"]),
    }


@dataclass
class EvaluationResult:
    per_fold: pd.DataFrame
    aggregate: pd.DataFrame
    predictions: dict[str, pd.DataFrame] = field(default_factory=dict)
    lead_times: pd.DataFrame | None = None


def evaluate_model(
    name: str,
    factory: Callable[[], object],
    X: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    splits: list[SplitResult],
    cfg: BoltConfig,
) -> tuple[dict[str, dict[str, float]], pd.DataFrame]:
    """Fit and score one model on every fold. Returns (per-fold metrics, predictions)."""
    per_fold: dict[str, dict[str, float]] = {}
    frames: list[pd.DataFrame] = []

    for split in splits:
        X_train, y_train = X[split.train_idx], y[split.train_idx]
        X_test, y_test = X[split.test_idx], y[split.test_idx]

        # Guard 5: fitted on the training fold, applied to test. Never the reverse.
        scaler = TrainOnlyScaler().fit(X_train)
        assert scaler.n_samples_seen_ == len(X_train)
        X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

        # The operating point must be tuned OUT OF SAMPLE or it is useless.
        # Tuning it on the training predictions themselves produced thresholds a
        # tree ensemble never reaches on unseen data (measured: RF and XGB scored
        # f1=0.000, recall=0.000 across every fold because they never fired).
        # Tuning it on TEST scores would be leakage. So the training fold is split
        # chronologically into a fit part and a calibration part, with the same
        # embargo applied between them, and the threshold comes from the
        # calibration part - unseen by the model, and still strictly in the past.
        fit_idx, cal_idx = _calibration_split(
            meta, split.train_idx, cfg.embargo_days
        )
        model = factory()
        model.fit(X_train_s[fit_idx], y_train[fit_idx])

        if len(cal_idx) and len(np.unique(y_train[cal_idx])) > 1:
            threshold, _ = find_best_threshold(
                y_train[cal_idx], model.predict_proba(X_train_s[cal_idx])
            )
        else:
            threshold = 0.5
            log.warning("%s %s: calibration slice unusable; falling back to 0.5",
                        name, split.name)

        scores = model.predict_proba(X_test_s)
        metrics = compute_metrics(y_test, scores, threshold)
        metrics["threshold"] = threshold
        metrics["pr_auc_baseline"] = baseline_pr_auc(y_test)
        metrics["pr_auc_lift"] = metrics["pr_auc"] - metrics["pr_auc_baseline"]
        per_fold[split.name] = metrics

        frames.append(pd.DataFrame({
            "fold": split.name,
            "model": name,
            "date": meta.loc[split.test_idx, "window_end"].to_numpy(),
            "asset": meta.loc[split.test_idx, "asset"].to_numpy(),
            "y_true": y_test,
            "y_score": scores,
            "threshold": threshold,
        }))
        log.info("  %-7s %-16s pr_auc=%.3f (base %.3f) f1=%.3f recall=%.3f far=%.3f",
                 name, split.name, metrics["pr_auc"], metrics["pr_auc_baseline"],
                 metrics["f1"], metrics["recall"], metrics["false_alarm_rate"])

    return per_fold, pd.concat(frames, ignore_index=True)


def run_evaluation(
    cfg: BoltConfig,
    X: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    models: list[str] | None = None,
) -> EvaluationResult:
    """Full walk-forward evaluation across the requested models."""
    registry = model_registry(cfg)
    names = list(registry) if not models or "all" in models else list(models)
    unknown = set(names) - set(registry)
    if unknown:
        raise ValueError(f"unknown model(s) {sorted(unknown)}; known: {sorted(registry)}")

    splits = walk_forward_splits(meta, list(cfg.folds), cfg.embargo_days)

    rows: list[dict] = []
    aggregates: dict[str, pd.Series] = {}
    predictions: dict[str, pd.DataFrame] = {}

    for name in names:
        log.info("evaluating %s", name)
        per_fold, frame = evaluate_model(name, registry[name], X, y, meta, splits, cfg)
        predictions[name] = frame
        aggregates[name] = aggregate_folds(per_fold)
        for fold_name, metrics in per_fold.items():
            rows.append({"model": name, "fold": fold_name, **metrics})

    per_fold_frame = pd.DataFrame(rows)
    aggregate_frame = pd.DataFrame(aggregates).T
    aggregate_frame.index.name = "model"
    guard_metrics(per_fold_frame.columns)

    lead = _lead_time_table(cfg, predictions)
    return EvaluationResult(per_fold_frame, aggregate_frame, predictions, lead)


def _lead_time_table(cfg: BoltConfig, predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Lead time per model per episode, on the primary asset."""
    rows = []
    primary = cfg.primary_asset
    for name, frame in predictions.items():
        block = frame[frame["asset"] == primary]
        if block.empty:
            continue
        series = block.set_index(pd.DatetimeIndex(block["date"]))["y_score"].sort_index()
        threshold = float(block["threshold"].iloc[0])
        table = lead_time_days(
            series, threshold, list(cfg.episodes),
            int(cfg.raw["evaluation"]["lead_time"]["sustained_days"]),
        )
        table.insert(0, "model", name)
        rows.append(table)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def write_report(cfg: BoltConfig, result: EvaluationResult) -> dict[str, Path]:
    """Write every table and figure. Returns the paths written."""
    tables = cfg.path("tables")
    figures = cfg.path("figures")
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}

    summary = _summary_table(result.aggregate)
    summary.to_csv(tables / "model_comparison.csv")
    written["model_comparison"] = tables / "model_comparison.csv"

    result.per_fold.to_csv(tables / "model_comparison_per_fold.csv", index=False)
    written["per_fold"] = tables / "model_comparison_per_fold.csv"

    if result.lead_times is not None and not result.lead_times.empty:
        result.lead_times.to_csv(tables / "lead_time.csv", index=False)
        written["lead_time"] = tables / "lead_time.csv"

    written["figure"] = _comparison_figure(result, figures)
    return written


def _summary_table(aggregate: pd.DataFrame) -> pd.DataFrame:
    """Human-readable mean +/- std per metric."""
    out = pd.DataFrame(index=aggregate.index)
    for metric in REPORT_COLUMNS:
        mean, std = f"{metric}_mean", f"{metric}_std"
        if mean in aggregate.columns:
            out[metric] = [
                f"{m:.3f} +/- {s:.3f}"
                for m, s in zip(aggregate[mean], aggregate[std])
            ]
    for extra in ("pr_auc_baseline_mean", "pr_auc_lift_mean"):
        if extra in aggregate.columns:
            out[extra.replace("_mean", "")] = aggregate[extra].round(4)
    if "n_samples" in aggregate.columns:
        out["n_test_samples"] = aggregate["n_samples"].astype(int)
    return out.sort_values("pr_auc", ascending=False)


def _comparison_figure(result: EvaluationResult, figures: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    aggregate = result.aggregate
    metrics = [m for m in ["pr_auc", "f1", "recall", "precision"] if f"{m}_mean" in aggregate]
    order = aggregate["pr_auc_mean"].sort_values(ascending=False).index

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.0 * len(metrics), 4.4), sharey=False)
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        means = aggregate.loc[order, f"{metric}_mean"]
        errors = aggregate.loc[order, f"{metric}_std"]
        ax.barh(range(len(order)), means, xerr=errors, color="#4C72B0",
                edgecolor="black", linewidth=0.6, capsize=3)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order)
        ax.invert_yaxis()
        ax.set_title(metric.replace("_", " ").upper())
        ax.grid(axis="x", alpha=0.3)
        if metric == "pr_auc" and "pr_auc_baseline_mean" in aggregate:
            base = float(aggregate["pr_auc_baseline_mean"].mean())
            ax.axvline(base, color="crimson", linestyle="--", linewidth=1.2)
            ax.text(base, -0.6, f" random = {base:.3f}", color="crimson", fontsize=8)

    fig.suptitle("Model comparison, walk-forward with embargo (mean +/- std across folds)")
    fig.tight_layout()
    path = figures / "model_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
