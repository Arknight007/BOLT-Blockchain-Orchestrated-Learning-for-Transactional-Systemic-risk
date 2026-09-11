"""Automated monitoring cycle (ChainGuard automation layer).

One cycle does three things, in this order:

1. **Resolve** every prediction whose horizon has now closed, using the
   Evaluation Agent, and append the outcome to the ledger.
2. **Predict** for every target asset as of the run date, running the full agent
   chain, and append each prediction (optionally committing it on-chain).
3. **Monitor** with the Health Agent: feature drift against the training
   distribution, metric decay across folds, and per-agent reliability.

Resolving comes FIRST on purpose. A cycle that predicted before scoring its own
previous calls would let today's prediction be made in ignorance of yesterday's
failures - and the health signal would always lag by a full cycle.

This is what makes the Evaluation and Health agents real rather than
theoretical: without a loop to live in, they are code nobody ever calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from bolt.agents.base import AgentReport
from bolt.agents.evaluation import EvaluationAgent, Outcome
from bolt.agents.health import HealthAgent
from bolt.agents.pipeline import ChainGuardPipeline, build_context
from bolt.config import BoltConfig
from bolt.logging_setup import get_logger
from bolt.store import PredictionStore

log = get_logger(__name__)

ALERT_BANDS = ("HIGH", "CRITICAL")


@dataclass
class CycleResult:
    """Everything one monitoring cycle produced."""

    as_of: str
    predictions: list[dict] = field(default_factory=list)
    outcomes: list[Outcome] = field(default_factory=list)
    evaluation: AgentReport | None = None
    health: AgentReport | None = None
    errors: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        alerts = [
            p for p in self.predictions
            if p["payload"]["severity_band"] in ALERT_BANDS
        ]
        return {
            "as_of": self.as_of,
            "predictions": len(self.predictions),
            "alerts": len(alerts),
            "alert_assets": [p["payload"]["asset"] for p in alerts],
            "resolved": len(self.outcomes),
            "committed": sum(1 for p in self.predictions if p.get("committed")),
            "errors": self.errors,
            "health": (
                self.health.extra.get("recommendation") if self.health else None
            ),
        }


def resolve_matured(
    cfg: BoltConfig, store: PredictionStore, panel: pd.DataFrame, as_of: pd.Timestamp
) -> tuple[list[Outcome], AgentReport | None]:
    """Score every prediction whose horizon has closed on or before ``as_of``."""
    agent = EvaluationAgent(threshold=cfg.drawdown_threshold, alert_bands=ALERT_BANDS)
    resolved: list[Outcome] = []

    for record in store.unresolved():
        payload = record["payload"]
        matures = pd.Timestamp(payload["as_of_date"], tz="UTC") + pd.Timedelta(
            int(payload["horizon_days"]), "D"
        )
        if matures > as_of:
            continue  # the horizon has not closed; the prediction still stands

        try:
            prices = panel.xs(payload["asset"], level="asset")["close"]
        except KeyError:
            log.warning("%s not in the panel; cannot resolve %s",
                        payload["asset"], payload["prediction_id"][:18])
            continue

        outcome = agent.resolve(payload, prices)
        if not outcome.resolved:
            continue
        store.record_outcome(outcome.to_dict())
        resolved.append(outcome)
        log.info("resolved %s: %s", outcome.prediction_id[:18], outcome.classification)

    if not resolved:
        return [], None

    all_outcomes = [Outcome.from_dict(o) for o in store.outcomes()]
    return resolved, agent.analyse(all_outcomes)


def predict_all(
    cfg: BoltConfig,
    store: PredictionStore,
    panel: pd.DataFrame,
    as_of: pd.Timestamp,
    models: dict,
    scaler,
    commit: bool = False,
    analogue_source=None,
) -> tuple[list[dict], list[str]]:
    """Run the agent chain for every target asset and append each prediction.

    ``analogue_source`` activates the Skeptic's precedent check; without it that
    challenge never fires.
    """
    pipeline = ChainGuardPipeline(cfg, models, scaler, analogue_source=analogue_source)
    features = list(cfg.feature_columns)
    records: list[dict] = []
    errors: list[str] = []

    for asset in cfg.target_assets:
        try:
            context = build_context(cfg, panel, asset.symbol, as_of, features)
        except ValueError as exc:
            errors.append(f"{asset.symbol}: {exc}")
            log.warning("%s: %s", asset.symbol, exc)
            continue

        try:
            result = pipeline.run(context, commit=commit)
        except Exception as exc:  # noqa: BLE001 - one asset must not kill the cycle
            errors.append(f"{asset.symbol}: {type(exc).__name__}: {exc}")
            log.exception("%s: agent chain failed", asset.symbol)
            continue

        commitment = result.commitment
        record = store.record_prediction(
            payload=commitment.prediction.to_dict(),
            digest=commitment.digest,
            committed=commitment.committed,
            tx_hash=commitment.tx_hash,
            block_time=commitment.block_time,
        )
        records.append(record)
        log.info("%-6s %s  %s %.0f/100 (confidence %.0f%%)",
                 asset.symbol, as_of.date(), result.decision.risk.value,
                 result.decision.score, 100 * result.decision.confidence)

    return records, errors


def check_health(
    cfg: BoltConfig, X: np.ndarray, meta: pd.DataFrame, as_of: pd.Timestamp
) -> AgentReport | None:
    """Feature drift and metric decay. Returns None when there is nothing to compare."""
    agent = HealthAgent()
    window_end = pd.to_datetime(meta["window_end"], utc=True)

    # Reference = the training era; current = the most recent 180 days available.
    boundary = pd.Timestamp(cfg.folds[-1].test_start, tz="UTC")
    reference_mask = (window_end < boundary).to_numpy()
    current_mask = (window_end >= as_of - pd.Timedelta(180, "D")).to_numpy()

    drift = None
    if reference_mask.sum() > 200 and current_mask.sum() > 30:
        drift = agent.feature_drift(
            X[reference_mask], X[current_mask], list(cfg.feature_columns)
        )

    performance = None
    per_fold_path = cfg.path("tables") / "model_comparison_per_fold.csv"
    if per_fold_path.is_file():
        performance = agent.performance_trend(pd.read_csv(per_fold_path))

    if drift is None and performance is None:
        return None
    return agent.analyse(drift=drift, performance=performance)


def run_cycle(
    cfg: BoltConfig,
    as_of: pd.Timestamp | str | None = None,
    commit: bool = False,
    store: PredictionStore | None = None,
) -> CycleResult:
    """One full monitoring cycle: resolve, then predict, then monitor."""
    from bolt.commands import load_models, load_panel, load_windows

    panel = load_panel(cfg, targets_only=False)
    available = panel.index.get_level_values("date").max()

    if as_of is None:
        as_of = available
    else:
        stamp = pd.Timestamp(as_of)
        as_of = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    if as_of > available:
        log.warning("as-of %s is beyond the frozen dataset (%s); using %s",
                    as_of.date(), available.date(), available.date())
        as_of = available

    result = CycleResult(as_of=as_of.date().isoformat())
    store = store or PredictionStore(cfg.path("predictions") / "ledger")

    # 1. Resolve before predicting, so today's run knows yesterday's failures.
    try:
        result.outcomes, result.evaluation = resolve_matured(cfg, store, panel, as_of)
    except Exception as exc:  # noqa: BLE001
        result.errors.append(f"resolve: {type(exc).__name__}: {exc}")
        log.exception("resolution step failed")

    # 2. Predict.
    quant_models = list(cfg.section("agents")["quant_models"])
    try:
        models, scaler = load_models(cfg, quant_models)
    except FileNotFoundError as exc:
        result.errors.append(str(exc))
        log.warning("%s -- the quantitative agent will report UNAVAILABLE", exc)
        models, scaler = {}, None

    # Windows serve two purposes below: the Skeptic's precedent check and the
    # drift comparison. Load once.
    windows = None
    try:
        windows = load_windows(cfg)
    except Exception as exc:  # noqa: BLE001
        result.errors.append(f"windows: {type(exc).__name__}: {exc}")
        log.warning("windows unavailable (%s); precedent check and drift disabled", exc)

    analogue_source = None
    if windows is not None:
        X, _, meta = windows
        # Unscaled: the pipeline scales only the masked asset slice.
        analogue_source = (X, meta, panel["close"])

    predictions, errors = predict_all(
        cfg, store, panel, as_of, models, scaler, commit, analogue_source
    )
    result.predictions = predictions
    result.errors.extend(errors)

    # 3. Monitor.
    if windows is not None:
        try:
            X, _, meta = windows
            result.health = check_health(cfg, X, meta, as_of)
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"health: {type(exc).__name__}: {exc}")
            log.exception("health step failed")

    store.record_run(result.summary())
    return result
