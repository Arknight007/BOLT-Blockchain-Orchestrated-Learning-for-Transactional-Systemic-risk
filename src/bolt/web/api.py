"""ChainGuard terminal API (read-only over real pipeline artefacts).

Every endpoint here reads something the pipeline actually produced: the frozen
parquet, the generated tables, the prediction ledger, the live agent chain. None
of it is mocked or pre-baked. If `bolt build` has not run, the endpoints say so
rather than inventing a dataset (Rule 12.2, Rule 12.3).

The console is a VIEW over the system, not a second implementation of it. A
dashboard that recomputed its own numbers would be a second place for results to
diverge from the pipeline, which is exactly the failure this project exists to
avoid.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from bolt.config import BoltConfig
from bolt.logging_setup import get_logger
from bolt.store import PredictionStore
from bolt.version import __version__, git_commit_sha

log = get_logger(__name__)


def _clean(value: Any) -> Any:
    """JSON-safe conversion. NaN becomes null, never the string 'NaN'."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else round(float(value), 6)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if value is pd.NaT or value is None:
        return None
    return value


def _frame(df: pd.DataFrame) -> list[dict]:
    return _clean(df.replace({np.nan: None}).to_dict("records"))


def build_router(cfg: BoltConfig) -> APIRouter:
    router = APIRouter(prefix="/api")
    state = _State(cfg)

    # -- system ---------------------------------------------------------
    @router.get("/status")
    def status() -> dict:
        """What exists on disk, so the console can say what is missing."""
        processed = cfg.path("processed")
        tables = cfg.path("tables")
        panel_path = processed / "panel.parquet"
        models_dir = processed / "models"

        artefacts = {
            "panel": panel_path.is_file(),
            "data_card": (processed / "DATA_CARD.md").is_file(),
            "models": sorted(p.stem for p in models_dir.glob("*.pkl") if p.stem != "scaler")
            if models_dir.is_dir() else [],
            "model_comparison": (tables / "model_comparison.csv").is_file(),
            "consistency": (tables / "attribution_consistency.csv").is_file(),
        }
        digest = (
            hashlib.sha256(panel_path.read_bytes()).hexdigest()
            if panel_path.is_file() else None
        )
        ready = artefacts["panel"] and bool(artefacts["models"])
        return _clean({
            "version": __version__,
            "code_version": git_commit_sha(cfg.repo_root),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "ready": ready,
            "artefacts": artefacts,
            "dataset_sha256": digest,
            "config": {
                "threshold": cfg.drawdown_threshold,
                "horizon_days": cfg.horizon_days,
                "lookback_days": cfg.lookback_days,
                "embargo_days": cfg.embargo_days,
                "primary": cfg.primary_asset,
                "features": len(cfg.feature_columns),
                "seed": cfg.seed,
            },
            "missing": [k for k, v in artefacts.items() if not v],
        })

    @router.get("/universe")
    def universe() -> dict:
        panel = state.panel()
        rows = []
        for asset in cfg.assets:
            block = (
                panel.xs(asset.symbol, level="asset")
                if asset.symbol in panel.index.get_level_values("asset") else None
            )
            rows.append({
                "symbol": asset.symbol,
                "name": asset.name,
                "role": asset.role,
                "inception": asset.inception.isoformat(),
                "rows": 0 if block is None else len(block),
                "first": None if block is None else block.index.min().date().isoformat(),
                "last": None if block is None else block.index.max().date().isoformat(),
                "positive_labels": 0 if block is None
                else int(block["label"].sum(skipna=True)),
                "last_close": None if block is None else float(block["close"].iloc[-1]),
            })
        return _clean({
            "assets": rows,
            "episodes": [
                {"name": e.name, "start": e.start.isoformat(), "end": e.end.isoformat()}
                for e in cfg.episodes
            ],
        })

    # -- results --------------------------------------------------------
    @router.get("/models")
    def models() -> dict:
        comparison = state.table("model_comparison.csv", index_col=0)
        per_fold = state.table("model_comparison_per_fold.csv")
        lead = state.table("lead_time.csv")
        if comparison is None:
            raise HTTPException(404, "run `bolt evaluate --all --report` first")

        comparison = comparison.reset_index().rename(columns={"index": "model"})
        baseline = (
            float(per_fold["pr_auc_baseline"].mean())
            if per_fold is not None and "pr_auc_baseline" in per_fold else None
        )
        return _clean({
            "comparison": _frame(comparison),
            "per_fold": _frame(per_fold) if per_fold is not None else [],
            "lead_time": _frame(lead) if lead is not None else [],
            "baseline_pr_auc": baseline,
        })

    @router.get("/labels")
    def labels() -> dict:
        table = state.table("label_sensitivity.csv")
        if table is None:
            raise HTTPException(404, "run `bolt build` first")
        return _clean({
            "sensitivity": _frame(table),
            "default": {
                "threshold": cfg.drawdown_threshold,
                "horizon_days": cfg.horizon_days,
            },
        })

    @router.get("/consistency")
    def consistency() -> dict:
        summary = state.table("attribution_consistency.csv")
        spearman = state.table("attribution_spearman_matrix.csv", index_col=0)
        jaccard = state.table("attribution_jaccard_matrix.csv", index_col=0)
        profiles = state.table("attribution_profiles_by_episode.csv", index_col=0)
        top = state.table("top_features_per_episode.csv")
        if summary is None or spearman is None:
            raise HTTPException(404, "run `bolt explain --model xgb --consistency` first")

        return _clean({
            "summary": _frame(summary)[0],
            "episodes": list(spearman.columns),
            "spearman": spearman.to_numpy().tolist(),
            "jaccard": jaccard.to_numpy().tolist() if jaccard is not None else [],
            "profiles": {
                column: {
                    "features": profiles.index.tolist(),
                    "values": profiles[column].tolist(),
                }
                for column in profiles.columns
            } if profiles is not None else {},
            "top_features": _frame(top) if top is not None else [],
        })

    @router.get("/attribution")
    def attribution(model: str = Query("xgb")) -> dict:
        table = state.table(f"attribution_{model}.csv")
        timestep = state.table(f"attribution_timestep_{model}.csv", index_col=0)
        if table is None:
            raise HTTPException(404, f"run `bolt explain --model {model}` first")
        return _clean({
            "model": model,
            "features": _frame(table),
            "timestep": {
                "positions": timestep.index.tolist(),
                "values": timestep.iloc[:, 0].tolist(),
            } if timestep is not None else None,
        })

    # -- market ---------------------------------------------------------
    @router.get("/timeline")
    def timeline(asset: str = Query(None), limit: int = Query(2200)) -> dict:
        asset = asset or cfg.primary_asset
        panel = state.panel()
        symbols = set(panel.index.get_level_values("asset"))
        if asset not in symbols:
            raise HTTPException(404, f"{asset} is not in the panel")

        block = panel.xs(asset, level="asset").tail(limit)
        return _clean({
            "asset": asset,
            "dates": [d.date().isoformat() for d in block.index],
            "close": block["close"].tolist(),
            "label": block["label"].fillna(-1).tolist(),
            "realized_vol_30": block.get(
                "realized_vol_30", pd.Series(index=block.index, dtype=float)
            ).tolist(),
            "drawdown_depth": block.get(
                "drawdown_depth", pd.Series(index=block.index, dtype=float)
            ).tolist(),
            "episodes": [
                {"name": e.name, "start": e.start.isoformat(), "end": e.end.isoformat()}
                for e in cfg.episodes
            ],
        })

    @router.get("/features")
    def features(asset: str = Query(None), as_of: str = Query(None)) -> dict:
        """Current feature vector with each value's percentile in its own history."""
        asset = asset or cfg.primary_asset
        panel = state.panel()
        try:
            block = panel.xs(asset, level="asset")
        except KeyError:
            raise HTTPException(404, f"{asset} is not in the panel")

        if as_of:
            block = block.loc[: pd.Timestamp(as_of, tz="UTC")]
        if block.empty:
            raise HTTPException(404, f"no data for {asset} on or before {as_of}")

        latest = block.iloc[-1]
        from bolt.ingest.provenance import REGISTRY

        rows = []
        for name in cfg.feature_columns:
            value = float(latest.get(name, np.nan))
            history = block[name].dropna() if name in block else pd.Series(dtype=float)
            percentile = (
                float((history <= value).mean())
                if len(history) and np.isfinite(value) else None
            )
            source = REGISTRY.get(name)
            rows.append({
                "feature": name,
                "value": value,
                "percentile": percentile,
                "family": _family(cfg, name),
                "provenance": source.provenance.value if source else "DERIVED",
            })
        return _clean({
            "asset": asset,
            "as_of": block.index[-1].date().isoformat(),
            "features": rows,
        })

    return router, state


def _family(cfg: BoltConfig, name: str) -> str:
    families = cfg.section("features")
    for family in ("technical", "onchain", "sentiment", "contagion"):
        if name in families[family]:
            return family
    return "other"


class _State:
    """Lazily-loaded, cached access to pipeline artefacts."""

    def __init__(self, cfg: BoltConfig) -> None:
        self.cfg = cfg
        self._panel: pd.DataFrame | None = None
        self._models: tuple[dict, Any] | None = None

    def panel(self) -> pd.DataFrame:
        if self._panel is None:
            path = self.cfg.path("processed") / "panel.parquet"
            if not path.is_file():
                raise HTTPException(
                    503, "no frozen dataset. Run `bolt build` first; the console "
                         "will not invent one."
                )
            self._panel = pd.read_parquet(path)
        return self._panel

    def models(self) -> tuple[dict, Any]:
        if self._models is None:
            from bolt.commands import load_models

            names = list(self.cfg.section("agents")["quant_models"])
            try:
                self._models = load_models(self.cfg, names)
            except FileNotFoundError:
                self._models = ({}, None)
        return self._models

    def table(self, name: str, index_col=None) -> pd.DataFrame | None:
        path = self.cfg.path("tables") / name
        if not path.is_file():
            return None
        return pd.read_csv(path, index_col=index_col)

    def store(self) -> PredictionStore:
        return PredictionStore(self.cfg.path("predictions") / "ledger")
