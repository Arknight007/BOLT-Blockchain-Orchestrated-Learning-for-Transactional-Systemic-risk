"""Command implementations behind the CLI (BOLT_SPEC.md Section 10).

Kept separate from ``cli.py`` so the argument surface stays readable and each
command is importable and testable on its own.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from bolt.config import BoltConfig
from bolt.logging_setup import get_logger

log = get_logger(__name__)

MODELS_DIR = "models"
WINDOWS_CACHE = "windows.npz"


def _panel_path(cfg: BoltConfig) -> Path:
    path = cfg.path("processed") / "panel.parquet"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Run `bolt build` first to produce the frozen dataset."
        )
    return path


def load_panel(cfg: BoltConfig, targets_only: bool = True) -> pd.DataFrame:
    panel = pd.read_parquet(_panel_path(cfg))
    if targets_only:
        targets = {a.symbol for a in cfg.target_assets}
        panel = panel[panel.index.get_level_values("asset").isin(targets)]
    return panel


def load_windows(cfg: BoltConfig) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build (or reload) the window tensors."""
    from bolt.windows import build_windows

    cache = cfg.path("processed") / WINDOWS_CACHE
    meta_cache = cfg.path("processed") / "windows_meta.parquet"
    if cache.is_file() and meta_cache.is_file():
        stored = np.load(cache)
        log.info("loaded cached windows from %s", cache)
        return stored["X"], stored["y"], pd.read_parquet(meta_cache)

    panel = load_panel(cfg)
    X, y, meta = build_windows(
        panel, cfg.lookback_days, list(cfg.feature_columns), "label", cfg.horizon_days,
        int(cfg.raw["windows"]["stride"]),
    )
    np.savez_compressed(cache, X=X, y=y)
    meta.to_parquet(meta_cache)
    return X, y, meta


# ---------------------------------------------------------------------------
# bolt ingest / build
# ---------------------------------------------------------------------------

def do_ingest(cfg: BoltConfig, refresh: bool = False) -> int:
    from bolt.align.build import ingest_all
    from bolt.ingest.cache import DiskCache

    cache = DiskCache(cfg.path("raw"))
    raw = ingest_all(cfg, cache, refresh=refresh)
    log.info("ingest complete: %s", cache.summary())
    for symbol, frame in raw["ohlcv"].items():
        log.info("  %-6s %5d rows  %s -> %s  source=%s", symbol, len(frame),
                 frame.index.min().date(), frame.index.max().date(), frame["source"].iloc[0])
    if raw["excluded_assets"]:
        log.warning("excluded (no source covers the window): %s", raw["excluded_assets"])
    return 0


def do_build(cfg: BoltConfig, refresh: bool = False, skip_windows: bool = False) -> int:
    from bolt.align.build import run_build

    result = run_build(cfg, refresh=refresh)
    log.info("panel: %s -> %s", result.panel.shape, result.panel_path)
    log.info("sha256: %s", result.sha256)
    log.info("data card: %s", result.data_card)

    if not skip_windows:
        for stale in (cfg.path("processed") / WINDOWS_CACHE,
                      cfg.path("processed") / "windows_meta.parquet"):
            stale.unlink(missing_ok=True)
        X, y, _ = load_windows(cfg)
        log.info("windows: %s, %d positive (%.2f%%)", X.shape, int(y.sum()), 100 * y.mean())
    return 0


# ---------------------------------------------------------------------------
# bolt train / evaluate
# ---------------------------------------------------------------------------

def do_train(cfg: BoltConfig, model: str = "all") -> int:
    """Fit models on ALL data up to the final fold's train_end and persist them.

    These are the artefacts `bolt predict` uses. Evaluation refits per fold; this
    is the deployment fit.
    """
    from bolt.evaluate.report import model_registry
    from bolt.evaluate.splits import TrainOnlyScaler

    X, y, meta = load_windows(cfg)
    registry = model_registry(cfg)
    names = list(registry) if model == "all" else [model]

    # Train on everything whose label resolves before the last fold's test start,
    # minus the embargo. A deployment model must respect the same boundary.
    boundary = pd.Timestamp(cfg.folds[-1].test_start, tz="UTC") - pd.Timedelta(
        cfg.embargo_days, "D"
    )
    label_end = pd.to_datetime(meta["label_end"], utc=True)
    train_mask = (label_end < boundary).to_numpy()
    log.info("training on %d of %d windows (labels resolving before %s)",
             int(train_mask.sum()), len(X), boundary.date())

    scaler = TrainOnlyScaler().fit(X[train_mask])
    X_scaled = scaler.transform(X[train_mask])

    out_dir = cfg.path("processed") / MODELS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "scaler.pkl").open("wb") as handle:
        pickle.dump(scaler, handle)

    for name in names:
        log.info("fitting %s", name)
        fitted = registry[name]()
        fitted.fit(X_scaled, y[train_mask])
        with (out_dir / f"{name}.pkl").open("wb") as handle:
            pickle.dump(fitted, handle)
        log.info("  saved %s", out_dir / f"{name}.pkl")
    return 0


def load_models(cfg: BoltConfig, names: list[str] | None = None) -> tuple[dict, object]:
    """Load persisted models and the train-fitted scaler."""
    out_dir = cfg.path("processed") / MODELS_DIR
    scaler_path = out_dir / "scaler.pkl"
    if not scaler_path.is_file():
        raise FileNotFoundError(f"{scaler_path} not found. Run `bolt train` first.")
    with scaler_path.open("rb") as handle:
        scaler = pickle.load(handle)

    models = {}
    for path in sorted(out_dir.glob("*.pkl")):
        if path.stem == "scaler":
            continue
        if names and path.stem not in names:
            continue
        with path.open("rb") as handle:
            models[path.stem] = pickle.load(handle)
    if not models:
        raise FileNotFoundError(f"no fitted models in {out_dir}. Run `bolt train` first.")
    return models, scaler


def do_evaluate(cfg: BoltConfig, models: list[str] | None, report: bool = True) -> int:
    from bolt.evaluate.report import run_evaluation, write_report, _summary_table

    X, y, meta = load_windows(cfg)
    result = run_evaluation(cfg, X, y, meta, models)

    print("\n=== MODEL COMPARISON (walk-forward, embargoed) ===")
    print(_summary_table(result.aggregate).to_string())
    if result.lead_times is not None and not result.lead_times.empty:
        print("\n=== LEAD TIME PER EPISODE (primary asset) ===")
        print(result.lead_times.to_string(index=False))

    if report:
        for label, path in write_report(cfg, result).items():
            log.info("wrote %s: %s", label, path)
    return 0


# ---------------------------------------------------------------------------
# bolt explain  (G4)
# ---------------------------------------------------------------------------

def do_explain(cfg: BoltConfig, model: str = "lstm", consistency: bool = True,
               analogue: bool = False) -> int:
    from bolt.explain.attribution import attribute, timestep_profile
    from bolt.explain.consistency import (
        consistency_scores,
        episode_attribution_profiles,
        top_features_by_episode,
        write_figures,
    )

    X, y, meta = load_windows(cfg)
    models, scaler = load_models(cfg, [model])
    fitted = models[model]
    X_scaled = scaler.transform(X)
    features = list(cfg.feature_columns)
    tables = cfg.path("tables")
    tables.mkdir(parents=True, exist_ok=True)

    overall = attribute(fitted, X_scaled, features, max_samples=1000)
    ranked = overall.ranked()
    ranked.to_csv(tables / f"attribution_{model}.csv", index=False)
    print(f"\n=== TOP DRIVERS OVERALL ({model}, {overall.method}) ===")
    print(ranked.head(10).to_string(index=False))

    profile = timestep_profile(overall)
    profile.to_csv(tables / f"attribution_timestep_{model}.csv")
    print(f"\n=== ATTRIBUTION BY WINDOW POSITION ===")
    print(f"earliest day t-{cfg.lookback_days - 1}: {profile.iloc[0]:.4g}  "
          f"final day t-0: {profile.iloc[-1]:.4g}")

    if not consistency:
        return 0

    profiles = episode_attribution_profiles(
        fitted, X_scaled, meta, list(cfg.episodes), features
    )
    scores = consistency_scores(profiles, int(cfg.section("explain")["top_k"]))

    profiles.to_csv(tables / "attribution_profiles_by_episode.csv")
    top_features_by_episode(profiles, scores["top_k"]).to_csv(
        tables / "top_features_per_episode.csv", index=False
    )
    summary = pd.DataFrame([{
        "model": model,
        "method": overall.method,
        "stability_index": scores["stability_index"],
        "n_episodes": scores["n_episodes"],
        "n_pairs": scores["n_pairs"],
        "top_k": scores["top_k"],
        "interpretation": scores["interpretation"],
    }])
    summary.to_csv(tables / "attribution_consistency.csv", index=False)
    scores["spearman_matrix"].to_csv(tables / "attribution_spearman_matrix.csv")
    scores["top_k_jaccard"].to_csv(tables / "attribution_jaccard_matrix.csv")

    for label, path in write_figures(profiles, scores, cfg.path("figures")).items():
        log.info("wrote %s: %s", label, path)

    print("\n=== G4: ATTRIBUTION CONSISTENCY ACROSS CRISES ===")
    print(scores["spearman_matrix"].round(3).to_string())
    print(f"\nstability index : {scores['stability_index']:.3f}")
    print(f"interpretation  : {scores['interpretation']}")

    if analogue:
        _run_analogue(cfg, X_scaled, meta, features)
    return 0


def _run_analogue(cfg: BoltConfig, X_scaled, meta, features) -> None:
    from bolt.explain.analogue import nearest_historical_analogue, summarise_analogues
    from bolt.windows import flatten_windows

    settings = cfg.section("explain")["analogue"]
    panel = load_panel(cfg)
    primary = meta["asset"] == cfg.primary_asset
    if not primary.any():
        log.warning("no windows for %s; skipping analogue retrieval", cfg.primary_asset)
        return

    subset = flatten_windows(X_scaled[primary.to_numpy()])
    sub_meta = meta[primary].reset_index(drop=True)
    results = nearest_historical_analogue(
        subset[-1], subset, sub_meta, prices=panel["close"],
        k=int(settings["k"]), exclude_days=int(settings["exclude_days"]),
        metric=str(settings["metric"]),
        outcome_window_days=int(settings["outcome_window_days"]),
    )
    print(f"\n=== NEAREST HISTORICAL ANALOGUES ({cfg.primary_asset}) ===")
    print(summarise_analogues(results))
    pd.DataFrame(results).to_csv(cfg.path("tables") / "analogues.csv", index=False)


# ---------------------------------------------------------------------------
# bolt predict  (the agent chain + G5)
# ---------------------------------------------------------------------------

def do_predict(cfg: BoltConfig, asset: str, as_of: str, model: str = "lstm",
               commit: bool = False) -> int:
    from bolt.agents.pipeline import ChainGuardPipeline, build_context

    panel = load_panel(cfg, targets_only=False)
    features = list(cfg.feature_columns)

    quant_models = list(cfg.section("agents")["quant_models"])
    try:
        models, scaler = load_models(cfg, quant_models)
        log.info("quantitative agent using: %s", sorted(models))
    except FileNotFoundError as exc:
        log.warning("%s -- the quantitative agent will report UNAVAILABLE", exc)
        models, scaler = {}, None

    # Supplying the analogue source is what activates the Skeptic's precedent
    # check. Without it that challenge never fires and the explanation never
    # cites a precedent - both were silently dead here until it was spotted.
    try:
        X, _, meta = load_windows(cfg)
        analogue_source = (scaler.transform(X) if scaler is not None else X,
                           meta, panel["close"])
    except Exception as exc:  # noqa: BLE001
        log.warning("analogues unavailable (%s); the precedent check will not fire", exc)
        analogue_source = None

    context = build_context(cfg, panel, asset, pd.Timestamp(as_of), features)
    pipeline = ChainGuardPipeline(cfg, models, scaler, analogue_source=analogue_source)
    result = pipeline.run(context, commit=commit)

    print()
    print(result.render())

    if commit and result.commitment is not None and not result.commitment.committed:
        log.error("commitment did NOT land: %s", result.commitment.reason)
        return 4
    return 0


def do_verify(payload: str, prediction_id: str, rpc_url: str | None,
              contract: str | None, offline: bool = False) -> int:
    from bolt.chain.verify import verify

    result = verify(payload, prediction_id, rpc_url, contract, offline=offline)
    print()
    print(result.render())
    return 0 if result.match else 1


# ---------------------------------------------------------------------------
# bolt monitor  (automation layer)
# ---------------------------------------------------------------------------

def do_monitor(cfg: BoltConfig, as_of: str | None = None, commit: bool = False,
               cycles: int = 1, every_days: int = 0) -> int:
    """Run one or more automated monitoring cycles.

    Each cycle resolves matured predictions, predicts for every target asset,
    and runs the health check. With ``cycles > 1`` and ``every_days > 0`` it
    walks backwards through history, which is how a track record is built from
    a frozen dataset without waiting for real time to pass.
    """
    from bolt.automation import run_cycle
    from bolt.store import PredictionStore

    store = PredictionStore(cfg.path("predictions") / "ledger")
    panel = load_panel(cfg, targets_only=False)
    latest = panel.index.get_level_values("date").max()
    start = pd.Timestamp(as_of) if as_of else latest
    if start.tzinfo is None:
        start = start.tz_localize("UTC")

    if cycles > 1 and every_days <= 0:
        raise SystemExit("bolt monitor: --cycles > 1 requires --every-days")

    # Walk forward from the oldest cycle so resolutions land in causal order.
    dates = [start - pd.Timedelta(every_days * i, "D") for i in range(cycles)][::-1]

    for number, date in enumerate(dates, start=1):
        log.info("cycle %d/%d  as-of %s", number, len(dates), date.date())
        result = run_cycle(cfg, as_of=date, commit=commit, store=store)
        summary = result.summary()
        print(
            f"{summary['as_of']}  predictions={summary['predictions']:2d}  "
            f"alerts={summary['alerts']:2d}  resolved={summary['resolved']:2d}  "
            f"committed={summary['committed']:2d}"
            + (f"  ALERTS: {', '.join(summary['alert_assets'])}" if summary["alerts"] else "")
        )

    boundary = (
        pd.Timestamp(cfg.folds[-1].test_start, tz="UTC")
        - pd.Timedelta(cfg.embargo_days, "D")
    ).date().isoformat()
    record = store.track_record(training_boundary=boundary)
    print()
    print("=== TRACK RECORD ===")
    print(f"  predictions issued : {record['predictions']}")
    print(f"  committed on-chain : {record['committed']}")
    print(f"  resolved           : {record['resolved']}")
    print(f"  pending horizon    : {record['pending']}")
    for key, value in sorted(record["counts"].items()):
        print(f"    {key:<16} {value}")
    for label, key in (("IN-SAMPLE", "in_sample"), ("OUT OF SAMPLE", "out_of_sample")):
        block = record.get(key)
        if not block or not block["resolved"]:
            continue
        print()
        print(f"  {label} (training boundary {boundary})")
        print(f"    resolved   : {block['resolved']}")
        print(f"    alerts     : {block['alerts']}")
        for name, value in sorted(block["counts"].items()):
            print(f"      {name:<16} {value}")
        if block["precision"] is not None:
            print(f"    precision  : {block['precision']:.3f}")
        if block["recall"] is not None:
            print(f"    recall     : {block['recall']:.3f}")

    print()
    print("  IN-SAMPLE figures are NOT evidence of foresight: the deployment models")
    print("  were fitted on that period. Only the out-of-sample block is a real test.")
    return 0


def do_serve(cfg: BoltConfig, host: str = "127.0.0.1", port: int = 8000,
             reload: bool = False) -> int:
    """Serve the ChainGuard terminal."""
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(
            "the web console needs fastapi and uvicorn: "
            "pip install 'fastapi>=0.115' 'uvicorn[standard]>=0.32'"
        ) from exc

    from bolt.web.app import create_app

    log.info("ChainGuard terminal on http://%s:%d", host, port)
    uvicorn.run(create_app(cfg), host=host, port=port, log_level="info")
    return 0
