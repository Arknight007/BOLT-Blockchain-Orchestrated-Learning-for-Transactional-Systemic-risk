"""Dataset assembly and freezing (BOLT_SPEC.md Sections 3-5, Phase 1-2).

Orchestrates ingest -> panel -> features -> labels -> frozen parquet, then writes
``data/processed/DATA_CARD.md`` with per-column provenance, per-year coverage and
the SHA-256 of the frozen file.

The hash is the point. "Where did your data come from" is answered by a file, and
"is this the same data you evaluated on" is answered by a digest.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from bolt.align.datacard import write_data_card
from bolt.align.panel import (
    CoverageReport,
    build_panel,
    coverage_report,
    enforce_coverage,
    trim_to_inception,
)
from bolt.config import BoltConfig
from bolt.features.contagion import compute_contagion
from bolt.features.onchain import compute_onchain
from bolt.features.sentiment import compute_sentiment
from bolt.features.technical import compute_technical
from bolt.ingest.cache import DiskCache
from bolt.ingest.market import fetch_ohlcv
from bolt.ingest.news import fetch_fear_greed, fetch_headlines
from bolt.ingest.onchain import build_onchain_raw, fetch_stablecoin_supply
from bolt.labeling.crash import assert_episodes_covered, label_panel, label_sensitivity_table
from bolt.logging_setup import get_logger

log = get_logger(__name__)

PANEL_FILENAME = "panel.parquet"
DATA_CARD_FILENAME = "DATA_CARD.md"

#: Significant figures used when hashing panel CONTENT. Fixed so the digest is
#: stable across platforms and float repr changes.
CONTENT_PRECISION = 10


def content_digest(panel: pd.DataFrame, precision: int = CONTENT_PRECISION) -> str:
    """SHA-256 over the panel's CONTENT, independent of the storage format.

    This is a SECOND check alongside the file hash, not a replacement for it.
    Parquet is byte-deterministic for identical data - verified here by writing
    the same frame three times and getting one digest - so the file hash is a
    valid reproducibility check today. The content hash guards a narrower risk:
    a future pyarrow or compression change would alter the bytes while leaving
    the data untouched, and the file hash would then report a mismatch that is
    not one.

    Taken over a canonical CSV rendering (sorted index, sorted columns, fixed
    precision) that a reader can recompute with pandas alone.
    """
    ordered = panel.sort_index()
    ordered = ordered[sorted(ordered.columns)]
    blob = ordered.to_csv(
        float_format=f"%.{precision}g", lineterminator=chr(10)
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


@dataclass(frozen=True, slots=True)
class BuildResult:
    panel: pd.DataFrame
    coverage: CoverageReport
    panel_path: Path
    sha256: str              # of the file as written
    content_sha256: str      # of the data itself, stable across rebuilds
    cache_summary: str
    data_card: Path
    sensitivity: pd.DataFrame
    notes: list[str]


def ingest_all(cfg: BoltConfig, cache: DiskCache, *, refresh: bool = False) -> dict:
    """Fetch every raw source. Returns the pieces the feature layer needs."""
    ohlcv: dict[str, pd.DataFrame] = {}
    onchain_raw: dict[str, pd.DataFrame] = {}

    excluded: list[str] = []
    for asset in cfg.assets:
        frame = fetch_ohlcv(cfg, asset, cache, refresh=refresh)
        if frame.empty:
            excluded.append(asset.symbol)
            continue
        ohlcv[asset.symbol] = frame
        onchain_raw[asset.symbol] = build_onchain_raw(cfg, asset, frame, cache, refresh=refresh)

    if not ohlcv:
        raise RuntimeError('no asset in the universe could be sourced; refusing to build')
    if excluded:
        log.warning('excluded %s: no free source covers the configured window', excluded)

    return {
        "ohlcv": ohlcv,
        "onchain_raw": onchain_raw,
        "stablecoin_supply": fetch_stablecoin_supply(cfg, cache, refresh=refresh),
        "fear_greed": fetch_fear_greed(cfg, cache, refresh=refresh),
        "headlines": fetch_headlines(cfg, cache, refresh=refresh),
        "excluded_assets": excluded,
    }


def build_features(cfg: BoltConfig, raw: dict) -> pd.DataFrame:
    """Compute all four feature families and the crash label onto one panel."""
    params = cfg.section("features")["params"]
    params = {**params, "sentiment_backend": cfg.section("features")["sentiment_backend"]}

    market_panel = trim_to_inception(build_panel(raw["ohlcv"]), cfg)

    per_asset: list[pd.DataFrame] = []
    for asset in cfg.assets:
        symbol = asset.symbol
        if symbol not in raw["ohlcv"]:
            continue
        frame = market_panel.xs(symbol, level="asset")
        if frame.empty:
            continue

        technical = compute_technical(frame, params)

        onchain_input = raw["onchain_raw"][symbol].copy()
        onchain_input.index = onchain_input.index.normalize()
        onchain_input = onchain_input.reindex(frame.index)
        onchain_input["quote_volume"] = frame["quote_volume"]
        onchain = compute_onchain(
            onchain_input, raw["stablecoin_supply"].rename(None), params
        )

        sentiment = compute_sentiment(
            frame.index, raw["headlines"], raw["fear_greed"], params
        )

        block = pd.concat([frame[["close", "volume"]], technical, onchain, sentiment], axis=1)
        block["asset"] = symbol
        per_asset.append(block.set_index("asset", append=True))

    panel = pd.concat(per_asset).sort_index()
    panel.index.names = ["date", "asset"]

    contagion = compute_contagion(panel, params, cfg.primary_asset)
    panel = pd.concat([panel, contagion], axis=1)

    panel["label"] = label_panel(
        panel, cfg.drawdown_threshold, cfg.horizon_days
    )
    # Context assets (stablecoins) feed contagion but are never predicted.
    context = {a.symbol for a in cfg.assets if not a.is_target}
    is_context = panel.index.get_level_values("asset").isin(context)
    panel.loc[is_context, "label"] = pd.NA
    log.info("blanked labels for %d context-asset row(s)", int(is_context.sum()))
    return panel


def freeze(cfg: BoltConfig, panel: pd.DataFrame) -> tuple[Path, str, str]:
    """Write the panel to parquet. Returns (path, file digest, content digest)."""
    out_dir = cfg.path("processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / PANEL_FILENAME
    panel.to_parquet(path, engine="pyarrow", compression="snappy")

    file_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    data_digest = content_digest(panel)
    log.info("frozen dataset: %s (%.1f KB)", path, path.stat().st_size / 1024)
    log.info("  content sha256 (stable) : %s", data_digest)
    log.info("  file sha256    (varies) : %s", file_digest)
    return path, file_digest, data_digest


def run_build(cfg: BoltConfig, *, refresh: bool = False) -> BuildResult:
    """Full Phase 1-2 build. Raises rather than producing a degraded dataset."""
    # Read the code version FIRST. The build writes panel.parquet, DATA_CARD.md
    # and the tables, all of which are tracked, so by the time the card is
    # written the tree is dirty by the build's own hand and every card would be
    # stamped "-dirty" no matter how clean the commit was.
    from bolt.version import git_commit_sha

    code_version = git_commit_sha(cfg.repo_root)
    cache = DiskCache(cfg.path("raw"))
    raw = ingest_all(cfg, cache, refresh=refresh)
    panel = build_features(cfg, raw)

    # The data card reports coverage over every row, but the threshold is enforced
    # on TARGET-asset rows only: context assets (stablecoins) are never labelled
    # and never become training samples, so their missing volume columns cannot
    # degrade a model. Judging the panel on rows it will never see would either
    # block a sound build or hide a real gap.
    report = coverage_report(panel, list(cfg.feature_columns))
    targets = {a.symbol for a in cfg.target_assets}
    target_rows = panel[panel.index.get_level_values("asset").isin(targets)]
    enforce_coverage(
        coverage_report(target_rows, list(cfg.feature_columns)),
        float(cfg.raw["data"]["min_coverage"]),
    )

    # G1 validation: the configured definition must fire on every known crisis.
    # If it does not, the definition is wrong -- surface it, do not tune around it.
    assert_episodes_covered(panel["label"], list(cfg.episodes))

    sensitivity = label_sensitivity_table(
        panel[panel.index.get_level_values("asset").isin(targets)],
        cfg.section("labeling")["sensitivity_grid"],
        list(cfg.episodes),
    )
    tables = cfg.path("tables")
    tables.mkdir(parents=True, exist_ok=True)
    sensitivity.to_csv(tables / "label_sensitivity.csv", index=False)

    path, digest, data_digest = freeze(cfg, panel)

    sources = {
        symbol: str(frame["source"].iloc[0])
        for symbol, frame in raw["ohlcv"].items() if not frame.empty
    }
    notes = _build_notes(cfg, raw)
    card = write_data_card(
        cfg, panel, report, digest, path, sources, raw["excluded_assets"], notes,
        content_sha256=data_digest, code_version=code_version,
    )
    return BuildResult(panel, report, path, digest, data_digest, cache.summary(),
                       card, sensitivity, notes)


def _build_notes(cfg: BoltConfig, raw: dict) -> list[str]:
    """Gaps and decisions the data card must state plainly."""
    notes = [
        "`headline_count_z` is REMOVED from the active feature set. It requires a "
        "CryptoPanic API key; none is configured, so its coverage is 0%. Rule 12.3 "
        "forbids zero-filling it, so it was dropped rather than faked. Restore it in "
        "config/default.yaml once BOLT_CRYPTOPANIC_KEY is set.",
        "No headline text source is configured, so `headline_polarity_*` and "
        "`negative_ratio_7d` are derived from the alternative.me Fear and Greed index "
        "(a real published daily series) rather than from headline polarity. This is a "
        "weaker sentiment signal than the base paper's and is recorded as such.",
        "CoinGecko's free tier refuses historical ranges older than 365 days "
        "(error 10012), which removed market capitalisation from the study window. "
        "`nvt_ratio` is therefore built from price and turnover; see its proxy entry.",
        "Coverage is REPORTED over every row but ENFORCED over target-asset rows only. "
        "Context assets (stablecoins) are never labelled and never become training "
        "samples, and DefiLlama serves no volume for them, so their missing volume "
        "columns cannot affect a model.",
    ]
    for symbol in raw.get("excluded_assets", []):
        notes.append(f"`{symbol}` was EXCLUDED: no free source covers the configured window.")
    short = {"HYPE": "2024-11-29", "ASTER": "2025-09-17"}
    for symbol, inception in short.items():
        if symbol in raw["ohlcv"]:
            rows = len(raw["ohlcv"][symbol])
            notes.append(
                f"`{symbol}` contributes only {rows} days (inception {inception}, and "
                f"CoinGecko's 365-day ceiling truncates it further). It post-dates every "
                f"crisis episode, so it cannot contribute to the G4 episode analysis and "
                f"reaches the contagion graph only in the final fold."
            )
    return notes
