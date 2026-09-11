"""DATA_CARD.md generation (BOLT_SPEC.md Section 3).

The panel will ask "where did your data come from". This file is the answer, and
it is generated from the pipeline rather than written by hand, so it cannot drift
away from what the code actually did.

Contents: source per column, date coverage, missing-data percentage per column
per year, every proxy substitution with its stated limitation, every excluded
asset and removed feature, and the SHA-256 of the frozen dataset file.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from bolt.align.panel import CoverageReport
from bolt.config import BoltConfig
from bolt.ingest.provenance import REGISTRY, Provenance
from bolt.logging_setup import get_logger
from bolt.version import git_commit_sha

log = get_logger(__name__)


def _fmt_pct(value: float) -> str:
    return "-" if pd.isna(value) else f"{value:.1%}"


def _coverage_table(report: CoverageReport) -> str:
    years = list(report.by_column.columns)
    header = "| column | overall | " + " | ".join(str(y) for y in years) + " |"
    divider = "|---|---|" + "---|" * len(years)
    lines = [header, divider]
    for column in report.by_column.index:
        overall = 1.0 - report.overall.get(column, float("nan"))
        cells = [_fmt_pct(1.0 - report.by_column.loc[column, y]) for y in years]
        lines.append(f"| `{column}` | {_fmt_pct(overall)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _source_table(columns: list[str]) -> str:
    lines = ["| column | source | kind | definition |", "|---|---|---|---|"]
    for column in columns:
        entry = REGISTRY.get(column)
        if entry is None:
            lines.append(f"| `{column}` | derived | DERIVED | computed from primary columns |")
            continue
        lines.append(
            f"| `{column}` | {entry.source} | {entry.provenance.value} | {entry.definition} |"
        )
    return "\n".join(lines)


def _proxy_section() -> str:
    proxies = [s for s in REGISTRY.values() if s.provenance is Provenance.PROXY]
    if not proxies:
        return "No proxy substitutions: every column is a primary or derived observation.\n"
    blocks = []
    for entry in proxies:
        blocks.append(
            f"### `{entry.column}`\n\n"
            f"- **Stands in for:** {entry.proxy_for}\n"
            f"- **Actually measures:** {entry.definition}\n"
            f"- **Source:** {entry.source}\n"
            f"- **Limitation:** {entry.limitation}\n"
        )
    return "\n".join(blocks)


def _asset_table(cfg: BoltConfig, panel: pd.DataFrame, sources: dict[str, str]) -> str:
    counts = panel.groupby(level="asset").size()
    dates = panel.index.get_level_values("date")
    lines = ["| asset | role | rows | first | last | source |", "|---|---|---|---|---|---|"]
    for asset in cfg.assets:
        symbol = asset.symbol
        if symbol not in counts.index:
            lines.append(f"| `{symbol}` | {asset.role} | **0 (EXCLUDED)** | - | - | none available |")
            continue
        block = panel.xs(symbol, level="asset")
        lines.append(
            f"| `{symbol}` | {asset.role} | {counts[symbol]} | {block.index.min().date()} "
            f"| {block.index.max().date()} | {sources.get(symbol, '?')} |"
        )
    return "\n".join(lines)


def write_data_card(
    cfg: BoltConfig,
    panel: pd.DataFrame,
    report: CoverageReport,
    sha256: str,
    panel_path: Path,
    sources: dict[str, str],
    excluded: list[str],
    notes: list[str],
) -> Path:
    """Write ``data/processed/DATA_CARD.md`` and return its path."""
    out = cfg.path("processed") / "DATA_CARD.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    features = list(cfg.feature_columns)

    labelled = panel["label"].dropna()
    positive_rate = float(labelled.mean()) if len(labelled) else float("nan")

    body = f"""# DATA CARD

Generated automatically by `bolt build` on {generated}.
Do not edit by hand: this file is regenerated on every build and any manual
change will be overwritten.

- **Code version:** `{git_commit_sha(cfg.repo_root)}`
- **Config:** `{cfg.config_path.name}`, universe `{cfg.assets_path.name}`
- **Frozen dataset:** `{panel_path.relative_to(cfg.repo_root).as_posix()}`
- **SHA-256:** `{sha256}`
- **Shape:** {len(panel):,} rows x {panel.shape[1]} columns
- **Coverage:** {report.date_min} to {report.date_max}, {report.assets} assets
- **Label:** {cfg.drawdown_threshold:.0%} drawdown within {cfg.horizon_days} days,
  {int(labelled.sum()):,} positive of {len(labelled):,} labelled ({positive_rate:.2%})

## Assets

{_asset_table(cfg, panel, sources)}

Assets are absent before their inception date rather than back-filled: the panel
is deliberately ragged, because inventing history for an asset that did not yet
trade is exactly what Guard 1 forbids.

## Column sources

{_source_table(features)}

## Proxy substitutions

No Etherscan API key is configured for this build, so the four ledger-native
metrics are documented proxies. Each states what it stands in for and where it
falls short. Nothing is silently substituted.

{_proxy_section()}
## Missing data by year

Percentages are **coverage** (1 - missing), computed over every row in the panel
including context assets.

{_coverage_table(report)}

## Known gaps and decisions

{chr(10).join("- " + note for note in notes) if notes else "- None."}

## Reproduction

```bash
pip install -r requirements.txt && pip install -e .
bolt ingest --config config/default.yaml    # cached; a second run makes zero API calls
bolt build  --config config/default.yaml    # regenerates this file and the parquet above
```

Verify you hold the same dataset that produced the reported results:

```bash
python -c "import hashlib,pathlib; print(hashlib.sha256(pathlib.Path('{panel_path.relative_to(cfg.repo_root).as_posix()}').read_bytes()).hexdigest())"
# expected: {sha256}
```
"""
    out.write_text(body, encoding="utf-8")
    log.info("data card: %s", out)
    return out
