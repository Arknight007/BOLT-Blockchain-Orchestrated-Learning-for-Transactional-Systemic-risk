"""Configuration loading and validation.

Implements BOLT_SPEC.md Section 2: `config/default.yaml` is the single source of
truth for every parameter. This module parses it into frozen dataclasses so that
nothing downstream can mutate a parameter mid-run, and validates the invariants
that the leakage guards of Section 4 depend on -- most importantly that
``split.embargo_days >= windows.lookback_days + labeling.horizon_days`` (Guard 4).

A configuration that violates an invariant raises at load time. It must never be
possible to start a run that is guaranteed to leak.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

__all__ = [
    "ConfigError",
    "Asset",
    "Episode",
    "Fold",
    "BoltConfig",
    "load_config",
]

DEFAULT_CONFIG_PATH = Path("config/default.yaml")


class ConfigError(ValueError):
    """Raised when the configuration is missing, malformed, or self-inconsistent."""


def _parse_date(value: Any, where: str) -> date:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:  # pragma: no cover - message is the point
        raise ConfigError(f"{where}: {value!r} is not an ISO date (YYYY-MM-DD)") from exc


def _require(mapping: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"{where}: missing required key {key!r}")
    return mapping[key]


@dataclass(frozen=True, slots=True)
class Asset:
    """One member of the frozen universe (``config/assets.yaml``)."""

    symbol: str
    name: str
    coingecko_id: str
    binance_symbol: str | None
    role: str          # "target" (labelled and predicted) | "context" (features only)
    inception: date

    @property
    def is_target(self) -> bool:
        return self.role == "target"


@dataclass(frozen=True, slots=True)
class Episode:
    """A named crisis window used for episode-level analysis (G4)."""

    name: str
    start: date
    end: date

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ConfigError(f"crisis_episodes: {self.name!r} ends before it starts")


@dataclass(frozen=True, slots=True)
class Fold:
    """One walk-forward fold: train on everything up to ``train_end``, test on the range."""

    train_end: date
    test_start: date
    test_end: date

    def __post_init__(self) -> None:
        if self.test_start <= self.train_end:
            raise ConfigError(
                f"split.folds: test range starts {self.test_start} which is not after "
                f"train_end {self.train_end}"
            )
        if self.test_end < self.test_start:
            raise ConfigError(f"split.folds: test range {self.test_start}..{self.test_end} is empty")


@dataclass(frozen=True, slots=True)
class BoltConfig:
    """Validated, immutable view over ``config/default.yaml`` plus the asset universe."""

    raw: Mapping[str, Any]
    assets: tuple[Asset, ...]
    episodes: tuple[Episode, ...]
    folds: tuple[Fold, ...]
    config_path: Path
    assets_path: Path
    repo_root: Path = field(default_factory=Path.cwd)

    # -- convenience accessors used all over the pipeline ------------------
    @property
    def seed(self) -> int:
        return int(self.raw["project"]["seed"])

    @property
    def start_date(self) -> date:
        return _parse_date(self.raw["data"]["start_date"], "data.start_date")

    @property
    def end_date(self) -> date:
        return _parse_date(self.raw["data"]["end_date"], "data.end_date")

    @property
    def lookback_days(self) -> int:
        return int(self.raw["windows"]["lookback_days"])

    @property
    def horizon_days(self) -> int:
        return int(self.raw["labeling"]["horizon_days"])

    @property
    def embargo_days(self) -> int:
        return int(self.raw["split"]["embargo_days"])

    @property
    def drawdown_threshold(self) -> float:
        return float(self.raw["labeling"]["drawdown_threshold"])

    @property
    def primary_asset(self) -> str:
        return str(self.raw["assets"]["primary"])

    @property
    def target_assets(self) -> tuple[Asset, ...]:
        return tuple(a for a in self.assets if a.is_target)

    @property
    def feature_columns(self) -> tuple[str, ...]:
        families = self.raw["features"]
        names: list[str] = []
        for family in ("technical", "onchain", "sentiment", "contagion"):
            names.extend(families[family])
        return tuple(names)

    def path(self, key: str) -> Path:
        """Resolve a ``paths.*`` entry against the repository root."""
        paths = self.raw["paths"]
        if key not in paths:
            raise ConfigError(f"paths: no entry named {key!r}")
        return self.repo_root / str(paths[key])

    def section(self, name: str) -> Mapping[str, Any]:
        if name not in self.raw:
            raise ConfigError(f"config: no section named {name!r}")
        return self.raw[name]

    def model_params(self, model: str) -> Mapping[str, Any]:
        models = self.raw["models"]
        if model not in models:
            raise ConfigError(
                f"models: no entry for {model!r}; known models are {sorted(models)}"
            )
        return models[model]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

_REQUIRED_SECTIONS = (
    "project", "paths", "data", "assets", "ingest", "labeling", "windows",
    "features", "split", "imbalance", "models", "evaluation",
    "crisis_episodes", "explain", "chain", "logging",
)

_VALID_ROLES = frozenset({"target", "context"})


def _load_yaml(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise ConfigError(f"configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path}: expected a YAML mapping at the top level")
    return loaded


def _parse_assets(path: Path) -> tuple[Asset, ...]:
    doc = _load_yaml(path)
    entries = _require(doc, "universe", str(path))
    if not isinstance(entries, list) or not entries:
        raise ConfigError(f"{path}: 'universe' must be a non-empty list")

    assets: list[Asset] = []
    for i, entry in enumerate(entries):
        where = f"{path}: universe[{i}]"
        role = str(_require(entry, "role", where))
        if role not in _VALID_ROLES:
            raise ConfigError(f"{where}: role {role!r} must be one of {sorted(_VALID_ROLES)}")
        binance = entry.get("binance_symbol")
        assets.append(
            Asset(
                symbol=str(_require(entry, "symbol", where)),
                name=str(_require(entry, "name", where)),
                coingecko_id=str(_require(entry, "coingecko_id", where)),
                binance_symbol=None if binance in (None, "", "null") else str(binance),
                role=role,
                inception=_parse_date(_require(entry, "inception", where), where),
            )
        )

    symbols = [a.symbol for a in assets]
    duplicates = {s for s in symbols if symbols.count(s) > 1}
    if duplicates:
        raise ConfigError(f"{path}: duplicate symbols {sorted(duplicates)}")
    if not any(a.is_target for a in assets):
        raise ConfigError(f"{path}: the universe contains no assets with role 'target'")
    return tuple(assets)


def _parse_folds(raw: Mapping[str, Any]) -> tuple[Fold, ...]:
    entries = _require(raw["split"], "folds", "split")
    folds: list[Fold] = []
    for i, entry in enumerate(entries):
        where = f"split.folds[{i}]"
        test = _require(entry, "test", where)
        if not isinstance(test, (list, tuple)) or len(test) != 2:
            raise ConfigError(f"{where}: 'test' must be a [start, end] pair")
        folds.append(
            Fold(
                train_end=_parse_date(_require(entry, "train_end", where), where),
                test_start=_parse_date(test[0], where),
                test_end=_parse_date(test[1], where),
            )
        )
    if not folds:
        raise ConfigError("split.folds: at least one fold is required")
    return tuple(folds)


def _parse_episodes(raw: Mapping[str, Any]) -> tuple[Episode, ...]:
    entries = raw["crisis_episodes"]
    if not entries:
        raise ConfigError("crisis_episodes: at least one episode is required for G4")
    episodes = []
    for i, entry in enumerate(entries):
        where = f"crisis_episodes[{i}]"
        episodes.append(
            Episode(
                name=str(_require(entry, "name", where)),
                start=_parse_date(_require(entry, "start", where), where),
                end=_parse_date(_require(entry, "end", where), where),
            )
        )
    return tuple(episodes)


def _validate(cfg: BoltConfig) -> None:
    """Cross-section invariants. Each failure here would otherwise become a silent bug."""
    raw = cfg.raw

    # --- Guard 4 (BOLT_SPEC.md Section 4): the embargo must cover a full window
    # plus its forward label horizon, or a training sample can overlap the test range.
    minimum = cfg.lookback_days + cfg.horizon_days
    if cfg.embargo_days < minimum:
        raise ConfigError(
            f"split.embargo_days={cfg.embargo_days} violates Guard 4: it must be >= "
            f"windows.lookback_days ({cfg.lookback_days}) + labeling.horizon_days "
            f"({cfg.horizon_days}) = {minimum}. A shorter embargo lets a training "
            f"window's label period reach into the test range."
        )

    # --- the accuracy ban must not be switched off from config (Rule 12.1)
    if not raw["evaluation"].get("forbid_accuracy", False):
        raise ConfigError(
            "evaluation.forbid_accuracy must remain true: with a ~2% positive rate "
            "accuracy is meaningless and reporting it destroys the result's credibility."
        )
    reported = set(raw["evaluation"]["report"])
    banned = reported & {"accuracy", "accuracy_score"}
    if banned:
        raise ConfigError(f"evaluation.report lists banned metric(s): {sorted(banned)}")

    # --- dates
    if cfg.end_date <= cfg.start_date:
        raise ConfigError(f"data.end_date {cfg.end_date} must be after start_date {cfg.start_date}")

    # --- labelling
    threshold = cfg.drawdown_threshold
    if not 0.0 < threshold < 1.0:
        raise ConfigError(f"labeling.drawdown_threshold must be a fraction in (0,1), got {threshold}")
    if cfg.horizon_days < 1:
        raise ConfigError("labeling.horizon_days must be >= 1")
    if cfg.lookback_days < 2:
        raise ConfigError("windows.lookback_days must be >= 2 for a sequence model")

    # --- folds must be chronological and inside the data range
    previous_end: date | None = None
    for fold in cfg.folds:
        if previous_end is not None and fold.test_start <= previous_end:
            raise ConfigError(
                f"split.folds must be chronological and non-overlapping: fold starting "
                f"{fold.test_start} overlaps the previous fold ending {previous_end}"
            )
        if fold.test_start < cfg.start_date or fold.test_end > cfg.end_date:
            raise ConfigError(
                f"split.folds: test range {fold.test_start}..{fold.test_end} falls outside "
                f"data range {cfg.start_date}..{cfg.end_date}"
            )
        previous_end = fold.test_end

    # --- primary asset must be in the universe and labellable
    symbols = {a.symbol for a in cfg.assets}
    if cfg.primary_asset not in symbols:
        raise ConfigError(
            f"assets.primary={cfg.primary_asset!r} is not in the frozen universe {sorted(symbols)}"
        )
    if not next(a for a in cfg.assets if a.symbol == cfg.primary_asset).is_target:
        raise ConfigError(f"assets.primary={cfg.primary_asset!r} has role 'context' and is never labelled")

    # --- imbalance ablation must include the primary method's alternatives
    method = raw["imbalance"]["method"]
    known = {"class_weight", "smote", "none"}
    unknown = ({method} | set(raw["imbalance"]["compare_with"])) - known
    if unknown:
        raise ConfigError(f"imbalance: unknown method(s) {sorted(unknown)}; expected {sorted(known)}")

    # --- explainability
    if int(raw["explain"]["analogue"]["exclude_days"]) < cfg.lookback_days:
        raise ConfigError(
            "explain.analogue.exclude_days must be >= windows.lookback_days, otherwise a "
            "'historical analogue' can be an overlapping window from the same week."
        )


def load_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    repo_root: str | Path | None = None,
) -> BoltConfig:
    """Load, parse and validate the BOLT configuration.

    Args:
        path: path to the main YAML config.
        repo_root: root against which relative ``paths.*`` entries resolve.
            Defaults to the parent of the config file's directory.

    Returns:
        A frozen :class:`BoltConfig`.

    Raises:
        ConfigError: if the file is missing, malformed, or violates an invariant
            that the leakage guards depend on.
    """
    config_path = Path(path).resolve()
    raw = _load_yaml(config_path)

    missing = [s for s in _REQUIRED_SECTIONS if s not in raw]
    if missing:
        raise ConfigError(f"{config_path}: missing required section(s) {missing}")

    root = Path(repo_root).resolve() if repo_root is not None else config_path.parent.parent
    assets_path = (root / str(raw["assets"]["universe_file"])).resolve()

    cfg = BoltConfig(
        raw=raw,
        assets=_parse_assets(assets_path),
        episodes=_parse_episodes(raw),
        folds=_parse_folds(raw),
        config_path=config_path,
        assets_path=assets_path,
        repo_root=root,
    )
    _validate(cfg)
    return cfg
