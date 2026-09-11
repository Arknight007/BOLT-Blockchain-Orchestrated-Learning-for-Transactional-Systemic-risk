"""Price and volume features (BOLT_SPEC.md Section 5).

Every rolling window here is TRAILING (``.rolling(w)``); ``center=True`` is
forbidden by Guard 1 and would make each feature peek at its own future.

The base paper deliberately excluded conventional price-based variables, arguing
they had already received extensive scrutiny. BOLT adds them back so that the
on-chain and contagion families are tested *alongside* them rather than in their
absence - if volatility alone carries the signal, the comparison should say so.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bolt.logging_setup import get_logger

log = get_logger(__name__)

TECHNICAL_COLUMNS = [
    "realized_vol_7", "realized_vol_14", "realized_vol_30", "volume_z_20",
    "drawdown_depth", "drawdown_duration", "rsi_14", "momentum_10", "atr_14",
]


def realized_volatility(close: pd.Series, window: int) -> pd.Series:
    """Annualised standard deviation of trailing log returns."""
    returns = np.log(close / close.shift(1))
    return returns.rolling(window, min_periods=window).std() * np.sqrt(365.0)


def volume_zscore(volume: pd.Series, window: int) -> pd.Series:
    """How anomalous today's volume is against its own trailing distribution."""
    mean = volume.rolling(window, min_periods=window).mean()
    std = volume.rolling(window, min_periods=window).std()
    return (volume - mean) / std.replace(0.0, np.nan)


def drawdown_depth(close: pd.Series) -> pd.Series:
    """Current decline from the running peak, as a positive fraction.

    The running maximum is expanding and therefore trailing: ``cummax`` at t
    uses only closes up to t.
    """
    peak = close.cummax()
    return (peak - close) / peak


def drawdown_duration(close: pd.Series) -> pd.Series:
    """Days since the running peak was last set."""
    peak = close.cummax()
    at_peak = close >= peak
    # Days since the most recent True, computed forward in time only.
    groups = at_peak.cumsum()
    counter = at_peak.groupby(groups).cumcount()
    return counter.astype(float)


def rsi(close: pd.Series, period: int) -> pd.Series:
    """Wilder's Relative Strength Index over a trailing window."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def momentum(close: pd.Series, period: int) -> pd.Series:
    """Fractional price change over the trailing ``period`` days."""
    return close / close.shift(period) - 1.0


def average_true_range(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int
) -> pd.Series:
    """ATR, normalised by close so it is comparable across assets and eras."""
    previous_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()], axis=1
    ).max(axis=1)
    return true_range.rolling(period, min_periods=period).mean() / close


def compute_technical(frame: pd.DataFrame, params: dict) -> pd.DataFrame:
    """All nine technical features for one asset's OHLCV frame.

    Args:
        frame: single-asset OHLCV indexed by date.
        params: ``features.params`` from the configuration.

    Returns:
        A frame of :data:`TECHNICAL_COLUMNS` on the same index.
    """
    close, high, low = frame["close"], frame["high"], frame["low"]
    volume = frame["volume"]

    out = pd.DataFrame(index=frame.index)
    for window in params["realized_vol_windows"]:
        out[f"realized_vol_{window}"] = realized_volatility(close, window)
    out["volume_z_20"] = volume_zscore(volume, params["volume_z_window"])
    out["drawdown_depth"] = drawdown_depth(close)
    out["drawdown_duration"] = drawdown_duration(close)
    out["rsi_14"] = rsi(close, params["rsi_period"])
    out["momentum_10"] = momentum(close, params["momentum_period"])
    out["atr_14"] = average_true_range(high, low, close, params["atr_period"])

    missing = [c for c in TECHNICAL_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"compute_technical did not produce {missing}")
    return out[TECHNICAL_COLUMNS]
