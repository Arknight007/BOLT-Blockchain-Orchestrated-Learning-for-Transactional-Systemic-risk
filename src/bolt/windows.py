"""Sliding-window tensor construction (BOLT_SPEC.md Section 5).

The returned metadata frame is required, not a convenience: ``evaluate/splits.py``
builds the Guard 4 embargo from its ``[window_start, label_end]`` intervals. A
window tensor without that metadata cannot be split without leaking, because
nothing records how far into the future each sample's label reaches.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bolt.logging_setup import get_logger

log = get_logger(__name__)

META_COLUMNS = ["asset", "window_start", "window_end", "label_end"]


def build_windows(
    panel: pd.DataFrame,
    lookback: int,
    feature_cols: list[str],
    label_col: str,
    horizon: int,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build ``(N, lookback, F)`` tensors from a (date, asset) panel.

    A window is emitted only when the full lookback is present with no missing
    feature values and the label is known. Partial windows are dropped rather
    than imputed: an imputed window is a fabricated observation.

    Args:
        panel: (date, asset) panel containing ``feature_cols`` and ``label_col``.
        lookback: window length in days.
        feature_cols: ordered feature names; the F axis follows this order.
        label_col: name of the binary label column.
        horizon: label horizon in days, recorded in ``label_end``.
        stride: step between consecutive window ends.

    Returns:
        ``(X, y, meta)`` where X is ``(N, lookback, F)`` float64, y is ``(N,)``
        int8, and meta has :data:`META_COLUMNS`.
    """
    missing = [c for c in feature_cols + [label_col] if c not in panel.columns]
    if missing:
        raise ValueError(f"build_windows: panel is missing {missing}")

    tensors: list[np.ndarray] = []
    labels: list[int] = []
    records: list[dict] = []
    skipped_incomplete = 0

    for asset, group in panel.groupby(level="asset", sort=True):
        ordered = group.sort_index(level="date")
        dates = ordered.index.get_level_values("date")
        features = ordered[feature_cols].to_numpy(dtype=np.float64)
        targets = ordered[label_col].to_numpy(dtype=np.float64)

        for end in range(lookback - 1, len(ordered), stride):
            start = end - lookback + 1
            block = features[start:end + 1]
            label = targets[end]
            if not np.isfinite(block).all() or not np.isfinite(label):
                skipped_incomplete += 1
                continue
            tensors.append(block)
            labels.append(int(label))
            records.append({
                "asset": asset,
                "window_start": dates[start],
                "window_end": dates[end],
                # The label at window_end looks `horizon` days forward. This is
                # the value the embargo is computed from.
                "label_end": dates[end] + pd.Timedelta(horizon, "D"),
            })

    if not tensors:
        raise ValueError(
            "build_windows produced no complete windows. Either the lookback exceeds "
            "the available history or a feature column is entirely missing."
        )

    X = np.stack(tensors)
    y = np.asarray(labels, dtype=np.int8)
    meta = pd.DataFrame.from_records(records, columns=META_COLUMNS)

    log.info(
        "windows: %d samples of shape (%d, %d), %d positive (%.2f%%), %d incomplete skipped",
        len(X), lookback, len(feature_cols), int(y.sum()), 100.0 * y.mean(), skipped_incomplete,
    )
    return X, y, meta


def flatten_windows(X: np.ndarray) -> np.ndarray:
    """Flatten ``(N, T, F)`` to ``(N, T*F)`` for tabular models.

    Column order is timestep-major: ``[t0f0, t0f1, ..., t1f0, ...]``. Every model
    in the bench sees the SAME inputs through this one function; if the tabular
    models received different features the comparison would not be fair, and a
    panel would be right to say so.
    """
    if X.ndim != 3:
        raise ValueError(f"expected (N, T, F), got shape {X.shape}")
    return X.reshape(X.shape[0], -1)


def flattened_feature_names(feature_cols: list[str], lookback: int) -> list[str]:
    """Names matching :func:`flatten_windows` column order, for SHAP output."""
    return [f"{name}[t-{lookback - 1 - t}]" for t in range(lookback) for name in feature_cols]
