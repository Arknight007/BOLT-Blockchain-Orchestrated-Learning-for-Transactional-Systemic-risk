"""Attribution consistency across crisis episodes - G4 (BOLT_SPEC.md Section 8).

This module answers the project's headline research question:

    Do the same drivers explain crashes across independent episodes, or does each
    crisis have its own signature?

The base paper applies SHAP but never tests whether its explanations are stable.
An explanation that changes completely between the Terra/LUNA collapse and the
FTX collapse is not a generalisable account of crash risk; it is a per-episode
description dressed up as a mechanism.

**The result is reported honestly either way.** High consistency is evidence of a
generalisable crash signature. Low consistency is an important negative result
about the limits of explanation-based risk systems. Both are publishable; only a
fabricated result is not.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from bolt.config import Episode
from bolt.explain.attribution import Attribution, attribute
from bolt.logging_setup import get_logger

log = get_logger(__name__)

MIN_SAMPLES_PER_EPISODE = 5


def episode_mask(meta: pd.DataFrame, episode: Episode, lookahead_days: int = 45) -> np.ndarray:
    """Rows whose window ends in the run-up to, or during, an episode.

    The run-up matters more than the episode itself: a warning system is judged
    on what it said BEFORE the crash, so attributions are taken from the window
    leading into the event, not from the wreckage afterwards.
    """
    dates = pd.to_datetime(meta["window_end"], utc=True)
    start = pd.Timestamp(episode.start, tz="UTC")
    end = pd.Timestamp(episode.end, tz="UTC")
    return ((dates >= start - pd.Timedelta(lookahead_days, "D")) & (dates <= end)).to_numpy()


def episode_attribution_profiles(
    model,
    X: np.ndarray,
    meta: pd.DataFrame,
    episodes: list[Episode],
    feature_names: list[str],
    background: np.ndarray | None = None,
) -> pd.DataFrame:
    """Mean |attribution| per feature, computed separately for each crisis episode.

    Returns:
        A frame indexed by feature with one column per episode that had enough
        samples. Episodes with too few windows are excluded and logged, never
        silently filled.
    """
    profiles: dict[str, pd.Series] = {}
    for episode in episodes:
        mask = episode_mask(meta, episode)
        count = int(mask.sum())
        if count < MIN_SAMPLES_PER_EPISODE:
            log.warning(
                "episode %r has only %d window(s) in range; excluded from the "
                "consistency analysis rather than reported on thin evidence",
                episode.name, count,
            )
            continue
        subset = X[mask]
        attribution = attribute(model, subset, feature_names, background, max_samples=len(subset))
        profiles[episode.name] = attribution.mean_absolute()
        log.info("episode %r: %d windows, top driver %s",
                 episode.name, count, profiles[episode.name].index[0])

    if len(profiles) < 2:
        raise ValueError(
            f"only {len(profiles)} episode(s) had enough windows; consistency between "
            f"episodes needs at least two. Widen the episode windows or check that the "
            f"test folds actually cover the configured crisis dates."
        )
    frame = pd.DataFrame(profiles)
    frame.index.name = "feature"
    return frame.reindex(feature_names).fillna(0.0)


def consistency_scores(profiles: pd.DataFrame, top_k: int = 5) -> dict:
    """Pairwise agreement between episodes' feature rankings.

    Returns:
        - ``spearman_matrix``: pairwise rank correlation of feature rankings.
        - ``top_k_jaccard``: Jaccard overlap of each pair's top-k features.
        - ``stability_index``: mean pairwise Spearman across all episode pairs.
        - ``interpretation``: a plain-language reading of the stability index.
    """
    names = list(profiles.columns)
    spearman = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    jaccard = pd.DataFrame(np.eye(len(names)), index=names, columns=names)

    top_sets = {
        name: set(profiles[name].sort_values(ascending=False).head(top_k).index)
        for name in names
    }

    pairwise: list[float] = []
    for a, b in combinations(names, 2):
        rho = float(spearmanr(profiles[a], profiles[b]).statistic)
        rho = 0.0 if np.isnan(rho) else rho
        spearman.loc[a, b] = spearman.loc[b, a] = rho
        pairwise.append(rho)

        union = top_sets[a] | top_sets[b]
        overlap = len(top_sets[a] & top_sets[b]) / len(union) if union else 0.0
        jaccard.loc[a, b] = jaccard.loc[b, a] = overlap

    stability = float(np.mean(pairwise)) if pairwise else float("nan")
    return {
        "spearman_matrix": spearman,
        "top_k_jaccard": jaccard,
        "stability_index": stability,
        "n_episodes": len(names),
        "n_pairs": len(pairwise),
        "top_k": top_k,
        "interpretation": interpret_stability(stability),
    }


def interpret_stability(stability: float) -> str:
    """Plain-language reading of the stability index, fixed in advance.

    The thresholds are stated here, before any result is computed, so the
    conclusion cannot be retrofitted to whatever number came out.
    """
    if np.isnan(stability):
        return "undetermined: too few episode pairs to compare"
    if stability >= 0.7:
        return (
            "HIGH consistency: the same drivers dominate across independent crises, "
            "which is evidence of a generalisable crash signature rather than "
            "per-episode curve fitting"
        )
    if stability >= 0.4:
        return (
            "MODERATE consistency: a shared core of drivers recurs, but each episode "
            "carries a substantial signature of its own"
        )
    if stability >= 0.0:
        return (
            "LOW consistency: each crisis is explained by largely different drivers. "
            "This is a negative result and an important one - it bounds how far any "
            "explanation-based crash warning generalises out of sample"
        )
    return (
        "NEGATIVE consistency: drivers that matter in one episode are actively "
        "unimportant in another, which argues against a single shared mechanism"
    )


def top_features_by_episode(profiles: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    """Long-form table of each episode's top-k drivers, for the bar chart."""
    rows = []
    for episode in profiles.columns:
        ranked = profiles[episode].sort_values(ascending=False).head(top_k)
        total = float(profiles[episode].sum()) or 1.0
        for rank, (feature, value) in enumerate(ranked.items(), start=1):
            rows.append({
                "episode": episode, "rank": rank, "feature": feature,
                "mean_abs_attribution": float(value), "share_of_total": float(value) / total,
            })
    return pd.DataFrame(rows)


def write_figures(profiles: pd.DataFrame, scores: dict, figures_dir) -> dict:
    """Spearman heatmap and grouped top-feature bar chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    written = {}

    # --- heatmap ---------------------------------------------------------
    matrix = scores["spearman_matrix"]
    fig, ax = plt.subplots(figsize=(1.6 * len(matrix) + 3.0, 1.4 * len(matrix) + 2.4))
    image = ax.imshow(matrix.to_numpy(), cmap="RdYlGn", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(matrix)))
    ax.set_xticklabels(matrix.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(matrix)))
    ax.set_yticklabels(matrix.index)
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            value = matrix.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                    color="black" if abs(value) < 0.6 else "white", fontsize=10)
    ax.set_title(
        f"G4: attribution consistency across crises\n"
        f"stability index = {scores['stability_index']:.3f}"
    )
    fig.colorbar(image, ax=ax, label="Spearman rank correlation")
    fig.tight_layout()
    path = figures_dir / "attribution_consistency_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written["heatmap"] = path

    # --- grouped bars ----------------------------------------------------
    top = top_features_by_episode(profiles, scores["top_k"])
    features = list(dict.fromkeys(top["feature"]))
    episodes = list(profiles.columns)
    width = 0.8 / max(len(episodes), 1)

    fig, ax = plt.subplots(figsize=(max(8.0, 1.1 * len(features)), 5.0))
    for offset, episode in enumerate(episodes):
        block = top[top["episode"] == episode].set_index("feature")
        heights = [float(block["share_of_total"].get(f, 0.0)) for f in features]
        ax.bar([i + offset * width for i in range(len(features))], heights,
               width=width, label=episode, edgecolor="black", linewidth=0.4)
    ax.set_xticks([i + 0.4 - width / 2 for i in range(len(features))])
    ax.set_xticklabels(features, rotation=40, ha="right")
    ax.set_ylabel("share of total |attribution|")
    ax.set_title("G4: top drivers per crisis episode")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = figures_dir / "top_features_per_episode.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written["bars"] = path
    return written
