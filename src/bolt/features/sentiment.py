"""Headline sentiment features (BOLT_SPEC.md Section 5).

Strictly point-in-time under Guard 2: a headline contributes to date ``t`` only
if its publication timestamp is at or before ``t``. Text published after a crash
trivially predicts it, and enforcing this is a correctness requirement rather
than a best-effort target.

Default backend is a deterministic finance lexicon: no GPU, no model download,
identical output on every run. The optional FinBERT backend
(``features.sentiment_backend.backend: finbert``) runs locally at inference only
- BOLT never trains or fine-tunes a language model.

When no headline source is configured, the polarity columns are left MISSING
rather than zero-filled. Zero is not "no news"; zero is "neutral news", and the
two are different claims.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bolt.align.panel import assert_no_future_timestamps
from bolt.logging_setup import get_logger

log = get_logger(__name__)

SENTIMENT_COLUMNS = [
    "headline_polarity_1d", "headline_polarity_7d", "headline_count_z", "negative_ratio_7d",
]

# A compact finance-domain lexicon. Deliberately small and auditable: a panel can
# read it in full, which is not true of a 110M-parameter transformer.
NEGATIVE_TERMS = frozenset({
    "crash", "plunge", "plummet", "collapse", "slump", "tumble", "dive", "sink",
    "selloff", "sell-off", "bear", "bearish", "downturn", "decline", "drop", "fall",
    "liquidation", "liquidated", "insolvency", "insolvent", "bankrupt", "bankruptcy",
    "hack", "hacked", "exploit", "breach", "stolen", "scam", "fraud", "ponzi",
    "ban", "banned", "crackdown", "lawsuit", "sue", "sued", "investigation", "probe",
    "halt", "halted", "freeze", "frozen", "suspend", "suspended", "default",
    "depeg", "de-peg", "contagion", "panic", "fear", "capitulation", "margin",
    "warning", "risk", "loss", "losses", "weak", "weakness", "concern", "worry",
})

POSITIVE_TERMS = frozenset({
    "surge", "soar", "rally", "rebound", "recover", "recovery", "gain", "gains",
    "bull", "bullish", "uptrend", "breakout", "record", "high", "growth", "rise",
    "adoption", "approval", "approved", "partnership", "launch", "upgrade",
    "institutional", "inflow", "accumulate", "accumulation", "optimism", "confidence",
    "milestone", "boost", "strong", "strength", "support", "expand", "expansion",
})


def score_lexicon(title: str) -> float:
    """Polarity in [-1, 1] from term counts. Deterministic and inspectable."""
    if not title:
        return 0.0
    words = [w.strip(".,!?:;\"'()[]").lower() for w in title.split()]
    negative = sum(1 for w in words if w in NEGATIVE_TERMS)
    positive = sum(1 for w in words if w in POSITIVE_TERMS)
    total = negative + positive
    if total == 0:
        return 0.0
    return (positive - negative) / total


def score_headlines(headlines: pd.DataFrame, backend: dict) -> pd.DataFrame:
    """Attach a ``polarity`` column. Backend is ``lexicon`` (default) or ``finbert``."""
    if headlines.empty:
        return headlines.assign(polarity=pd.Series(dtype=float))

    name = backend.get("backend", "lexicon")
    if name == "lexicon":
        polarity = headlines["title"].map(score_lexicon)
    elif name == "finbert":
        polarity = _score_finbert(headlines["title"].tolist(), backend)
    else:
        raise ValueError(f"unknown sentiment backend {name!r}; expected 'lexicon' or 'finbert'")
    return headlines.assign(polarity=np.asarray(polarity, dtype=float))


def _score_finbert(titles: list[str], backend: dict) -> np.ndarray:
    """Optional transformer backend. Requires requirements-sentiment.txt."""
    try:
        from transformers import pipeline
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "the finbert backend needs `pip install -r requirements-sentiment.txt`; "
            "or set features.sentiment_backend.backend to 'lexicon'"
        ) from exc
    classifier = pipeline("sentiment-analysis", model=backend["finbert_model"], truncation=True)
    limit = backend["max_headline_chars"]
    scores = []
    for result in classifier([t[:limit] for t in titles]):
        sign = {"positive": 1.0, "negative": -1.0}.get(result["label"].lower(), 0.0)
        scores.append(sign * float(result["score"]))
    return np.asarray(scores, dtype=float)


def compute_sentiment(
    dates: pd.DatetimeIndex,
    headlines: pd.DataFrame,
    fear_greed: pd.Series,
    params: dict,
) -> pd.DataFrame:
    """The four configured sentiment features on the given date index.

    Guard 2 is enforced per date: only headlines published at or before the date
    contribute to that date's values.
    """
    out = pd.DataFrame(index=dates, columns=SENTIMENT_COLUMNS, dtype=float)

    if headlines.empty:
        # No headline source configured. Fear and Greed is a real published daily
        # series, so it stands in for aggregate polarity - and says so in the card.
        log.warning(
            "no headlines available; deriving polarity from the Fear and Greed index "
            "and leaving headline_count_z missing rather than zero-filling it"
        )
        fg = fear_greed.reindex(dates)
        # Map 0..100 onto -1..1 so the sign convention matches lexicon polarity.
        polarity = (fg - 50.0) / 50.0
        windows = params["sentiment_windows"]
        out["headline_polarity_1d"] = polarity
        out["headline_polarity_7d"] = polarity.rolling(
            windows[-1], min_periods=windows[-1]
        ).mean()
        out["negative_ratio_7d"] = (
            (polarity < 0).astype(float).rolling(windows[-1], min_periods=windows[-1]).mean()
        )
        out["headline_count_z"] = np.nan
        return out

    assert_no_future_timestamps(headlines, "published_at", dates.max())
    scored = score_headlines(headlines, params["sentiment_backend"])
    scored = scored.set_index(pd.DatetimeIndex(scored["published_at"]).normalize())

    daily_polarity = scored["polarity"].groupby(level=0).mean().reindex(dates)
    daily_count = scored["polarity"].groupby(level=0).size().reindex(dates).fillna(0.0)
    daily_negative = (
        (scored["polarity"] < 0).groupby(level=0).mean().reindex(dates)
    )

    short, long = params["sentiment_windows"][0], params["sentiment_windows"][-1]
    count_window = params["sentiment_count_z_window"]

    out["headline_polarity_1d"] = daily_polarity.rolling(short, min_periods=1).mean()
    out["headline_polarity_7d"] = daily_polarity.rolling(long, min_periods=long).mean()
    out["negative_ratio_7d"] = daily_negative.rolling(long, min_periods=long).mean()
    mean = daily_count.rolling(count_window, min_periods=count_window).mean()
    std = daily_count.rolling(count_window, min_periods=count_window).std()
    out["headline_count_z"] = (daily_count - mean) / std.replace(0.0, np.nan)
    return out
