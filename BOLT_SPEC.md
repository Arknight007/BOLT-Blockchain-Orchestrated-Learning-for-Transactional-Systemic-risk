# BOLT — Build Specification for Claude Code

Paste this file into the repo root as `BOLT_SPEC.md` and point Claude Code at it.
Suggested opening instruction: *"Read BOLT_SPEC.md and implement Phase 0 through Phase 4. Do not skip the leakage guards. Stop after each phase and show me the acceptance test output."*

---

## 0. What you are building

**BOLT** — a cryptocurrency crash early-warning system that predicts the probability of a severe market decline within a forward horizon, explains every warning in terms of named risk drivers, and commits each prediction to a public blockchain **before the outcome is known** so the track record is independently auditable.

This is a final-year research capstone. The output must be **reproducible, honestly evaluated, and defensible under examination**. Impressive-looking numbers obtained by leakage are worse than modest honest ones. Read Section 4 before writing any feature code.

### Base paper and the gaps this system fills

Base paper: **Ke, Z., Cao, Y., Chen, Z., Yin, Y., He, S., Cheng, Y. (2025), "Early warning of cryptocurrency reversal risks via multi-source data," *Finance Research Letters*, Art. 107890.** It uses LSTM on blockchain metrics + social sentiment + regulatory signals to predict pin-bar reversal events on Bitcoin, reporting F1 ≈ 0.703, with SHAP showing blockchain features contribute >33% of predictive power, and SMOTE improving rare-event recall by ~19%.

Every module below exists to close one of five gaps. **Keep these visible in the code** — each gap maps to a module, and the README must state which module closes which gap.

| Gap | In the base paper | What BOLT must do | Module |
|---|---|---|---|
| **G1** | Target is a candlestick pattern (pin-bar reversal), not an economically meaningful event | Predict a drawdown beyond a threshold within a forward horizon | `labeling/` |
| **G2** | Single asset (Bitcoin); no cross-asset structure — the authors' own stated future work | Multi-asset with correlation, lead-lag and contagion features | `features/contagion.py` |
| **G3** | Comparison set omits gradient boosting and GRU | Benchmark LSTM, GRU, XGBoost, RF, LogReg, volatility rule on identical features | `models/` |
| **G4** | SHAP applied but explanations never tested for stability across separate episodes | Measure whether the same drivers recur across independent crisis episodes | `explain/consistency.py` |
| **G5** | Results rest on a private backtest no reader can verify | Hash and commit every prediction on-chain before the outcome | `chain/` |

**G4 and G5 are the novel contributions.** If time runs short, cut scope elsewhere — never these two.

---

## 1. Repository layout

Create exactly this structure.

```
bolt/
├── BOLT_SPEC.md
├── README.md
├── requirements.txt
├── config/
│   ├── default.yaml           # single source of truth for all parameters
│   └── assets.yaml            # frozen asset list — never edited after Phase 1
├── data/
│   ├── raw/                   # untouched API responses (gitignored, cached)
│   ├── interim/               # aligned panels
│   └── processed/             # frozen dataset + labels (small enough to commit)
├── src/bolt/
│   ├── __init__.py
│   ├── config.py              # loads + validates YAML, returns frozen dataclass
│   ├── ingest/
│   │   ├── market.py          # CoinGecko + Binance
│   │   ├── onchain.py         # Etherscan / Dune / public endpoints
│   │   ├── news.py            # CryptoPanic / RSS
│   │   └── cache.py           # disk cache keyed by (source, asset, date-range)
│   ├── align/
│   │   └── panel.py           # build (date, asset) panel; enforce point-in-time
│   ├── features/
│   │   ├── technical.py
│   │   ├── onchain.py
│   │   ├── sentiment.py
│   │   └── contagion.py       # G2
│   ├── labeling/
│   │   └── crash.py           # G1
│   ├── windows.py             # sliding-window tensor builder
│   ├── models/
│   │   ├── base.py            # common fit/predict_proba interface
│   │   ├── lstm_numpy.py      # from scratch — no framework
│   │   ├── gru_numpy.py
│   │   ├── gbm.py             # XGBoost
│   │   ├── classical.py       # LogReg, RandomForest, MLP
│   │   └── rule.py            # volatility-threshold baseline
│   ├── evaluate/
│   │   ├── splits.py          # walk-forward, event-aware, embargoed
│   │   ├── metrics.py         # PR-AUC, F1, lead time, Brier, FAR
│   │   └── report.py
│   ├── explain/
│   │   ├── attribution.py     # SHAP + gradient attribution
│   │   ├── consistency.py     # G4
│   │   └── analogue.py        # nearest historical regime retrieval
│   ├── chain/
│   │   ├── payload.py         # canonical JSON + SHA-256
│   │   ├── client.py          # web3 submit / read
│   │   └── verify.py          # independent verification path
│   └── cli.py                 # typer/argparse entrypoints
├── contracts/
│   ├── PredictionRegistry.sol
│   └── deploy.py
├── notebooks/
│   └── demo.ipynb             # the thing you show the panel
├── tests/
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── predictions/
└── scripts/
    └── run_all.sh             # full reproduction, raw → figures
```

---

## 2. Configuration

Everything parameterised lives in `config/default.yaml`. **No magic numbers anywhere in the code.**

```yaml
data:
  start_date: "2020-01-01"
  end_date:   "2025-12-31"
  frequency:  "1d"

assets:                        # frozen in config/assets.yaml, declared BEFORE any modelling
  universe_file: "config/assets.yaml"
  primary: "BTC"

labeling:                      # G1
  drawdown_threshold: 0.20     # 20% decline
  horizon_days: 14
  method: "forward_min"        # min of forward window vs today's close
  sensitivity_grid:            # sensitivity analysis — required, not optional
    thresholds: [0.15, 0.20, 0.25]
    horizons:   [7, 14, 30]

windows:
  lookback_days: 30
  stride: 1

features:
  technical:  [realized_vol_7, realized_vol_14, realized_vol_30, volume_z_20,
               drawdown_depth, drawdown_duration, rsi_14, momentum_10, atr_14]
  onchain:    [exchange_netflow_7, whale_tx_count, active_addresses,
               stablecoin_netflow, nvt_ratio]
  sentiment:  [headline_polarity_1d, headline_polarity_7d, headline_count_z,
               negative_ratio_7d]
  contagion:  [mean_pairwise_corr_30, corr_dispersion_30, btc_lead_lag_5,
               eigen_centrality_30]

split:
  scheme: "walk_forward_expanding"
  folds:
    - {train_end: "2021-12-31", test: ["2022-01-01","2022-12-31"]}
    - {train_end: "2022-12-31", test: ["2023-01-01","2023-12-31"]}
    - {train_end: "2023-12-31", test: ["2024-01-01","2024-12-31"]}
    - {train_end: "2024-12-31", test: ["2025-01-01","2025-12-31"]}
  embargo_days: 30             # MUST be >= lookback_days + horizon_days... see §4

imbalance:
  method: "class_weight"       # primary
  compare_with: ["smote", "none"]   # ablation vs base paper's SMOTE

models:
  lstm:  {hidden: 32, epochs: 60, lr: 0.003, seed: 42, dropout: 0.1}
  gru:   {hidden: 32, epochs: 60, lr: 0.003, seed: 42}
  xgb:   {n_estimators: 400, max_depth: 4, learning_rate: 0.05, seed: 42}
  rf:    {n_estimators: 400, max_depth: 8, seed: 42}
  logreg:{C: 1.0, max_iter: 2000, seed: 42}
  rule:  {vol_quantile: 0.90}

evaluation:
  primary_metric: "pr_auc"
  report: ["pr_auc","f1","precision","recall","false_alarm_rate","brier","lead_time_days"]
  forbid_accuracy: true        # enforce in code — see §6

crisis_episodes:               # for episode-level analysis (G4)
  - {name: "May 2021 selloff", start: "2021-05-10", end: "2021-05-25"}
  - {name: "Terra/LUNA",       start: "2022-05-07", end: "2022-05-16"}
  - {name: "FTX collapse",     start: "2022-11-06", end: "2022-11-14"}
  - {name: "USDC depeg",       start: "2023-03-10", end: "2023-03-14"}

chain:
  network: "polygon_amoy"
  rpc_env_var: "BOLT_RPC_URL"
  private_key_env_var: "BOLT_PRIVATE_KEY"   # NEVER hardcode; read from env only
  contract_address_file: "outputs/contract_address.txt"
```

---

## 3. Data ingestion

### Requirements

- Every ingest function returns a `pandas.DataFrame` with a **timezone-aware UTC** `timestamp` index.
- **Cache everything to `data/raw/`.** Re-running must not re-hit APIs. Cache key = `(source, asset, start, end, frequency)`.
- Handle rate limits with exponential backoff. Never lose data silently — log every gap.
- If a source is unavailable, the pipeline must **fail loudly**, not silently produce a column of zeros.

### Sources

| Module | Source | Notes |
|---|---|---|
| `ingest/market.py` | CoinGecko free API, Binance public REST | OHLCV, volume, market cap, BTC dominance. Binance is the primary source for candles (it's the venue), CoinGecko for cap/dominance. |
| `ingest/onchain.py` | Etherscan API, public blockchain endpoints, Dune (optional) | Exchange netflow, whale transaction counts, active addresses, stablecoin flows. **Where a metric is unavailable free, implement it as a documented proxy and record that in `data/processed/DATA_CARD.md`.** Do not silently substitute. |
| `ingest/news.py` | CryptoPanic API, news RSS archives | Headline text + **publication timestamp**. The timestamp is load-bearing — see §4. |

### Write a data card

`data/processed/DATA_CARD.md`, generated automatically, containing: source per column, date coverage, missing-data percentage per column per year, any proxy substitutions, and the SHA-256 of the frozen dataset file. The panel will ask "where did your data come from" — this file is the answer.

---

## 4. Leakage guards — READ BEFORE WRITING FEATURE CODE

This is where projects like this silently fail. A leaking model produces excellent metrics and is worthless. Implement all five guards as **runtime assertions**, not comments.

**Guard 1 — point-in-time features.** Every feature value at date `t` may use only data with timestamp ≤ `t`. Implement `assert_point_in_time(df, feature_cols)` in `align/panel.py`, run it in CI. Rolling windows must be **trailing only** (`.rolling(w)`, never `center=True`).

**Guard 2 — news timestamps.** A headline may only contribute to sentiment at date `t` if its publication timestamp ≤ `t`. This is the single most common failure: text published *after* a crash trivially "predicts" it. Assert `headline.published_at <= feature_date` for every contributing row.

**Guard 3 — no target leakage in features.** No feature may be computed from any price at `t+1` or later. Write a unit test that shifts the entire price series forward by one day, recomputes features, and asserts the feature matrix changes only where it should.

**Guard 4 — embargo between train and test.** Because windows are `lookback_days` long and labels look `horizon_days` forward, a training window ending near the split boundary can overlap the test period. **Drop all samples whose `[window_start, label_end]` interval intersects the test range.** Set `embargo_days >= lookback_days + horizon_days`. Assert zero overlap after splitting.

**Guard 5 — scaler fitted on train only.** Fit normalisation on the training fold, apply to test. Never fit on the full dataset. Assert the scaler's `n_samples_seen_` equals the train-fold size.

**Additionally:** if you implement SMOTE for the ablation, apply it **inside the training fold only, after splitting** — never before. Synthetic oversampling of overlapping time-series windows creates near-duplicates that leak across folds. Document this in the ablation results.

---

## 5. Feature engineering and labelling

### `labeling/crash.py` (G1)

```python
def label_crashes(prices: pd.Series, threshold: float, horizon: int) -> pd.Series:
    """
    Binary label per date. 1 if the minimum close over (t, t+horizon]
    falls at least `threshold` below close[t].
    Returns int8 Series aligned to `prices.index`, with NaN for the final
    `horizon` dates (insufficient forward data — these MUST be dropped, not filled).
    """
```

Also implement `label_sensitivity_table(prices, grid)` returning a DataFrame of positive-class rate and episode coverage for every (threshold, horizon) pair. **This table goes in the report** — it is what makes the crash definition defensible rather than arbitrary.

Validation requirement: assert that the default parameters produce positive labels overlapping every episode listed in `crisis_episodes`. If they don't, the definition is wrong — surface this, don't tune around it.

### `features/contagion.py` (G2 — a core novelty)

Cross-asset features computed on a trailing 30-day window over the asset universe:

- `mean_pairwise_corr_30` — mean of the off-diagonal correlation matrix. Rising correlation means diversification is disappearing, which is a recognised stress signature.
- `corr_dispersion_30` — standard deviation of pairwise correlations.
- `btc_lead_lag_5` — cross-correlation of BTC returns at lags 1–5 against each altcoin; report the max-correlation lag.
- `eigen_centrality_30` — eigenvector centrality of the correlation graph (NetworkX), per asset.

These must be **trailing-window only** (Guard 1). Build the correlation graph with `networkx` and cache it per date.

### `windows.py`

```python
def build_windows(panel: pd.DataFrame, lookback: int, feature_cols: list[str],
                  label_col: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Returns X of shape (N, lookback, F), y of shape (N,), and a metadata
    DataFrame with columns [asset, window_start, window_end, label_end].
    The metadata is REQUIRED — the embargo logic in evaluate/splits.py uses it.
    """
```

---

## 6. Models

### Common interface — `models/base.py`

```python
class BoltModel(Protocol):
    name: str
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight=None) -> None: ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray:  # shape (N,), P(crash)
        ...
```

Sequence models take `X` of shape `(N, T, F)`. Tabular models take the same tensor and flatten it internally via a documented `flatten_windows()` helper — **the same inputs must reach every model**, otherwise the comparison is not fair and the panel will say so.

### `models/lstm_numpy.py` — implement from scratch

No PyTorch, no TensorFlow. Pure NumPy. This is a hard requirement: the report claims direct access to gate activations for attribution, and the from-scratch implementation is part of the contribution.

Implement:
- Forward pass with forget / input / candidate / output gates and cell state, exactly as in the standard formulation.
- Backpropagation through time with gradient clipping (clip norm at 5.0).
- Adam optimiser.
- Class-weighted binary cross-entropy loss.
- Deterministic seeding — same seed must reproduce identical weights.
- `gate_activations(X)` returning per-timestep gate values, for the attribution module.

Verify correctness with `tests/test_lstm_gradcheck.py`: numerical gradient check against analytic gradients on a small random batch, asserting relative error < 1e-5. **Do not proceed until this test passes.** A silently wrong backward pass will invalidate every result downstream.

### `models/gru_numpy.py`

Same discipline, GRU formulation (reset + update gates). Same gradient check test.

### `models/gbm.py`, `classical.py`, `rule.py`

XGBoost, Random Forest, Logistic Regression, MLP via scikit-learn. `rule.py` implements the non-ML baseline: fire when trailing realised volatility exceeds its rolling 90th percentile. This baseline exists to prove the ML models are earning their complexity — **if it wins, report that.**

---

## 7. Evaluation

### `evaluate/splits.py`

Implement `walk_forward_splits(meta, folds, embargo_days)` returning `(train_idx, test_idx)` pairs. Assert after construction:

```python
assert meta.loc[train_idx, "label_end"].max() < meta.loc[test_idx, "window_start"].min()
```

If this assertion fails, the split is leaking. Do not weaken the assertion — fix the split.

### `evaluate/metrics.py`

Implement PR-AUC, F1, precision, recall, false-alarm rate, Brier score, and:

```python
def lead_time_days(y_true_events: pd.Series, y_score: pd.Series,
                   threshold: float, episodes: list[dict]) -> pd.Series:
    """
    For each crisis episode, the number of days between the FIRST sustained
    threshold crossing (>= 2 consecutive days above threshold, to avoid counting
    single-day noise) and the episode start date. Negative means the warning
    came too late. Return one value per episode; do not average away failures.
    """
```

**Enforce the accuracy ban in code:**

```python
FORBIDDEN = {"accuracy", "accuracy_score"}
def guard_metrics(names):
    bad = FORBIDDEN & set(names)
    if bad:
        raise ValueError(
            f"{bad} is banned: with ~2% positive rate, predicting 'no crash' "
            f"always scores >95%. Use PR-AUC, F1 and lead time."
        )
```

This is not decoration — it is the single sentence that will earn credibility with an examiner.

### `evaluate/report.py`

Produce `outputs/tables/model_comparison.csv` and a matching matplotlib figure: one row per model, one column per metric, mean ± std across folds. Also produce per-fold results — a model that wins on average but fails on the FTX fold is an important finding.

---

## 8. Explainability

### `explain/attribution.py`

- SHAP `TreeExplainer` for XGBoost and Random Forest.
- SHAP `LinearExplainer` for Logistic Regression.
- Gradient × input attribution for the NumPy LSTM/GRU, using the gate activations exposed by the model.
- Aggregate per-timestep attributions to per-feature by summing over the window, and record both.

Output per positive prediction: a ranked list of `(feature, contribution, share_of_total)`.

### `explain/consistency.py` — G4, the headline research contribution

This module answers: **do the same drivers explain crashes across independent episodes, or does each crisis have its own signature?**

```python
def episode_attribution_profiles(model, X, meta, episodes) -> pd.DataFrame:
    """Mean |attribution| per feature, computed separately for each crisis episode."""

def consistency_scores(profiles: pd.DataFrame) -> dict:
    """
    Returns:
      - spearman_matrix: pairwise rank correlation of feature rankings between episodes
      - top_k_jaccard:   Jaccard overlap of the top-5 features between each episode pair
      - stability_index: mean pairwise Spearman across all episode pairs
    """
```

Produce a heatmap figure of the pairwise Spearman matrix and a grouped bar chart of the top features per episode.

**Report the result honestly either way.** High consistency = evidence of a generalisable crash signature. Low consistency = an important negative result about the limits of explanation-based risk systems. Both are publishable; only a fabricated result is not.

### `explain/analogue.py`

```python
def nearest_historical_analogue(current_vector, history_matrix, meta, k=3,
                                exclude_days=90) -> list[dict]:
    """
    Cosine or Mahalanobis nearest neighbours in standardised feature space.
    exclude_days prevents returning adjacent dates as 'analogues' — a window
    from last week is not a historical precedent.
    Returns [{date, similarity, what_happened_next_30d}, ...]
    """
```

---

## 9. Blockchain commitment layer — G5

### `contracts/PredictionRegistry.sol`

Keep it minimal — under 60 lines. Solidity ^0.8.20.

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract PredictionRegistry {
    struct Commitment { bytes32 digest; uint256 blockTime; address committer; }
    mapping(bytes32 => Commitment) private _commitments;

    event Committed(bytes32 indexed predictionId, bytes32 digest, uint256 blockTime);

    error AlreadyCommitted(bytes32 predictionId);

    function commit(bytes32 predictionId, bytes32 digest) external {
        if (_commitments[predictionId].blockTime != 0) revert AlreadyCommitted(predictionId);
        _commitments[predictionId] = Commitment(digest, block.timestamp, msg.sender);
        emit Committed(predictionId, digest, block.timestamp);
    }

    function get(bytes32 predictionId)
        external view returns (bytes32 digest, uint256 blockTime, address committer)
    {
        Commitment memory c = _commitments[predictionId];
        return (c.digest, c.blockTime, c.committer);
    }
}
```

Immutability comes from the overwrite revert. That single `revert` is the property the whole G5 claim rests on — write a test for it.

### `chain/payload.py`

```python
def canonical_payload(prediction: dict) -> bytes:
    """
    Deterministic serialisation: json.dumps(obj, sort_keys=True, separators=(',',':'),
    ensure_ascii=True).encode('utf-8').
    Determinism is mandatory — a verifier recomputing the hash must get a
    bit-identical result. Write a test asserting stability across dict insertion orders.
    """

def digest(payload: bytes) -> bytes:   # SHA-256
```

Payload fields: `prediction_id`, `asset`, `as_of_date`, `horizon_days`, `risk_score`, `probability`, `severity_band`, `top_drivers` (ranked list), `model_name`, `model_version` (git commit SHA), `feature_hash`, `code_version`.

### `chain/verify.py`

An **independent** verification path that a third party could run: given a published payload JSON and a prediction ID, recompute the digest, read the on-chain value, and report match/mismatch with the block timestamp. It must not import anything from the prediction pipeline — verification that depends on the system it verifies proves nothing.

Provide a CLI: `bolt verify --payload outputs/predictions/<id>.json --id <id>`.

### Safety

Read the private key from an environment variable only. Never write it to a file, never log it, never commit it. Testnet only — put a comment in `deploy.py` stating this. Add `.env` to `.gitignore` in Phase 0.

---

## 10. CLI

```bash
bolt ingest      --config config/default.yaml
bolt build       # align panel, features, labels, windows → data/processed/
bolt train       --model lstm|gru|xgb|rf|logreg|rule|all
bolt evaluate    --all --report
bolt explain     --model lstm --consistency
bolt predict     --asset BTC --as-of 2025-11-01 --commit
bolt verify      --payload <path> --id <prediction_id>
```

`scripts/run_all.sh` runs the entire chain from raw ingestion to every figure and table in `outputs/`. **A third party must be able to clone the repo, set two env vars, run one script, and reproduce every number in the report.** That is the acceptance standard.

---

## 11. Build order and acceptance criteria

Work in phases. **Stop after each phase and show the acceptance output.** Do not start a phase until the previous one passes.

### Phase 0 — Scaffold
Repo structure, `requirements.txt`, config loader with validation, `.gitignore` (include `.env`, `data/raw/`), logging setup, pytest wired up.
**Accept when:** `pytest` runs green on an empty suite and `python -m bolt.cli --help` prints all commands.

### Phase 1 — Data ✅ *required for the 50% review*
Ingestion for all three source families with caching; panel alignment; `DATA_CARD.md` generation; dataset freeze with a SHA-256 recorded.
**Accept when:** `bolt ingest && bolt build` produces `data/processed/panel.parquet`, the data card lists coverage and missing-data rates per year, and re-running hits cache with zero API calls.

### Phase 2 — Labels + features + leakage guards ✅ *required*
Crash labelling with the sensitivity table; all four feature families; **all five guards from §4 implemented as assertions with tests**.
**Accept when:** `pytest tests/test_leakage.py` passes all five guards, the sensitivity table is written to `outputs/tables/label_sensitivity.csv`, and default labels overlap all four crisis episodes.

### Phase 3 — Models + evaluation ✅ *required*
NumPy LSTM and GRU with passing gradient checks; XGBoost, RF, LogReg, MLP, volatility rule; walk-forward splits with embargo assertion; full metric suite with the accuracy ban.
**Accept when:** `bolt evaluate --all --report` writes `outputs/tables/model_comparison.csv` with per-fold and aggregate results for all seven models, and the embargo assertion passes on every fold.

### Phase 4 — Explainability ✅ *required — this is G4*
SHAP + gradient attribution; episode attribution profiles; consistency scores; heatmap and bar figures; analogue retrieval.
**Accept when:** `bolt explain --model lstm --consistency` writes the Spearman heatmap, the top-features-per-episode chart, and a one-line stability index to `outputs/tables/attribution_consistency.csv`.

> **Phases 1–4 constitute the 50% implementation to present at the interim review.** Everything above this line must work end-to-end before moving on.

### Phase 5 — Blockchain — G5
Contract, deploy script, payload hashing, commit and verify paths, overwrite-revert test.
**Accept when:** a prediction is committed to Polygon Amoy, `bolt verify` independently confirms the digest and returns the block timestamp, and a second commit with the same ID reverts.

### Phase 6 — Demo and reproduction
`notebooks/demo.ipynb`: load frozen data → risk score timeline with episodes marked → model comparison table → one explained warning with its drivers → its historical analogue → its on-chain proof. `scripts/run_all.sh` end to end.
**Accept when:** a clean clone reproduces every figure and table with one command.

---

## 12. Standing rules for the implementation

1. **Never report accuracy.** Enforced in code (§7).
2. **Never fabricate or hardcode a result.** If a number is not computed from the pipeline, it does not go in `outputs/`.
3. **Fail loudly.** A missing data source raises. A failed guard raises. Never fill missing features with zeros silently — log, document in the data card, and if coverage falls below a configured threshold, refuse to build.
4. **Seed everything.** Same config + same seed = identical outputs. Add `tests/test_reproducibility.py` asserting two runs produce identical predictions.
5. **Version every prediction.** `model_version` = git commit SHA. A committed prediction must be traceable to the exact code that produced it.
6. **Write the test before the fix** whenever a bug is found.
7. **If a simpler model wins, say so in the results.** That is a finding, not a failure. Do not tune the deep model until it wins.
8. Keep functions under ~50 lines and modules under ~300. Type-hint public functions. Docstring every module with which spec section it implements.

---

## 13. What "done for the review" looks like

By the interim review the following must exist and run:

- A frozen, documented dataset with a data card and a hash.
- A labelling procedure with a published sensitivity table.
- Seven models trained and compared under leak-free walk-forward validation, with per-fold results.
- Passing gradient checks on both from-scratch sequence models.
- Five leakage guards implemented as passing tests.
- Attribution consistency measured across four crisis episodes, with figures.
- A demo notebook that runs top to bottom without error.

If asked "what have you actually built", the answer is: *a reproducible pipeline from public APIs to a leak-free evaluated model comparison, with attribution consistency analysis — and here is the notebook, running.*
