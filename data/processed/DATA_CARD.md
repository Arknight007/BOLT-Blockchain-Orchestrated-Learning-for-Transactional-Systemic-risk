# DATA CARD

Generated automatically by `bolt build` on 2026-09-11 03:18 UTC.
Do not edit by hand: this file is regenerated on every build and any manual
change will be overwritten.

- **Code version:** `f8c3ec511b20344a6ac5bd00a95a8c7d771a71d1-dirty`
- **Config:** `default.yaml`, universe `assets.yaml`
- **Frozen dataset:** `data/processed/panel.parquet`
- **SHA-256:** `f8532919fc0529d4d64aec855222259018f17f32c8722957a4145f448bc4eab1`
- **Shape:** 16,854 rows x 25 columns
- **Coverage:** 2020-01-01 to 2025-12-31, 10 assets
- **Label:** 20% drawdown within 14 days,
  1,443 positive of 12,995 labelled (11.10%)

## Assets

| asset | role | rows | first | last | source |
|---|---|---|---|---|---|
| `BTC` | target | 2192 | 2020-01-01 | 2025-12-31 | binance |
| `ETH` | target | 2192 | 2020-01-01 | 2025-12-31 | binance |
| `BNB` | target | 2192 | 2020-01-01 | 2025-12-31 | binance |
| `XRP` | target | 2192 | 2020-01-01 | 2025-12-31 | binance |
| `LINK` | target | 2192 | 2020-01-01 | 2025-12-31 | binance |
| `UNI` | target | 1932 | 2020-09-17 | 2025-12-31 | binance |
| `HYPE` | target | 111 | 2025-09-12 | 2025-12-31 | coingecko |
| `ASTER` | target | 104 | 2025-09-19 | 2025-12-31 | coingecko |
| `USDT` | context | 1816 | 2020-01-01 | 2025-12-31 | defillama |
| `DAI` | context | 1931 | 2020-01-01 | 2025-12-31 | defillama |

Assets are absent before their inception date rather than back-filled: the panel
is deliberately ragged, because inventing history for an asset that did not yet
trade is exactly what Guard 1 forbids.

## Column sources

| column | source | kind | definition |
|---|---|---|---|
| `realized_vol_7` | derived | DERIVED | computed from primary columns |
| `realized_vol_14` | derived | DERIVED | computed from primary columns |
| `realized_vol_30` | derived | DERIVED | computed from primary columns |
| `volume_z_20` | derived | DERIVED | computed from primary columns |
| `drawdown_depth` | derived | DERIVED | computed from primary columns |
| `drawdown_duration` | derived | DERIVED | computed from primary columns |
| `rsi_14` | derived | DERIVED | computed from primary columns |
| `momentum_10` | derived | DERIVED | computed from primary columns |
| `atr_14` | derived | DERIVED | computed from primary columns |
| `exchange_netflow_7` | Binance public API (taker flow) | PROXY | 7-day sum of (taker buy volume - taker sell volume) divided by total volume |
| `whale_tx_count` | Binance public API (klines) | PROXY | z-score of mean trade size (quote_volume / trades) over a trailing window |
| `active_addresses` | Binance public API (klines) | PROXY | daily count of executed trades, as a network-activity level |
| `stablecoin_netflow` | DefiLlama stablecoins API | DERIVED | 7-day fractional change in the aggregate circulating USD value of all USD-pegged stablecoins |
| `nvt_ratio` | Binance (price and turnover) | PROXY | log of close price divided by trailing mean quote volume |
| `headline_polarity_1d` | derived | DERIVED | computed from primary columns |
| `headline_polarity_7d` | derived | DERIVED | computed from primary columns |
| `negative_ratio_7d` | derived | DERIVED | computed from primary columns |
| `mean_pairwise_corr_30` | derived | DERIVED | computed from primary columns |
| `corr_dispersion_30` | derived | DERIVED | computed from primary columns |
| `btc_lead_lag_5` | derived | DERIVED | computed from primary columns |
| `eigen_centrality_30` | derived | DERIVED | computed from primary columns |

## Proxy substitutions

No Etherscan API key is configured for this build, so the four ledger-native
metrics are documented proxies. Each states what it stands in for and where it
falls short. Nothing is silently substituted.

### `exchange_netflow_7`

- **Stands in for:** net ERC-20/BTC transfer volume into labelled exchange wallets
- **Actually measures:** 7-day sum of (taker buy volume - taker sell volume) divided by total volume
- **Source:** Binance public API (taker flow)
- **Limitation:** Measures aggressive order flow ON an exchange, not deposits INTO one. It captures the same economic pressure (willingness to sell at market) but misses coins moved to an exchange and not yet sold, which is the leading half of the true signal.

### `whale_tx_count`

- **Stands in for:** count of on-chain transfers above a large-value threshold
- **Actually measures:** z-score of mean trade size (quote_volume / trades) over a trailing window
- **Source:** Binance public API (klines)
- **Limitation:** A rising mean trade size indicates large actors are active, but cannot distinguish one whale from many co-ordinated mid-size traders, and sees only exchange activity rather than the whole ledger.

### `active_addresses`

- **Stands in for:** count of distinct addresses transacting on-chain that day
- **Actually measures:** daily count of executed trades, as a network-activity level
- **Source:** Binance public API (klines)
- **Limitation:** Exchange trade count moves with on-chain activity but is dominated by high-frequency participants and excludes all off-exchange settlement. Correlated with the true metric, not equal to it.

### `nvt_ratio`

- **Stands in for:** network value to on-chain transaction volume
- **Actually measures:** log of close price divided by trailing mean quote volume
- **Source:** Binance (price and turnover)
- **Limitation:** Two substitutions, both forced by CoinGecko's free tier refusing historical market capitalisation (error 10012, 365-day ceiling). Price stands in for network value, and exchange turnover for on-chain settled value. Within one asset's history supply moves slowly, so price tracks market cap up to a slowly-varying factor -- but the resulting level is NOT comparable across assets, only across time within an asset.

## Missing data by year

Percentages are **coverage** (1 - missing), computed over every row in the panel
including context assets.

| column | overall | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|
| `realized_vol_7` | 99.6% | 97.7% | 100.0% | 100.0% | 100.0% | 100.0% | 99.5% |
| `realized_vol_14` | 99.2% | 95.3% | 100.0% | 100.0% | 100.0% | 100.0% | 99.1% |
| `realized_vol_30` | 98.2% | 90.0% | 100.0% | 100.0% | 100.0% | 100.0% | 98.0% |
| `volume_z_20` | 76.9% | 76.1% | 79.4% | 75.4% | 75.1% | 76.9% | 78.3% |
| `drawdown_depth` | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| `drawdown_duration` | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| `rsi_14` | 99.2% | 95.3% | 100.0% | 100.0% | 100.0% | 100.0% | 99.1% |
| `momentum_10` | 99.4% | 96.7% | 100.0% | 100.0% | 100.0% | 100.0% | 99.3% |
| `atr_14` | 99.2% | 95.7% | 100.0% | 100.0% | 100.0% | 100.0% | 99.1% |
| `exchange_netflow_7` | 76.3% | 79.3% | 79.4% | 75.4% | 75.1% | 76.9% | 72.4% |
| `whale_tx_count` | 75.5% | 73.6% | 79.4% | 75.4% | 75.1% | 76.9% | 72.4% |
| `active_addresses` | 76.5% | 80.8% | 79.4% | 75.4% | 75.1% | 76.9% | 72.4% |
| `stablecoin_netflow` | 99.6% | 97.7% | 100.0% | 100.0% | 100.0% | 100.0% | 99.5% |
| `nvt_ratio` | 77.5% | 79.3% | 79.4% | 75.4% | 75.1% | 76.9% | 79.2% |
| `headline_polarity_1d` | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 99.7% | 100.0% |
| `headline_polarity_7d` | 99.3% | 98.0% | 100.0% | 100.0% | 100.0% | 98.0% | 99.6% |
| `negative_ratio_7d` | 99.6% | 98.0% | 100.0% | 100.0% | 100.0% | 100.0% | 99.6% |
| `mean_pairwise_corr_30` | 98.9% | 92.5% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| `corr_dispersion_30` | 98.9% | 92.5% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| `btc_lead_lag_5` | 98.9% | 92.5% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| `eigen_centrality_30` | 96.2% | 87.6% | 95.0% | 100.0% | 100.0% | 96.4% | 96.7% |

## Known gaps and decisions

- `headline_count_z` is REMOVED from the active feature set. It requires a CryptoPanic API key; none is configured, so its coverage is 0%. Rule 12.3 forbids zero-filling it, so it was dropped rather than faked. Restore it in config/default.yaml once BOLT_CRYPTOPANIC_KEY is set.
- No headline text source is configured, so `headline_polarity_*` and `negative_ratio_7d` are derived from the alternative.me Fear and Greed index (a real published daily series) rather than from headline polarity. This is a weaker sentiment signal than the base paper's and is recorded as such.
- CoinGecko's free tier refuses historical ranges older than 365 days (error 10012), which removed market capitalisation from the study window. `nvt_ratio` is therefore built from price and turnover; see its proxy entry.
- Coverage is REPORTED over every row but ENFORCED over target-asset rows only. Context assets (stablecoins) are never labelled and never become training samples, and DefiLlama serves no volume for them, so their missing volume columns cannot affect a model.
- `HYPE` contributes only 111 days (inception 2024-11-29, and CoinGecko's 365-day ceiling truncates it further). It post-dates every crisis episode, so it cannot contribute to the G4 episode analysis and reaches the contagion graph only in the final fold.
- `ASTER` contributes only 104 days (inception 2025-09-17, and CoinGecko's 365-day ceiling truncates it further). It post-dates every crisis episode, so it cannot contribute to the G4 episode analysis and reaches the contagion graph only in the final fold.

## Reproduction

```bash
pip install -r requirements.txt && pip install -e .
bolt ingest --config config/default.yaml    # cached; a second run makes zero API calls
bolt build  --config config/default.yaml    # regenerates this file and the parquet above
```

Verify you hold the same dataset that produced the reported results:

```bash
python -c "import hashlib,pathlib; print(hashlib.sha256(pathlib.Path('data/processed/panel.parquet').read_bytes()).hexdigest())"
# expected: f8532919fc0529d4d64aec855222259018f17f32c8722957a4145f448bc4eab1
```
