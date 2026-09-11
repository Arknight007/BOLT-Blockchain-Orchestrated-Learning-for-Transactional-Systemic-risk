# ChainGuard

**A Multi-Agent, Explainable and Blockchain-Audited Cryptocurrency Crash Early-Warning System.**

> Neural networks predict patterns. Agents investigate and reason over evidence.
> Blockchain makes predictions auditable.

ChainGuard predicts the probability of a severe market decline within a forward
horizon, explains every warning in terms of named risk drivers, challenges its own
prediction before issuing it, and commits each prediction to a public blockchain
**before the outcome is known** — so the track record is independently auditable
rather than merely asserted.

Final-year research capstone. Engine package: `bolt`. Build specification:
[BOLT_SPEC.md](BOLT_SPEC.md). Panel rationale: `BOLT_Base_Paper_Answers.pdf`.

---

## Base paper

Ke, Z., Cao, Y., Chen, Z., Yin, Y., He, S., Cheng, Y. (2025).
*Early warning of cryptocurrency reversal risks via multi-source data.*
**Finance Research Letters**, Article 107890, Elsevier.

An LSTM on blockchain metrics, social sentiment and regulatory signals, predicting
pin-bar reversal events on Bitcoin. Reported F1 ≈ 0.703, SHAP attributing >33% of
predictive power to blockchain features, SMOTE improving recall by ~19%.

## Three technical contributions

| Layer | Question | Where |
|---|---|---|
| **AI/ML** | Can crashes be predicted at all? | [`src/bolt/models/`](src/bolt/models/) |
| **Agentic AI** | Can specialised agents investigate *and challenge* the prediction? | [`src/bolt/agents/`](src/bolt/agents/) |
| **Blockchain** | Can we *prove* the prediction was made before the outcome? | [`src/bolt/chain/`](src/bolt/chain/) |

## The five gaps, and the module closing each

| Gap | In the base paper | What ChainGuard does | Module |
|---|---|---|---|
| **G1** | Target is a candlestick pattern (pin-bar reversal) | Predicts a drawdown beyond a threshold within a forward horizon | [`labeling/crash.py`](src/bolt/labeling/crash.py) |
| **G2** | Single asset; no cross-asset structure — the authors' own stated future work | Multi-asset correlation, lead-lag and network-centrality features | [`features/contagion.py`](src/bolt/features/contagion.py) |
| **G3** | Comparison set omits gradient boosting and GRU | Benchmarks 7 models on identical features | [`models/`](src/bolt/models/) |
| **G4** | SHAP applied, never tested for stability across episodes | Measures whether the same drivers recur across independent crises | [`explain/consistency.py`](src/bolt/explain/consistency.py) |
| **G5** | Results rest on a private backtest no reader can verify | Hashes and commits every prediction on-chain before the outcome | [`chain/`](src/bolt/chain/) |

---

## Results (walk-forward, embargoed, 4 folds, 2022–2025)

**12,238 windows · 30 days × 21 features · 10.45% positive rate · random-baseline PR-AUC 0.083**

| model | PR-AUC | F1 | precision | recall | false alarm | lift over random |
|---|---|---|---|---|---|---|
| **rf** | **0.170 ± 0.107** | 0.126 | 0.221 | 0.194 | 0.099 | **+0.086** |
| logreg | 0.144 ± 0.103 | 0.072 | 0.159 | 0.250 | 0.226 | +0.061 |
| xgb | 0.134 ± 0.037 | 0.127 | 0.122 | 0.302 | 0.206 | +0.051 |
| lstm | 0.102 ± 0.059 | 0.100 | 0.071 | 0.363 | 0.365 | +0.018 |
| **rule** | 0.100 ± 0.052 | 0.112 | 0.082 | 0.413 | 0.368 | +0.017 |
| gru | 0.099 ± 0.049 | 0.122 | 0.082 | 0.473 | 0.443 | +0.015 |
| mlp | 0.085 ± 0.040 | 0.043 | 0.046 | 0.280 | 0.278 | +0.001 |

### Read this honestly

**Random Forest wins. The deep models do not.** And the one-line volatility rule
(PR-AUC 0.100) statistically ties the from-scratch LSTM (0.102) and beats both the
GRU and the MLP. Spec Rule 12.7 is explicit that this gets reported rather than
tuned away, so it is reported: **on this target, with this feature set, recurrence
is not earning its complexity.**

Three things follow, and all of them are findings:

1. **The task is genuinely hard.** The best model reaches roughly 2× the random
   baseline. Fold variance is large (RF ± 0.107) — a model that works in 2022 may
   not work in 2023.
2. **Our target is harder than the base paper's.** Ke et al. report F1 ≈ 0.703 on
   pin-bar reversals, a frequent microstructure pattern. A 20% drawdown within 14
   days is a rarer, more consequential event. The numbers are not comparable, and
   claiming otherwise would be dishonest.
3. **Any PR-AUC near 0.95 on this task would be evidence of leakage, not skill.**
   That is why §4's five guards are enforced as runtime assertions with
   [20 passing tests](tests/test_leakage.py).

Full per-fold results: [`outputs/tables/model_comparison_per_fold.csv`](outputs/tables/model_comparison_per_fold.csv).
A model that wins on average but fails on the FTX fold is an important finding, so
the per-fold table ships alongside the aggregate.

---

## The agent layer

Eleven agents. **Not all of them are LLM agents** — that is the common failure mode
of this architecture: eleven language-model calls wearing different hats, none
reproducible, testable, or defensible to an examiner. Each agent here uses the
simplest intelligence that does its job.

| Agent | Intelligence | Status |
|---|---|---|
| Market Intelligence | rules + analytics | ✅ implemented |
| On-Chain Intelligence | on-chain analytics | ✅ implemented |
| Quantitative | LSTM + XGBoost | ✅ implemented |
| Risk Orchestrator | weighted evidence aggregation | ✅ implemented |
| Skeptic / Red-Team | falsifiable rule checks | ✅ implemented |
| Decision | thresholds + abstention | ✅ implemented |
| Blockchain Audit | deterministic code | ✅ implemented |
| Evaluation | outcome analytics | ✅ implemented |
| Health | drift + decay monitoring | ✅ implemented |
| News & Sentiment | NLP | ⚠️ **partial** — sentiment level only, no event identification |
| Explanation | LLM narrative | ⚠️ **interface only** — real driver ranking, template prose |

**Why that split, and why it is principled rather than convenient:** every agent
whose output enters the hashed on-chain payload must be byte-for-byte
reproducible, or a verifier recomputes a different digest and the commitment
proves nothing. The two agents that remain partial are precisely the two whose
only job is to produce *prose* — the one thing a cryptographic commitment does not
need. Both declare their own limitation in every report they emit.

### The chain

```
Market ─┐
OnChain ─┼─> Orchestrator ─> Skeptic ─> Decision ─> Blockchain Audit ─> commitment
News   ─┤                        │
Quant  ─┘                        └─> Explanation (narrative, NOT committed)
```

Two properties worth demonstrating to a panel:

- **The Skeptic actively tries to disprove the warning.** Seven falsifiable checks:
  long-term trend conflict, contradictory stablecoin flow, model disagreement,
  degraded data, agent divergence, out-of-distribution volatility regime, weak
  historical analogue. Its counter-evidence is committed on-chain *alongside* the
  prediction — the record shows what the system knew might be wrong at the moment
  it committed.
- **The system can refuse to answer.** `INSUFFICIENT_EVIDENCE` is a first-class
  outcome. With no models loaded the Quant agent reports `UNAVAILABLE`, the
  Skeptic penalises the thin evidence base, and the Decision agent declines to
  state a risk level rather than issue one it cannot support.

### Automated monitoring and the honest track record

`bolt monitor` runs the loop that makes the Evaluation and Health agents real:
resolve every prediction whose horizon has closed, then predict for every asset,
then check drift. Run across the full history (44 cycles, 45 days apart) it
produced **269 predictions, 261 resolved** — and the ledger splits itself at the
training boundary, because a combined number would be worthless:

| period | resolved | alerts | TP | FP | FN | precision | recall |
|---|---|---|---|---|---|---|---|
| In-sample (to 2024-11-17) | 203 | 19 | 18 | 1 | 10 | 0.947 | 0.643 |
| **Out of sample (after)** | **58** | **0** | **0** | **0** | **9** | **undefined** | **0.000** |

> ### This is the headline finding, and it is not a good one
>
> **Out of sample the system issued no warnings at all** across 58 resolved
> predictions, and missed 9 real crashes. The in-sample precision of 0.947
> describes the period the models were fitted on and nothing else.
>
> Two things follow. First, the deployment models do not generalise past their
> training window on this target — consistent with the walk-forward table above,
> where the best PR-AUC is roughly 2× random. Second, and more usefully: this is
> exactly the failure an on-chain commitment record exists to expose. A private
> backtest would have reported 0.947 and stopped there.

### Agent chain on real dates

```
BTC 2022-05-06  (Terra/LUNA collapsed 2022-05-07)   CRITICAL  83/100
BTC 2022-11-05  (FTX collapsed 2022-11-06)          HIGH      69/100
BTC 2021-07-15                                      LOW       30/100
BTC 2023-08-01                                      LOW       24/100
BTC 2024-06-15                                      LOW       19/100
```

> ### ⚠️ Read this before quoting the two numbers above
>
> **These are IN-SAMPLE.** `bolt train` fits the deployment models on every
> window whose label resolves before 2024-11-17, so the 2022 episodes are inside
> the training data. Firing CRITICAL the day before Terra/LUNA is a demonstration
> that the **agent chain wires together correctly** — it is *not* evidence of
> predictive skill, and presenting it as such would be exactly the hindsight
> fitting that G5 exists to make impossible.
>
> The honest out-of-sample evidence is the walk-forward table above: **PR-AUC
> 0.170 against a 0.083 baseline.** That is the number to defend.

---

## Quick start

```bash
git clone <repo> && cd ChainGuard
python -m venv .venv && . .venv/Scripts/activate     # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
cp .env.example .env                                  # optional API keys
bolt --help
```

```
bolt ingest                                   # fetch + cache; a second run makes zero API calls
bolt build                                    # panel, features, labels, DATA_CARD.md, frozen parquet
bolt train    --model all                     # fit and persist all seven models
bolt evaluate --all --report                  # walk-forward metrics -> outputs/
bolt explain  --model xgb --consistency       # attribution + G4 consistency
bolt predict  --asset BTC --as-of 2025-11-01 --commit    # agent chain + on-chain commitment
bolt monitor  --cycles 44 --every-days 45       # automated resolve -> predict -> health
bolt serve    --port 8000                      # the ChainGuard terminal
bolt verify   --payload outputs/predictions/<id>.json --id <id>
```

## The terminal

```bash
bolt serve --port 8000     # then open http://127.0.0.1:8000
```

A read-only console over the real pipeline — no mock data, no second copy of the
numbers. Six views, keyboard-driven (`1`–`6`):

| View | What it shows |
|---|---|
| MONITOR | the agent chain streaming stage by stage as it executes, then the commitment and, for historical dates, what actually happened |
| MARKET | price with crisis bands and crash labels, the live feature vector with each value's percentile, the frozen universe |
| MODELS | the seven-model comparison, PR-AUC per fold, lead time per episode, label sensitivity |
| DRIVERS | G4 consistency matrix, top drivers by family, per-episode driver ranks |
| LEDGER | the prediction ledger split in-sample vs out-of-sample, with independent digest verification per row |
| SYSTEM | pipeline state, model health, feature drift (PSI), and which agents are implemented |

Bound to localhost by default: it exposes a prediction ledger and can spend
testnet gas. Nothing in it retrains or rewrites the frozen dataset.

Full reproduction: `bash scripts/run_all.sh`.

## Data

Every raw observation comes from a public API or the public blockchain. Nothing is
private, purchased, or fabricated; the dataset regenerates from the scripts.

| Source | What | Why authentic |
|---|---|---|
| Binance public API | OHLCV, taker flow, trade counts | The venue where trades occurred — primary, not aggregated |
| DefiLlama | prices for unlisted assets; stablecoin supply | Public, keyless, full history |
| alternative.me | Fear & Greed index | Published daily composite |
| CoinGecko | last-resort prices | Free tier caps history at 365 days |

**16,854 rows × 25 columns, 2020-01-01 → 2025-12-31, 10 assets.**
[`data/processed/DATA_CARD.md`](data/processed/DATA_CARD.md) is generated by the
pipeline — never hand-written — and records per-column provenance, per-year
coverage, every proxy with its stated limitation, and the dataset SHA-256.

**On-chain data is proxied.** No Etherscan key is configured, so four ledger-native
metrics are documented proxies, each declaring what it stands in for *and where it
falls short*. The `Provenance` type refuses to construct a `PROXY` column without
both. The On-Chain agent repeats the caveat in every report and marks itself
`DEGRADED`.

## Honesty constraints, enforced in code

1. **Accuracy is never reported.** At a 10.45% positive rate, always predicting
   "no crash" scores 89.55%. `guard_metrics()` raises on any attempt; the config
   loader rejects a config that switches the ban off.
2. **Five leakage guards as runtime assertions** — point-in-time features, news
   timestamps, no target leakage, a ≥ `lookback + horizon` embargo, train-only
   scalers. Each is tested by constructing data that *would* leak and asserting
   the guard catches it.
3. **Fail loudly.** Missing sources raise. Missing features are never zero-filled —
   `headline_count_z` was *removed* from the feature set rather than faked, and
   the removal is recorded in the data card.
4. **Nothing fabricated.** If a number was not computed by the pipeline it does not
   appear in `outputs/`.
5. **Seeded and reproducible.** Same config + same seed = identical predictions.

## Build status

| Phase | Contents | Status |
|---|---|---|
| 0 | Scaffold, validated config, CLI | ✅ |
| 1 | Ingestion + caching, panel, data card, frozen dataset | ✅ |
| 2 | Labels + sensitivity table, 4 feature families, 5 leakage guards | ✅ |
| 3 | NumPy LSTM/GRU + gradient checks, 7-model bench, embargoed walk-forward | ✅ |
| 4 | SHAP + gradient attribution, episode consistency (G4) | ✅ |
| 5 | `PredictionRegistry.sol`, payload hashing, commit + verify (G5) | ✅ contract & payload tested; testnet deploy pending |
| 6 | Demo notebook, one-command reproduction | ✅ |
| — | Automation loop (`bolt monitor`) + ChainGuard terminal (`bolt serve`) | ✅ |
| 7 | LLM narrative backend, news event identification | ⬜ not started |

## Tests

```
tests/test_leakage.py        20   the five guards, each tested against data that would leak
tests/test_lstm_gradcheck.py 15   analytic vs numerical gradients, both models
tests/test_chain.py          19   payload determinism, tamper detection, verifier independence
tests/test_contract.py       11   compiled EVM: the overwrite revert G5 rests on
tests/test_agents.py         23   abstention, skeptic challenges, deterministic serialisation
tests/test_automation.py     13   append-only ledger, resolve-before-predict ordering
tests/test_config.py         21   config invariants that would silently corrupt results
tests/test_cli.py            16   command surface
tests/test_scaffold.py       ...  structure, clone survival, secret hygiene
```

## License

MIT.
