# Notebooks

**This folder is intentionally empty until Phase 6.**

`demo.ipynb` is the Phase 6 deliverable — the artefact shown to the panel. Per
BOLT_SPEC.md §11 it runs top to bottom without error and walks through:

1. Load the frozen dataset (`data/processed/panel.parquet`) and verify its SHA-256
   against `DATA_CARD.md`.
2. Risk-score timeline with the four crisis episodes marked.
3. The model comparison table — all seven models, per-fold and aggregate.
4. One explained warning, with its ranked risk drivers.
5. That warning's nearest historical analogue.
6. That warning's on-chain proof: payload, digest, block timestamp.

It is deliberately written **last**. A demo notebook is a view over the pipeline,
so building it before the pipeline exists would mean hardcoding numbers into it —
which Rule 12.2 forbids. Every figure in the notebook is produced by the same code
paths `scripts/run_all.sh` calls, so the notebook cannot drift from the results.

## Running it (once Phase 6 lands)

```bash
pip install -r requirements.txt && pip install -e .
bash scripts/run_all.sh     # populates data/processed/ and outputs/
jupyter lab notebooks/demo.ipynb
```
