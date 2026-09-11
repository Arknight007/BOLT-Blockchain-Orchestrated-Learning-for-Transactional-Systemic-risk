#!/usr/bin/env bash
# ChainGuard - full reproduction: raw ingestion -> every figure and table in outputs/.
#
# Acceptance standard (BOLT_SPEC.md Section 10): a third party clones the repo,
# sets two environment variables, runs this script, and reproduces every number
# in the report.
#
#   git clone <repo> && cd ChainGuard
#   python -m venv .venv
#   . .venv/Scripts/activate          # Windows;  source .venv/bin/activate elsewhere
#   pip install -r requirements.txt && pip install -e .
#   cp .env.example .env              # optional: API keys
#   bash scripts/run_all.sh
#
# Runtime is roughly 25-35 minutes, dominated by the from-scratch NumPy LSTM and
# GRU. They are pure NumPy by design (BOLT_SPEC.md Section 6) - the cost is the
# point, not an oversight.

set -euo pipefail

CONFIG="${BOLT_CONFIG:-config/default.yaml}"
EXPLAIN_MODEL="${BOLT_EXPLAIN_MODEL:-xgb}"

echo "==> [0/7] environment"
python -c "import bolt; print('bolt', bolt.__version__, 'commit', bolt.git_commit_sha())"

echo
echo "==> [1/7] ingest (cached; a second run makes zero API calls)"
bolt ingest --config "$CONFIG"

echo
echo "==> [2/7] build panel, features, labels, windows, DATA_CARD.md"
bolt build --config "$CONFIG"

echo
echo "==> [3/7] train and persist every model"
bolt train --config "$CONFIG" --model all

echo
echo "==> [4/7] walk-forward evaluation with embargo"
bolt evaluate --config "$CONFIG" --all --report

echo
echo "==> [5/7] attribution and cross-episode consistency (G4)"
bolt explain --config "$CONFIG" --model "$EXPLAIN_MODEL" --consistency --analogue

echo
echo "==> [6/7] tests"
pytest -q

echo
echo "==> [7/7] demo notebook"
python scripts/make_demo_notebook.py

echo
echo "Done."
echo "  tables  : outputs/tables/"
echo "  figures : outputs/figures/"
echo "  notebook: notebooks/demo.ipynb"
echo
echo "Committing a live prediction on-chain is deliberately NOT part of this script:"
echo "it spends testnet gas and requires a funded key."
echo
echo "  python contracts/deploy.py                  # once, writes outputs/contract_address.txt"
echo "  bolt predict --asset BTC --as-of \$(date +%F) --commit"
echo "  bolt verify  --payload outputs/predictions/<id>.json --id <prediction_id>"
