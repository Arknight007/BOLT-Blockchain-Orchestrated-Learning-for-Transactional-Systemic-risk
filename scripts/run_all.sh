#!/usr/bin/env bash
# Full reproduction: raw ingestion -> every figure and table in outputs/.
#
# Acceptance standard (BOLT_SPEC.md Section 10): a third party clones the repo,
# sets two environment variables, runs this script, and reproduces every number
# in the report.
#
#   git clone <repo> && cd bolt
#   python -m venv .venv && . .venv/Scripts/activate   # Windows
#   pip install -r requirements.txt && pip install -e .
#   cp .env.example .env && $EDITOR .env
#   bash scripts/run_all.sh

set -euo pipefail

CONFIG="${BOLT_CONFIG:-config/default.yaml}"

echo "==> [0/6] environment"
python -c "import bolt; print('bolt', bolt.__version__, 'commit', bolt.git_commit_sha())"

echo "==> [1/6] ingest (cached; a second run makes zero API calls)"
bolt ingest --config "$CONFIG"

echo "==> [2/6] build panel, features, labels, windows"
bolt build --config "$CONFIG"

echo "==> [3/6] train every model on the walk-forward folds"
bolt train --config "$CONFIG" --model all

echo "==> [4/6] evaluate and report"
bolt evaluate --config "$CONFIG" --all --report

echo "==> [5/6] explain: attribution consistency across crises (G4)"
bolt explain --config "$CONFIG" --model lstm --consistency --analogue

echo "==> [6/6] tests"
pytest -q

echo
echo "Done. Tables in outputs/tables/, figures in outputs/figures/."
echo "Committing a live prediction on-chain is deliberately NOT part of this script:"
echo "  bolt predict --asset BTC --as-of \$(date +%F) --commit"
