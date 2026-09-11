# Contracts

**This folder is intentionally empty until Phase 5.**

Phase 5 adds two files, per BOLT_SPEC.md §9:

- `PredictionRegistry.sol` — under 60 lines, Solidity ^0.8.20. A single mapping
  from `predictionId` to `(digest, blockTime, committer)`, a `commit` function
  that **reverts if the id is already present**, and a `get` view function.
- `deploy.py` — compiles with py-solc-x and deploys to Polygon Amoy, writing the
  address to `outputs/contract_address.txt`.

## Why the revert is the whole contribution

G5 claims the BOLT track record is auditable rather than merely asserted. That
claim rests entirely on one line — the overwrite revert in `commit`. Without it, a
prediction could be silently replaced after the outcome was known, and the ledger
would prove nothing at all. It gets its own test.

## Safety

**Testnet only.** The deploy key is read from `BOLT_PRIVATE_KEY` in the
environment — never hardcoded, never logged, never written to a file. Use a
throwaway account funded only with Amoy test MATIC.
