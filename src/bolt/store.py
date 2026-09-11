"""Durable prediction ledger (ChainGuard automation layer).

An append-only JSON-lines store of every prediction the system has issued, and
of every outcome the Evaluation Agent has resolved.

Append-only is the point. The on-chain commitment proves a prediction existed at
a time; this ledger is the local record of what the system said and what
happened next. If it could be rewritten, the track record it produces would be
worth exactly as much as the private backtests G5 exists to replace.

Two files:

* ``predictions.jsonl`` - one canonical payload per line, as committed.
* ``outcomes.jsonl``    - one resolution per line, written once the horizon closes.

Resolutions are separate rather than patched into the prediction line, so a
prediction is never edited after the fact.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from bolt.logging_setup import get_logger

log = get_logger(__name__)

PREDICTIONS_FILE = "predictions.jsonl"
OUTCOMES_FILE = "outcomes.jsonl"
RUNS_FILE = "runs.jsonl"


@dataclass(frozen=True, slots=True)
class LedgerPaths:
    predictions: Path
    outcomes: Path
    runs: Path


class PredictionStore:
    """Append-only ledger of predictions, outcomes and automation runs."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.paths = LedgerPaths(
            predictions=self.root / PREDICTIONS_FILE,
            outcomes=self.root / OUTCOMES_FILE,
            runs=self.root / RUNS_FILE,
        )

    # -- writing -----------------------------------------------------------
    @staticmethod
    def _append(path: Path, record: dict) -> None:
        line = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def record_prediction(self, payload: dict, digest: str, committed: bool,
                          tx_hash: str | None = None, block_time: int | None = None) -> dict:
        """Append one prediction. Returns the stored record.

        A prediction id that is already present is NOT written again: the
        registry contract refuses overwrites, and the local ledger mirrors that
        rule so the two cannot disagree.
        """
        identifier = payload["prediction_id"]
        if self.has_prediction(identifier):
            log.info("prediction %s already in the ledger; not duplicating", identifier[:18])
            return self.get_prediction(identifier)

        record = {
            "prediction_id": identifier,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "digest": digest,
            "committed": bool(committed),
            "tx_hash": tx_hash,
            "block_time": block_time,
            "payload": payload,
        }
        self._append(self.paths.predictions, record)
        return record

    def record_outcome(self, outcome: dict) -> dict:
        record = {**outcome, "resolved_at": datetime.now(timezone.utc).isoformat()}
        self._append(self.paths.outcomes, record)
        return record

    def record_run(self, summary: dict) -> dict:
        record = {**summary, "ran_at": datetime.now(timezone.utc).isoformat()}
        self._append(self.paths.runs, record)
        return record

    # -- reading -----------------------------------------------------------
    @staticmethod
    def _read(path: Path) -> Iterator[dict]:
        if not path.is_file():
            return iter(())

        def generate() -> Iterator[dict]:
            with path.open("r", encoding="utf-8") as handle:
                for number, line in enumerate(handle, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        log.warning("%s line %d is not valid JSON; skipped", path.name, number)

        return generate()

    def predictions(self) -> list[dict]:
        return list(self._read(self.paths.predictions))

    def outcomes(self) -> list[dict]:
        return list(self._read(self.paths.outcomes))

    def runs(self) -> list[dict]:
        return list(self._read(self.paths.runs))

    def has_prediction(self, prediction_id: str) -> bool:
        return any(p["prediction_id"] == prediction_id for p in self.predictions())

    def get_prediction(self, prediction_id: str) -> dict | None:
        for record in self.predictions():
            if record["prediction_id"] == prediction_id:
                return record
        return None

    def resolved_ids(self) -> set[str]:
        return {o["prediction_id"] for o in self.outcomes()}

    def unresolved(self) -> list[dict]:
        """Predictions with no recorded outcome yet."""
        done = self.resolved_ids()
        return [p for p in self.predictions() if p["prediction_id"] not in done]

    # -- summary -----------------------------------------------------------
    def track_record(self, training_boundary: str | None = None) -> dict:
        """Aggregate counts, split by whether the call was genuinely out of sample.

        ``training_boundary`` is the last date the deployment models were fitted
        on. Predictions dated at or before it are IN-SAMPLE: the model already
        saw that period, so a hit proves nothing about foresight. Reporting a
        combined precision without that split would be the exact hindsight
        fitting G5 exists to make impossible.

        Deliberately does not compute accuracy.
        """
        outcomes = self.outcomes()
        counts: dict[str, int] = {}
        for outcome in outcomes:
            key = outcome.get("classification", "UNKNOWN")
            counts[key] = counts.get(key, 0) + 1

        resolved = sum(v for k, v in counts.items() if k != "UNRESOLVED")
        true_positive = counts.get("TRUE_POSITIVE", 0)
        false_positive = counts.get("FALSE_POSITIVE", 0)
        false_negative = counts.get("FALSE_NEGATIVE", 0)

        precision = (
            true_positive / (true_positive + false_positive)
            if (true_positive + false_positive) else None
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if (true_positive + false_negative) else None
        )
        record = {
            "predictions": len(self.predictions()),
            "committed": sum(1 for p in self.predictions() if p.get("committed")),
            "resolved": resolved,
            "pending": len(self.unresolved()),
            "counts": counts,
            "precision": precision,
            "recall": recall,
            "training_boundary": training_boundary,
        }
        if training_boundary:
            record["out_of_sample"] = self._subset(outcomes, training_boundary, True)
            record["in_sample"] = self._subset(outcomes, training_boundary, False)
        return record

    @staticmethod
    def _subset(outcomes: list[dict], boundary: str, after: bool) -> dict:
        """Counts, precision and recall for one side of the training boundary."""
        selected = [
            o for o in outcomes
            if (str(o.get("as_of_date", "")) > boundary) == after
            and o.get("classification") not in (None, "UNRESOLVED")
        ]
        counts: dict[str, int] = {}
        for outcome in selected:
            key = outcome["classification"]
            counts[key] = counts.get(key, 0) + 1

        true_positive = counts.get("TRUE_POSITIVE", 0)
        false_positive = counts.get("FALSE_POSITIVE", 0)
        false_negative = counts.get("FALSE_NEGATIVE", 0)
        alerts = true_positive + false_positive
        actual = true_positive + false_negative
        return {
            "resolved": len(selected),
            "counts": counts,
            "alerts": alerts,
            "precision": (true_positive / alerts) if alerts else None,
            "recall": (true_positive / actual) if actual else None,
        }
