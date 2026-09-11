"""Live agent-chain and ledger endpoints.

The agent chain is streamed as Server-Sent Events, one agent at a time in the
order they actually execute. That is not decoration: the whole argument of the
architecture is that specialised agents investigate INDEPENDENTLY, an
orchestrator aggregates, and a skeptic then challenges the result. Showing the
final number alone would hide exactly the part that is novel.

The stream carries real agent output. It does not replay a recording.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, AsyncIterator

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from bolt.config import BoltConfig
from bolt.logging_setup import get_logger

log = get_logger(__name__)

#: Pause between streamed stages, in seconds. Long enough that a viewer can see
#: the chain proceed, short enough not to feel artificial.
STAGE_DELAY = 0.22


def _sse(event: str, payload: Any) -> str:
    body = json.dumps(payload, default=str, separators=(",", ":"))
    return f"event: {event}\ndata: {body}\n\n"


def _report_dict(report) -> dict:
    from bolt.web.api import _clean

    return _clean({
        "agent": report.agent,
        "risk": report.risk.value,
        "score": report.score,
        "confidence": report.confidence,
        "status": report.status.value,
        "evidence": [
            {
                "name": e.name, "value": e.value, "direction": e.direction,
                "weight": e.weight, "detail": e.detail,
            }
            for e in report.top_evidence(6)
        ],
        "notes": list(report.notes),
        "extra": report.extra,
    })


def build_live_router(cfg: BoltConfig, state) -> APIRouter:
    router = APIRouter(prefix="/api")

    def _prepare(asset: str, as_of: str | None):
        from bolt.agents.pipeline import build_context

        panel = state.panel()
        symbols = set(panel.index.get_level_values("asset"))
        if asset not in symbols:
            raise HTTPException(404, f"{asset} is not in the panel")

        latest = panel.index.get_level_values("date").max()
        when = pd.Timestamp(as_of) if as_of else latest
        if when.tzinfo is None:
            when = when.tz_localize("UTC")
        if when > latest:
            when = latest

        try:
            context = build_context(cfg, panel, asset, when, list(cfg.feature_columns))
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        return context, panel

    @router.get("/run")
    async def run(asset: str = Query(None), as_of: str = Query(None),
                  commit: bool = Query(False)) -> StreamingResponse:
        """Stream the agent chain, one stage at a time, as it executes."""
        from bolt.agents.pipeline import ChainGuardPipeline

        asset = asset or cfg.primary_asset
        context, panel = _prepare(asset, as_of)
        models, scaler = state.models()

        async def generate() -> AsyncIterator[str]:
            pipeline = ChainGuardPipeline(cfg, models, scaler)
            yield _sse("start", {
                "asset": asset,
                "as_of": context.as_of.date().isoformat(),
                "models": sorted(models),
                "horizon_days": cfg.horizon_days,
                "threshold": cfg.drawdown_threshold,
            })
            await asyncio.sleep(STAGE_DELAY)

            # 1-4: the evidence agents, independently.
            reports = []
            for agent in (pipeline.market, pipeline.onchain, pipeline.news, pipeline.quant):
                report = agent.analyse(context)
                reports.append(report)
                yield _sse("agent", _report_dict(report))
                await asyncio.sleep(STAGE_DELAY)

            # 5: aggregate.
            preliminary = pipeline.orchestrator.analyse(context, reports)
            yield _sse("orchestrator", _report_dict(preliminary))
            await asyncio.sleep(STAGE_DELAY)

            # 6: analogue retrieval feeds the skeptic's precedent check.
            analogues = _analogues(cfg, state, asset, context.as_of)
            yield _sse("analogues", analogues)
            await asyncio.sleep(STAGE_DELAY)

            # 7: challenge.
            skeptic = pipeline.skeptic.analyse(context, preliminary, reports, analogues)
            yield _sse("skeptic", _report_dict(skeptic))
            await asyncio.sleep(STAGE_DELAY)

            # 8: decide.
            decision = pipeline.decision.analyse(context, preliminary, skeptic, reports)
            yield _sse("decision", _report_dict(decision))
            await asyncio.sleep(STAGE_DELAY)

            with_skeptic = reports + [skeptic]
            explanation = pipeline.explanation.analyse(
                context, decision, with_skeptic, None, analogues
            )
            yield _sse("explanation", _report_dict(explanation))
            await asyncio.sleep(STAGE_DELAY)

            # 9: commit.
            severity = {}
            from bolt.agents.base import RiskLevel

            if decision.risk.rank >= RiskLevel.MODERATE.rank:
                severity = pipeline.quant.expected_severity(context, decision.score / 100.0)

            audit, commitment = pipeline.blockchain.analyse(
                context, decision, with_skeptic, commit=commit,
                expected_severity=severity,
            )
            from bolt.web.api import _clean

            yield _sse("commitment", _clean({
                "agent": audit.agent,
                "prediction_id": commitment.prediction.prediction_id,
                "digest": commitment.digest,
                "payload_path": str(commitment.payload_path),
                "committed": commitment.committed,
                "tx_hash": commitment.tx_hash,
                "block_time": commitment.block_time,
                "explorer_url": commitment.explorer_url,
                "reason": commitment.reason,
                "payload": commitment.prediction.to_dict(),
                "expected_severity": severity,
            }))
            await asyncio.sleep(STAGE_DELAY)

            # 10: what actually happened, when the horizon has already closed.
            yield _sse("outcome", _hindsight(cfg, panel, asset, context.as_of))
            yield _sse("done", {"finished_at": datetime.now(timezone.utc).isoformat()})

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @router.get("/ledger")
    def ledger(limit: int = Query(200)) -> dict:
        store = state.store()
        predictions = store.predictions()[-limit:][::-1]
        outcomes = {o["prediction_id"]: o for o in store.outcomes()}

        rows = []
        for record in predictions:
            payload = record["payload"]
            outcome = outcomes.get(record["prediction_id"])
            rows.append({
                "prediction_id": record["prediction_id"],
                "asset": payload["asset"],
                "as_of_date": payload["as_of_date"],
                "severity_band": payload["severity_band"],
                "risk_score": payload["risk_score"],
                "probability": payload["probability"],
                "digest": record["digest"],
                "committed": record.get("committed", False),
                "tx_hash": record.get("tx_hash"),
                "classification": outcome["classification"] if outcome else "PENDING",
                "actual_max_drawdown": outcome["actual_max_drawdown"] if outcome else None,
            })

        from bolt.web.api import _clean

        # Everything the deployment models were fitted on is in-sample; a hit
        # there proves the chain works, not that the system has foresight.
        boundary = (
            pd.Timestamp(cfg.folds[-1].test_start, tz="UTC")
            - pd.Timedelta(cfg.embargo_days, "D")
        ).date().isoformat()

        return _clean({
            "rows": rows,
            "track_record": store.track_record(training_boundary=boundary),
            "runs": store.runs()[-30:][::-1],
        })

    @router.get("/verify")
    def verify(prediction_id: str = Query(...)) -> dict:
        """Recompute a payload's digest independently and compare with the ledger."""
        from bolt.chain.verify import compute_digest

        store = state.store()
        record = store.get_prediction(prediction_id)
        if record is None:
            raise HTTPException(404, "no such prediction in the ledger")

        path = cfg.path("predictions") / f"{prediction_id[:18]}.json"
        if not path.is_file():
            raise HTTPException(404, f"payload file missing: {path.name}")

        recomputed, document = compute_digest(path)
        matches = recomputed.lower() == record["digest"].lower()
        from bolt.web.api import _clean

        return _clean({
            "prediction_id": prediction_id,
            "payload_path": str(path),
            "recomputed_digest": recomputed,
            "ledger_digest": record["digest"],
            "match": matches,
            "committed": record.get("committed", False),
            "tx_hash": record.get("tx_hash"),
            "block_time": record.get("block_time"),
            "as_of_date": document.get("as_of_date"),
            "note": (
                "The published payload hashes to the recorded digest."
                if matches else
                "MISMATCH: the payload no longer hashes to the recorded digest. It has "
                "been altered since it was written."
            ),
            "chain_note": (
                "Committed on-chain; run `bolt verify` with an RPC endpoint to check "
                "the ledger entry against the chain itself."
                if record.get("committed") else
                "NOT committed on-chain. The digest is recorded locally only, which "
                "proves integrity but NOT timing - only the chain can prove the "
                "prediction existed before the outcome."
            ),
        })

    @router.get("/health")
    def health() -> dict:
        from bolt.agents.health import HealthAgent
        from bolt.commands import load_windows

        agent = HealthAgent()
        try:
            X, _, meta = load_windows(cfg)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(503, f"windows unavailable: {exc}")

        window_end = pd.to_datetime(meta["window_end"], utc=True)
        boundary = pd.Timestamp(cfg.folds[-1].test_start, tz="UTC")
        reference = (window_end < boundary).to_numpy()
        current = (window_end >= boundary).to_numpy()

        drift = []
        if reference.sum() > 200 and current.sum() > 30:
            drift = agent.feature_drift(X[reference], X[current], list(cfg.feature_columns))

        per_fold = state.table("model_comparison_per_fold.csv")
        performance = agent.performance_trend(per_fold) if per_fold is not None else None
        report = agent.analyse(
            drift=drift or None, performance=performance
        ) if (drift or performance) else None

        from bolt.web.api import _clean

        return _clean({
            "drift": [
                {"feature": d.feature, "psi": d.psi, "severity": d.severity}
                for d in drift
            ],
            "performance": performance,
            "report": _report_dict(report) if report else None,
            "reference_windows": int(reference.sum()),
            "current_windows": int(current.sum()),
        })

    return router


def _analogues(cfg: BoltConfig, state, asset: str, as_of: pd.Timestamp) -> list[dict]:
    """Nearest historical regimes, for the skeptic's precedent check."""
    try:
        from bolt.commands import load_windows
        from bolt.explain.analogue import nearest_historical_analogue
        from bolt.windows import flatten_windows

        X, _, meta = load_windows(cfg)
        mask = (meta["asset"] == asset).to_numpy()
        if mask.sum() < 10:
            return []
        flat = flatten_windows(X[mask])
        sub_meta = meta[mask].reset_index(drop=True)
        ends = pd.to_datetime(sub_meta["window_end"], utc=True)
        eligible = np.flatnonzero((ends <= as_of).to_numpy())
        if eligible.size == 0:
            return []

        settings = cfg.section("explain")["analogue"]
        return nearest_historical_analogue(
            flat[int(eligible[-1])], flat, sub_meta,
            prices=state.panel()["close"], k=int(settings["k"]),
            exclude_days=int(settings["exclude_days"]),
            metric=str(settings["metric"]), as_of=as_of,
            outcome_window_days=int(settings["outcome_window_days"]),
        )
    except Exception as exc:  # noqa: BLE001 - an analogue failure must not kill the run
        log.warning("analogue retrieval failed: %s", exc)
        return []


def _hindsight(cfg: BoltConfig, panel: pd.DataFrame, asset: str,
               as_of: pd.Timestamp) -> dict:
    """What actually happened after ``as_of``, when the horizon has closed.

    Shown ONLY as hindsight, clearly separated from the prediction. The system
    never sees this at prediction time; it exists so a viewer can score the call
    themselves rather than take the console's word for it.
    """
    try:
        prices = panel.xs(asset, level="asset")["close"].sort_index()
    except KeyError:
        return {"available": False, "reason": f"no prices for {asset}"}

    horizon = int(cfg.horizon_days)
    window = prices.loc[as_of: as_of + pd.Timedelta(horizon, "D")]
    if len(window) < 2:
        return {
            "available": False,
            "reason": f"the {horizon}-day horizon has not closed in the frozen dataset",
        }

    start = float(window.iloc[0])
    drawdown = (start - float(window.min())) / start
    from bolt.web.api import _clean

    return _clean({
        "available": True,
        "horizon_days": horizon,
        "start_price": start,
        "min_price": float(window.min()),
        "end_price": float(window.iloc[-1]),
        "max_drawdown": drawdown,
        "crashed": bool(drawdown >= cfg.drawdown_threshold),
        "threshold": cfg.drawdown_threshold,
    })
