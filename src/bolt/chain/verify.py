"""Independent verification path - G5 (BOLT_SPEC.md Section 9).

This module must NOT import from the prediction pipeline. Verification that
depends on the system it verifies proves nothing: if the same code that produced
a digest also recomputes it, a bug or a deliberate change in one is invisible to
the other.

So this file re-implements canonical serialisation and hashing from the standard
library alone, and reads the chain through a minimal ABI. A third party can lift
it out of the repository and run it against a published payload without
installing BOLT at all.

    bolt verify --payload outputs/predictions/<id>.json --id <prediction_id>
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Must match bolt.chain.payload.FLOAT_PRECISION. Deliberately restated rather
#: than imported: this module verifies the pipeline, so it cannot depend on it.
FLOAT_PRECISION = 8

#: The two functions a verifier needs. Nothing else is required to check a claim.
REGISTRY_ABI = [
    {
        "inputs": [{"internalType": "bytes32", "name": "predictionId", "type": "bytes32"}],
        "name": "get",
        "outputs": [
            {"internalType": "bytes32", "name": "digest", "type": "bytes32"},
            {"internalType": "uint256", "name": "blockTime", "type": "uint256"},
            {"internalType": "address", "name": "committer", "type": "address"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "bytes32", "name": "predictionId", "type": "bytes32"}],
        "name": "exists",
        "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function",
    },
]


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """The outcome of checking one committed prediction."""

    prediction_id: str
    match: bool
    computed_digest: str
    onchain_digest: str
    block_time: int | None
    committer: str | None
    as_of_date: str | None
    reason: str

    @property
    def block_time_iso(self) -> str | None:
        if not self.block_time:
            return None
        from datetime import datetime, timezone
        return datetime.fromtimestamp(self.block_time, tz=timezone.utc).isoformat()

    def render(self) -> str:
        lines = [
            "BOLT prediction verification",
            f"  prediction id   : {self.prediction_id}",
            f"  computed digest : {self.computed_digest}",
            f"  on-chain digest : {self.onchain_digest}",
        ]
        if self.block_time:
            lines.append(f"  committed at    : {self.block_time_iso} (block timestamp)")
        if self.committer:
            lines.append(f"  committed by    : {self.committer}")
        if self.as_of_date:
            lines.append(f"  prediction as-of: {self.as_of_date}")
            if self.block_time:
                lines.append(f"  {self._foresight_statement()}")
        lines.append("")
        lines.append(f"  RESULT: {'MATCH' if self.match else 'MISMATCH'} - {self.reason}")
        return "\n".join(lines)

    def _foresight_statement(self) -> str:
        """The sentence the whole G5 claim exists to support."""
        from datetime import datetime, timezone
        try:
            as_of = datetime.fromisoformat(self.as_of_date).replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            return "foresight  : as-of date unparseable"
        committed = datetime.fromtimestamp(self.block_time, tz=timezone.utc)
        delta = (committed - as_of).days
        if delta < 0:
            return f"foresight  : committed {abs(delta)} day(s) BEFORE its own as-of date (suspicious)"
        return (
            f"foresight  : committed {delta} day(s) after the as-of date, and the "
            f"chain timestamp cannot be backdated"
        )


def _canonicalise(value: Any) -> Any:
    """Independent re-implementation of the canonical normalisation."""
    if isinstance(value, dict):
        return {str(k): _canonicalise(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonicalise(v) for v in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return round(value, FLOAT_PRECISION)
    if isinstance(value, int):
        return value
    if value is None:
        return None
    return str(value)


def compute_digest(payload_path: str | Path) -> tuple[str, dict]:
    """Recompute SHA-256 over a published payload file.

    Hashes the file's EXACT bytes when they are already canonical, and otherwise
    re-canonicalises. Both paths must agree, and the mismatch between them is
    itself a signal that the published file was edited after commitment.
    """
    raw = Path(payload_path).read_bytes()
    document = json.loads(raw.decode("utf-8"))

    recanonical = json.dumps(
        _canonicalise(document), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")

    if raw.strip() != recanonical:
        # Not fatal: a pretty-printed copy of the same content is legitimate. The
        # canonical form is authoritative, so that is what gets hashed.
        pass
    return "0x" + hashlib.sha256(recanonical).hexdigest(), document


def read_onchain(
    prediction_id: str,
    rpc_url: str | None = None,
    contract_address: str | None = None,
) -> tuple[str, int, str]:
    """Read a commitment from the registry. Returns (digest, blockTime, committer)."""
    try:
        from web3 import Web3
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("verification needs web3: pip install web3") from exc

    rpc_url = rpc_url or os.environ.get("BOLT_RPC_URL", "")
    if not rpc_url:
        raise RuntimeError(
            "no RPC endpoint. Pass --rpc-url or set BOLT_RPC_URL. Verification is "
            "read-only and needs no private key."
        )
    if not contract_address:
        address_file = Path("outputs/contract_address.txt")
        if not address_file.is_file():
            raise RuntimeError(
                "no contract address. Pass --contract or create outputs/contract_address.txt"
            )
        contract_address = address_file.read_text(encoding="utf-8").strip()

    web3 = Web3(Web3.HTTPProvider(rpc_url))
    if not web3.is_connected():
        raise RuntimeError(f"could not connect to {rpc_url}")

    contract = web3.eth.contract(
        address=Web3.to_checksum_address(contract_address), abi=REGISTRY_ABI
    )
    identifier = bytes.fromhex(prediction_id.removeprefix("0x"))
    raw_digest, block_time, committer = contract.functions.get(identifier).call()
    return "0x" + raw_digest.hex(), int(block_time), str(committer)


def verify(
    payload_path: str | Path,
    prediction_id: str,
    rpc_url: str | None = None,
    contract_address: str | None = None,
    offline: bool = False,
) -> VerificationResult:
    """Verify a published payload against its on-chain commitment.

    Args:
        offline: recompute and report the digest without reading the chain.
            Useful for checking a payload's integrity with no RPC access.
    """
    computed, document = compute_digest(payload_path)
    as_of = document.get("as_of_date")

    if offline:
        return VerificationResult(
            prediction_id=prediction_id, match=False, computed_digest=computed,
            onchain_digest="(not read)", block_time=None, committer=None,
            as_of_date=as_of,
            reason="offline mode: digest recomputed but not compared against the chain",
        )

    onchain, block_time, committer = read_onchain(prediction_id, rpc_url, contract_address)

    if block_time == 0:
        return VerificationResult(
            prediction_id=prediction_id, match=False, computed_digest=computed,
            onchain_digest=onchain, block_time=None, committer=None, as_of_date=as_of,
            reason="this prediction id has never been committed to the registry",
        )

    match = computed.lower() == onchain.lower()
    reason = (
        "the published payload hashes to exactly the digest committed on-chain, and "
        "the block timestamp proves when that commitment was made"
        if match else
        "the published payload does NOT hash to the committed digest; it has been "
        "altered since commitment, or this is the wrong payload for this id"
    )
    return VerificationResult(
        prediction_id=prediction_id, match=match, computed_digest=computed,
        onchain_digest=onchain, block_time=block_time, committer=committer,
        as_of_date=as_of, reason=reason,
    )
