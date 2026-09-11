"""web3 submit and read against PredictionRegistry - G5 (BOLT_SPEC.md Section 9).

Safety, restated because it matters more than anything else in this file:

* The private key is read from an environment variable ONLY. It is never written
  to a file, never logged, never included in an error message, and never
  committed.
* **Testnet only.** The configured network is Polygon Amoy. Use a throwaway
  account funded with test MATIC and nothing else.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from bolt.logging_setup import get_logger

log = get_logger(__name__)

ABI_PATH = Path("contracts/PredictionRegistry.abi.json")

#: Networks this client will talk to. Mainnet chain ids are deliberately absent:
#: an accidental mainnet commitment would spend real money for no benefit.
ALLOWED_CHAIN_IDS = {
    80002: "polygon_amoy",
    11155111: "sepolia",
    31337: "local_hardhat",
}


class ChainError(RuntimeError):
    """A chain operation failed. Never contains key material."""


@dataclass(frozen=True, slots=True)
class CommitReceipt:
    prediction_id: str
    digest: str
    tx_hash: str
    block_number: int
    block_time: int
    committer: str
    explorer_url: str | None = None


def load_abi() -> list:
    """Load the compiled ABI, or fall back to the minimal interface."""
    if ABI_PATH.is_file():
        return json.loads(ABI_PATH.read_text(encoding="utf-8"))
    from bolt.chain.verify import REGISTRY_ABI

    return REGISTRY_ABI + [{
        "inputs": [
            {"internalType": "bytes32", "name": "predictionId", "type": "bytes32"},
            {"internalType": "bytes32", "name": "digest", "type": "bytes32"},
        ],
        "name": "commit", "outputs": [], "stateMutability": "nonpayable", "type": "function",
    }]


class RegistryClient:
    """Thin wrapper over the PredictionRegistry contract."""

    def __init__(self, cfg, rpc_url: str | None = None, contract_address: str | None = None):
        self.cfg = cfg
        chain = cfg.section("chain")
        self.chain_id = int(chain["chain_id"])
        if self.chain_id not in ALLOWED_CHAIN_IDS:
            raise ChainError(
                f"chain id {self.chain_id} is not an allowed testnet. BOLT commits to "
                f"testnets only; allowed: {sorted(ALLOWED_CHAIN_IDS)}"
            )
        self.rpc_url = rpc_url or os.environ.get(chain["rpc_env_var"], "")
        if not self.rpc_url:
            raise ChainError(
                f"no RPC endpoint: set ${chain['rpc_env_var']} or pass --rpc-url"
            )
        self.contract_address = contract_address or self._read_address(chain)
        self.gas_limit = int(chain["gas_limit"])
        self._web3 = None
        self._contract = None

    @staticmethod
    def _read_address(chain: dict) -> str:
        path = Path(chain["contract_address_file"])
        if not path.is_file():
            raise ChainError(
                f"no deployed contract address at {path}. Run `python contracts/deploy.py` "
                f"first, or pass --contract."
            )
        return path.read_text(encoding="utf-8").strip()

    @property
    def web3(self):
        if self._web3 is None:
            try:
                from web3 import Web3
            except ImportError as exc:  # pragma: no cover
                raise ChainError("web3 is required: pip install web3") from exc
            self._web3 = Web3(Web3.HTTPProvider(self.rpc_url, request_kwargs={"timeout": 30}))
            if not self._web3.is_connected():
                raise ChainError(f"could not connect to the RPC endpoint")
            actual = self._web3.eth.chain_id
            if actual != self.chain_id:
                raise ChainError(
                    f"connected to chain {actual} but config expects {self.chain_id} "
                    f"({ALLOWED_CHAIN_IDS[self.chain_id]}). Refusing to commit to the "
                    f"wrong network."
                )
        return self._web3

    @property
    def contract(self):
        if self._contract is None:
            from web3 import Web3
            self._contract = self.web3.eth.contract(
                address=Web3.to_checksum_address(self.contract_address), abi=load_abi()
            )
        return self._contract

    def _account(self):
        """Load the signing account from the environment. Never logged."""
        from eth_account import Account

        variable = self.cfg.section("chain")["private_key_env_var"]
        key = os.environ.get(variable, "").strip()
        if not key:
            raise ChainError(
                f"no signing key: set ${variable}. BOLT reads the key from the "
                f"environment only and never from a file. Use a THROWAWAY testnet key."
            )
        try:
            return Account.from_key(key)
        except (ValueError, TypeError) as exc:
            # Deliberately does not echo the value back.
            raise ChainError(f"${variable} is not a valid private key") from exc

    def exists(self, prediction_id: str) -> bool:
        identifier = bytes.fromhex(prediction_id.removeprefix("0x"))
        return bool(self.contract.functions.exists(identifier).call())

    def get(self, prediction_id: str) -> tuple[str, int, str]:
        identifier = bytes.fromhex(prediction_id.removeprefix("0x"))
        digest, block_time, committer = self.contract.functions.get(identifier).call()
        return "0x" + digest.hex(), int(block_time), str(committer)

    def commit(self, prediction_id: str, digest: str) -> CommitReceipt:
        """Submit a commitment and wait for confirmation.

        Raises:
            ChainError: if the id was already committed. That revert is the whole
                point of G5 and is surfaced, never swallowed.
        """
        account = self._account()
        identifier = bytes.fromhex(prediction_id.removeprefix("0x"))
        digest_bytes = bytes.fromhex(digest.removeprefix("0x"))

        if self.exists(prediction_id):
            raise ChainError(
                f"prediction {prediction_id[:18]}... is ALREADY committed. The registry "
                f"refuses overwrites by design: that immutability is what makes the "
                f"track record auditable."
            )

        transaction = self.contract.functions.commit(identifier, digest_bytes).build_transaction({
            "from": account.address,
            "nonce": self.web3.eth.get_transaction_count(account.address),
            "gas": self.gas_limit,
            "gasPrice": self.web3.eth.gas_price,
            "chainId": self.chain_id,
        })
        signed = account.sign_transaction(transaction)
        raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
        tx_hash = self.web3.eth.send_raw_transaction(raw)
        log.info("commit submitted: %s", tx_hash.hex())

        receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)
        if receipt["status"] != 1:
            raise ChainError(f"commit transaction reverted: {tx_hash.hex()}")

        block = self.web3.eth.get_block(receipt["blockNumber"])
        return CommitReceipt(
            prediction_id=prediction_id,
            digest=digest,
            tx_hash=tx_hash.hex(),
            block_number=int(receipt["blockNumber"]),
            block_time=int(block["timestamp"]),
            committer=account.address,
            explorer_url=self._explorer(tx_hash.hex()),
        )

    def _explorer(self, tx_hash: str) -> str | None:
        bases = {
            80002: "https://amoy.polygonscan.com/tx/",
            11155111: "https://sepolia.etherscan.io/tx/",
        }
        base = bases.get(self.chain_id)
        return f"{base}{tx_hash}" if base else None
