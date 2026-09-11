"""Compile and deploy PredictionRegistry - G5 (BOLT_SPEC.md Section 9).

*** TESTNET ONLY. ***

This script deploys to Polygon Amoy. It refuses any chain id that is not on the
allow-list in ``bolt.chain.client``, so an accidental mainnet deployment is not
possible through this path. Use a THROWAWAY account funded only with Amoy test
MATIC; never a key that controls anything of value.

The private key is read from the environment and nowhere else. It is never
written to a file, never logged, and never echoed in an error message.

    export BOLT_RPC_URL=https://rpc-amoy.polygon.technology
    export BOLT_PRIVATE_KEY=0x...          # throwaway testnet key
    python contracts/deploy.py

Writes the deployed address to ``outputs/contract_address.txt`` and the ABI to
``contracts/PredictionRegistry.abi.json``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPO_ROOT / "contracts" / "PredictionRegistry.sol"
ABI_OUT = REPO_ROOT / "contracts" / "PredictionRegistry.abi.json"
ADDRESS_OUT = REPO_ROOT / "outputs" / "contract_address.txt"
SOLC_VERSION = "0.8.20"

ALLOWED_CHAIN_IDS = {80002: "polygon_amoy", 11155111: "sepolia", 31337: "local"}


def compile_contract() -> tuple[list, str]:
    """Compile the registry. Returns (abi, bytecode)."""
    import solcx

    try:
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        pass

    installed = [str(v) for v in solcx.get_installed_solc_versions()]
    if SOLC_VERSION not in installed:
        print(f"installing solc {SOLC_VERSION}...")
        solcx.install_solc(SOLC_VERSION)

    compiled = solcx.compile_source(
        SOURCE.read_text(encoding="utf-8"),
        output_values=["abi", "bin"],
        solc_version=SOLC_VERSION,
    )
    _, interface = next(
        (k, v) for k, v in compiled.items() if k.endswith("PredictionRegistry")
    )
    return interface["abi"], interface["bin"]


def main() -> int:
    from eth_account import Account
    from web3 import Web3

    rpc_url = os.environ.get("BOLT_RPC_URL", "").strip()
    private_key = os.environ.get("BOLT_PRIVATE_KEY", "").strip()

    if not rpc_url:
        print("error: set BOLT_RPC_URL (e.g. https://rpc-amoy.polygon.technology)",
              file=sys.stderr)
        return 2
    if not private_key:
        print("error: set BOLT_PRIVATE_KEY. Use a THROWAWAY testnet key funded only "
              "with Amoy test MATIC.", file=sys.stderr)
        return 2

    web3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 60}))
    if not web3.is_connected():
        print(f"error: could not connect to the RPC endpoint", file=sys.stderr)
        return 3

    chain_id = web3.eth.chain_id
    if chain_id not in ALLOWED_CHAIN_IDS:
        print(f"REFUSING to deploy: chain id {chain_id} is not an allowed testnet "
              f"({sorted(ALLOWED_CHAIN_IDS)}). ChainGuard commits to testnets only.",
              file=sys.stderr)
        return 4

    account = Account.from_key(private_key)
    balance = web3.eth.get_balance(account.address)
    print(f"network  : {ALLOWED_CHAIN_IDS[chain_id]} (chain id {chain_id})")
    print(f"deployer : {account.address}")
    print(f"balance  : {web3.from_wei(balance, 'ether')} MATIC")
    if balance == 0:
        print("error: the deployer has no test MATIC. Fund it from a Polygon Amoy "
              "faucet and retry.", file=sys.stderr)
        return 5

    print("compiling...")
    abi, bytecode = compile_contract()

    contract = web3.eth.contract(abi=abi, bytecode=bytecode)
    transaction = contract.constructor().build_transaction({
        "from": account.address,
        "nonce": web3.eth.get_transaction_count(account.address),
        "gas": 600_000,
        "gasPrice": web3.eth.gas_price,
        "chainId": chain_id,
    })
    signed = account.sign_transaction(transaction)
    raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction

    print("deploying...")
    tx_hash = web3.eth.send_raw_transaction(raw)
    receipt = web3.eth.wait_for_transaction_receipt(tx_hash, timeout=300)
    if receipt["status"] != 1:
        print(f"error: deployment reverted ({tx_hash.hex()})", file=sys.stderr)
        return 6

    address = receipt["contractAddress"]
    ADDRESS_OUT.parent.mkdir(parents=True, exist_ok=True)
    ADDRESS_OUT.write_text(address + "\n", encoding="utf-8")
    ABI_OUT.write_text(json.dumps(abi, indent=2), encoding="utf-8")

    print()
    print(f"deployed : {address}")
    print(f"tx       : {tx_hash.hex()}")
    print(f"block    : {receipt['blockNumber']}")
    print(f"address written to {ADDRESS_OUT.relative_to(REPO_ROOT)}")
    if chain_id == 80002:
        print(f"explorer : https://amoy.polygonscan.com/address/{address}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
