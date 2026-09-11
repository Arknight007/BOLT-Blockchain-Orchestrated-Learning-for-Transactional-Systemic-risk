"""PredictionRegistry behaviour - G5 (BOLT_SPEC.md Section 9).

The whole G5 claim rests on one line in the contract: the revert in ``commit``
when a prediction id already exists. Without it, a prediction could be silently
replaced once the outcome was known and the ledger would prove nothing at all.

These tests compile the real Solidity source and execute it against an in-memory
EVM, so they check the deployed bytecode's behaviour rather than a Python model
of it. They skip (never silently pass) when no compiler is available.
"""

from __future__ import annotations

from pathlib import Path

import pytest

CONTRACT = Path("contracts/PredictionRegistry.sol")
SOLC_VERSION = "0.8.20"

ID_A = bytes.fromhex("11" * 32)
ID_B = bytes.fromhex("22" * 32)
DIGEST_A = bytes.fromhex("aa" * 32)
DIGEST_B = bytes.fromhex("bb" * 32)


@pytest.fixture(scope="module")
def deployed():
    """Compile and deploy the registry to an in-memory EVM."""
    solcx = pytest.importorskip("solcx", reason="py-solc-x not installed")
    eth_tester = pytest.importorskip("eth_tester", reason="eth-tester not installed")
    from web3 import EthereumTesterProvider, Web3

    try:
        installed = [str(v) for v in solcx.get_installed_solc_versions()]
        if SOLC_VERSION not in installed:
            # solc is downloaded over TLS; on networks with an inspecting proxy
            # certifi rejects the chain, so use the OS trust store here too.
            try:
                import truststore

                truststore.inject_into_ssl()
            except ImportError:
                pass
            solcx.install_solc(SOLC_VERSION)
    except Exception as exc:  # pragma: no cover - network dependent
        pytest.skip(f"could not obtain solc {SOLC_VERSION}: {exc}")

    compiled = solcx.compile_source(
        CONTRACT.read_text(encoding="utf-8"),
        output_values=["abi", "bin"],
        solc_version=SOLC_VERSION,
    )
    _, interface = next(
        (k, v) for k, v in compiled.items() if k.endswith("PredictionRegistry")
    )

    web3 = Web3(EthereumTesterProvider())
    account = web3.eth.accounts[0]
    contract = web3.eth.contract(abi=interface["abi"], bytecode=interface["bin"])
    tx_hash = contract.constructor().transact({"from": account})
    receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
    instance = web3.eth.contract(address=receipt["contractAddress"], abi=interface["abi"])
    return web3, instance, account


def test_commit_stores_the_digest(deployed):
    web3, registry, account = deployed
    registry.functions.commit(ID_A, DIGEST_A).transact({"from": account})

    stored_digest, block_time, committer = registry.functions.get(ID_A).call()
    assert stored_digest == DIGEST_A
    assert block_time > 0, "the chain must timestamp the commitment"
    assert committer == account


def test_commit_emits_an_event(deployed):
    web3, registry, account = deployed
    tx_hash = registry.functions.commit(ID_B, DIGEST_B).transact({"from": account})
    receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
    events = registry.events.Committed().process_receipt(receipt)
    assert len(events) == 1
    assert events[0]["args"]["digest"] == DIGEST_B


def test_overwriting_a_commitment_reverts(deployed):
    """THE test for G5.

    If this passes silently one day because the revert was removed, the entire
    auditability claim collapses: a prediction could be rewritten after its
    outcome was known and the ledger would still look clean.
    """
    web3, registry, account = deployed
    identifier = bytes.fromhex("33" * 32)
    registry.functions.commit(identifier, DIGEST_A).transact({"from": account})

    with pytest.raises(Exception) as exc:
        registry.functions.commit(identifier, DIGEST_B).transact({"from": account})
    assert "revert" in str(exc.value).lower() or "AlreadyCommitted" in str(exc.value)

    # And the original value must be untouched.
    stored_digest, _, _ = registry.functions.get(identifier).call()
    assert stored_digest == DIGEST_A, "the original commitment must survive the attempt"


def test_a_different_committer_also_cannot_overwrite(deployed):
    """Immutability is not merely per-account: nobody can overwrite."""
    web3, registry, account = deployed
    other = web3.eth.accounts[1]
    identifier = bytes.fromhex("44" * 32)
    registry.functions.commit(identifier, DIGEST_A).transact({"from": account})

    with pytest.raises(Exception):
        registry.functions.commit(identifier, DIGEST_B).transact({"from": other})


def test_empty_digest_is_refused(deployed):
    _, registry, account = deployed
    with pytest.raises(Exception):
        registry.functions.commit(
            bytes.fromhex("55" * 32), bytes(32)
        ).transact({"from": account})


def test_uncommitted_id_reads_as_zero(deployed):
    _, registry, _ = deployed
    stored_digest, block_time, committer = registry.functions.get(
        bytes.fromhex("99" * 32)
    ).call()
    assert stored_digest == bytes(32)
    assert block_time == 0
    assert int(committer, 16) == 0


def test_exists_reflects_commitment_state(deployed):
    _, registry, account = deployed
    identifier = bytes.fromhex("66" * 32)
    assert registry.functions.exists(identifier).call() is False
    registry.functions.commit(identifier, DIGEST_A).transact({"from": account})
    assert registry.functions.exists(identifier).call() is True


def test_total_commitments_increments(deployed):
    _, registry, account = deployed
    before = registry.functions.totalCommitments().call()
    registry.functions.commit(bytes.fromhex("77" * 32), DIGEST_A).transact({"from": account})
    assert registry.functions.totalCommitments().call() == before + 1


# ---------------------------------------------------------------------------
# Source-level guarantees, checkable without a compiler
# ---------------------------------------------------------------------------

def test_contract_is_under_sixty_lines_of_code():
    """Section 9: keep it minimal. A small contract is an auditable contract."""
    lines = [
        line for line in CONTRACT.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith(("//", "/*", "*"))
    ]
    assert len(lines) < 60, f"contract has {len(lines)} code lines; the spec caps it at 60"


def test_contract_contains_the_overwrite_revert():
    source = CONTRACT.read_text(encoding="utf-8")
    assert "AlreadyCommitted" in source
    assert "revert AlreadyCommitted" in source, (
        "the overwrite revert is the property the entire G5 claim rests on"
    )


def test_contract_has_no_owner_or_upgrade_path():
    """An upgradeable or owner-controlled registry would not be immutable."""
    source = CONTRACT.read_text(encoding="utf-8")
    for banned in ("selfdestruct", "delegatecall", "onlyOwner", "upgradeTo", "initialize("):
        assert banned not in source, (
            f"{banned} would let the registry's history be altered, defeating G5"
        )
