// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title PredictionRegistry
/// @notice Immutable, timestamped commitments to crash predictions (BOLT G5).
///
/// Every predictive paper in this field, the base paper included, asks the
/// reader to trust a backtest that could in principle have been tuned after the
/// fact. Nothing in a published record distinguishes genuine foresight from
/// hindsight fitting. This contract resolves that: a prediction's digest is
/// written to a public ledger BEFORE the outcome is known, and the chain
/// timestamps it.
///
/// The whole G5 claim rests on one line - the revert in `commit`. Without it a
/// prediction could be silently replaced once the outcome was known, and the
/// ledger would prove nothing at all. It has its own test.
///
/// Only the 32-byte digest is stored. The payload itself stays off-chain and is
/// published alongside; a verifier recomputes SHA-256 over the canonical payload
/// and compares. Storing raw market data on-chain would cost a fortune and add
/// no security.
contract PredictionRegistry {
    struct Commitment {
        bytes32 digest;
        uint256 blockTime;
        address committer;
    }

    mapping(bytes32 => Commitment) private _commitments;

    uint256 public totalCommitments;

    event Committed(bytes32 indexed predictionId, bytes32 digest, uint256 blockTime);

    error AlreadyCommitted(bytes32 predictionId);
    error EmptyDigest();

    /// @notice Record a prediction digest. Reverts if the id already exists.
    /// @param predictionId caller-chosen unique identifier for the prediction
    /// @param digest SHA-256 over the canonical payload JSON
    function commit(bytes32 predictionId, bytes32 digest) external {
        if (digest == bytes32(0)) revert EmptyDigest();
        if (_commitments[predictionId].blockTime != 0) revert AlreadyCommitted(predictionId);

        _commitments[predictionId] = Commitment(digest, block.timestamp, msg.sender);
        unchecked {
            totalCommitments += 1;
        }
        emit Committed(predictionId, digest, block.timestamp);
    }

    /// @notice Read a commitment. Returns zeroes when the id was never committed.
    function get(bytes32 predictionId)
        external
        view
        returns (bytes32 digest, uint256 blockTime, address committer)
    {
        Commitment memory c = _commitments[predictionId];
        return (c.digest, c.blockTime, c.committer);
    }

    /// @notice Whether a prediction id has been committed.
    function exists(bytes32 predictionId) external view returns (bool) {
        return _commitments[predictionId].blockTime != 0;
    }
}
