"""BOLT - blockchain-committed early warning for cryptocurrency crashes.

See BOLT_SPEC.md for the full build specification. The package is organised so
that each research gap maps onto one place in the tree:

    G1  economically meaningful target  -> bolt.labeling
    G2  cross-asset contagion structure -> bolt.features.contagion
    G3  a fair, complete model bench    -> bolt.models
    G4  attribution consistency         -> bolt.explain.consistency
    G5  on-chain prediction commitment  -> bolt.chain
"""

from bolt.config import BoltConfig, ConfigError, load_config
from bolt.logging_setup import get_logger, setup_logging
from bolt.version import __version__, git_commit_sha

__all__ = [
    "BoltConfig",
    "ConfigError",
    "load_config",
    "get_logger",
    "setup_logging",
    "git_commit_sha",
    "__version__",
]
