"""Command-line entrypoints (BOLT_SPEC.md Section 10).

    bolt ingest   --config config/default.yaml
    bolt build
    bolt train    --model lstm|gru|xgb|rf|logreg|mlp|rule|all
    bolt evaluate --all --report
    bolt explain  --model lstm --consistency
    bolt predict  --asset BTC --as-of 2025-11-01 --commit
    bolt verify   --payload <path> --id <prediction_id>

Every command loads and validates the configuration before doing anything else,
so a run that would violate a leakage guard fails at second zero rather than
after an hour of ingestion.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Sequence

from bolt.config import ConfigError, load_config
from bolt.logging_setup import get_logger, setup_logging
from bolt.version import __version__

log = get_logger(__name__)

MODEL_CHOICES = ("lstm", "gru", "xgb", "rf", "logreg", "mlp", "rule", "all")
IMBALANCE_CHOICES = ("class_weight", "smote", "none")


class CommandNotReady(NotImplementedError):
    """Raised by a command whose implementing phase has not landed yet."""


def _add_config_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/default.yaml"),
        metavar="PATH",
        help="path to the YAML configuration (default: config/default.yaml)",
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the full argument parser. Kept separate so tests can introspect it."""
    parser = argparse.ArgumentParser(
        prog="bolt",
        description=(
            "BOLT - cryptocurrency crash early warning with auditable, "
            "on-chain-committed predictions."
        ),
        epilog="Full build specification: BOLT_SPEC.md",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"bolt {__version__}")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="set log level to DEBUG"
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)

    p_ingest = sub.add_parser("ingest", help="fetch and cache raw market, on-chain and news data")
    _add_config_flag(p_ingest)
    p_ingest.add_argument("--source", choices=("market", "onchain", "news", "all"), default="all")
    p_ingest.add_argument("--refresh", action="store_true", help="ignore the disk cache and re-fetch")
    p_ingest.set_defaults(func=cmd_ingest)

    p_build = sub.add_parser("build", help="align the panel, build features and labels, freeze the dataset")
    _add_config_flag(p_build)
    p_build.add_argument("--skip-windows", action="store_true", help="stop after the feature panel")
    p_build.set_defaults(func=cmd_build)

    p_train = sub.add_parser("train", help="fit one model or every model on the walk-forward folds")
    _add_config_flag(p_train)
    p_train.add_argument("--model", choices=MODEL_CHOICES, default="all")
    p_train.add_argument("--imbalance", choices=IMBALANCE_CHOICES, default=None,
                         help="override imbalance.method for an ablation run")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate", help="score models and write the comparison table and figures")
    _add_config_flag(p_eval)
    p_eval.add_argument("--all", action="store_true", help="evaluate every model")
    p_eval.add_argument("--model", choices=MODEL_CHOICES, default=None)
    p_eval.add_argument("--report", action="store_true", help="write tables and figures to outputs/")
    p_eval.set_defaults(func=cmd_evaluate)

    p_explain = sub.add_parser("explain", help="attribution, cross-episode consistency (G4), analogues")
    _add_config_flag(p_explain)
    p_explain.add_argument("--model", choices=MODEL_CHOICES[:-1], default="lstm")
    p_explain.add_argument("--consistency", action="store_true",
                           help="run the cross-episode attribution consistency analysis (G4)")
    p_explain.add_argument("--analogue", action="store_true", help="retrieve nearest historical analogues")
    p_explain.set_defaults(func=cmd_explain)

    p_predict = sub.add_parser("predict", help="score one asset as of a date, optionally committing on-chain")
    _add_config_flag(p_predict)
    p_predict.add_argument("--asset", required=True, help="asset symbol, e.g. BTC")
    p_predict.add_argument("--as-of", required=True, metavar="YYYY-MM-DD",
                           help="the as-of date; only data up to this date is used")
    p_predict.add_argument("--model", choices=MODEL_CHOICES[:-1], default="lstm")
    p_predict.add_argument("--commit", action="store_true",
                           help="commit the prediction digest on-chain BEFORE the outcome is known (G5)")
    p_predict.set_defaults(func=cmd_predict)

    p_verify = sub.add_parser("verify", help="independently verify a committed prediction (G5)")
    p_verify.add_argument("--payload", type=Path, required=True, help="path to the published payload JSON")
    p_verify.add_argument("--id", dest="prediction_id", required=True, help="the prediction id")
    p_verify.add_argument("--rpc-url", default=None, help="override the RPC endpoint")
    p_verify.add_argument("--contract", default=None, help="override the registry contract address")
    p_verify.add_argument("--offline", action="store_true",
                          help="recompute the digest without reading the chain")
    p_verify.set_defaults(func=cmd_verify)

    return parser


# ---------------------------------------------------------------------------
# Commands
#
# Phase 0 wires the surface and the config contract. Each handler below is
# replaced by its real implementation in the phase named in its message; none of
# them returns placeholder output in the meantime (Rule 12.2).
# ---------------------------------------------------------------------------

def _load(args: argparse.Namespace):
    cfg = load_config(args.config)
    log.info("configuration loaded from %s", cfg.config_path)
    log.info(
        "universe: %d assets (%d labelled targets), %d features, %d walk-forward folds",
        len(cfg.assets), len(cfg.target_assets), len(cfg.feature_columns), len(cfg.folds),
    )
    return cfg


def _pending(command: str, phase: str, gap: str | None = None) -> None:
    """Raised by any command whose implementing phase has not landed."""
    detail = f" ({gap})" if gap else ""
    raise CommandNotReady(
        f"`bolt {command}`{detail} lands in {phase}. See BOLT_SPEC.md Section 11 for "
        f"the build order and that phase's acceptance criteria."
    )


def cmd_ingest(args: argparse.Namespace) -> int:
    from bolt.commands import do_ingest

    return do_ingest(_load(args), refresh=args.refresh)


def cmd_build(args: argparse.Namespace) -> int:
    from bolt.commands import do_build

    return do_build(_load(args), skip_windows=args.skip_windows)


def cmd_train(args: argparse.Namespace) -> int:
    from bolt.commands import do_train

    return do_train(_load(args), model=args.model)


def cmd_evaluate(args: argparse.Namespace) -> int:
    from bolt.commands import do_evaluate

    if not args.all and args.model is None:
        raise SystemExit("bolt evaluate: pass --all or --model MODEL")
    models = None if args.all else [args.model]
    return do_evaluate(_load(args), models, report=args.report)


def cmd_explain(args: argparse.Namespace) -> int:
    from bolt.commands import do_explain

    return do_explain(
        _load(args), model=args.model,
        consistency=args.consistency or not args.analogue,
        analogue=args.analogue,
    )


def cmd_predict(args: argparse.Namespace) -> int:
    from bolt.commands import do_predict

    return do_predict(
        _load(args), asset=args.asset, as_of=args.as_of,
        model=args.model, commit=args.commit,
    )


def cmd_verify(args: argparse.Namespace) -> int:
    # Deliberately does NOT load the pipeline configuration: verification must
    # stand alone (BOLT_SPEC.md Section 9).
    from bolt.commands import do_verify

    return do_verify(
        str(args.payload), args.prediction_id, args.rpc_url, args.contract,
        offline=args.offline,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Entrypoint. Returns a process exit code; never raises for user error."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        cfg_path = getattr(args, "config", None)
        settings = {"level": "DEBUG" if args.verbose else "INFO", "file": "outputs/bolt.log"}
        if cfg_path is not None and Path(cfg_path).is_file():
            try:
                settings = dict(load_config(cfg_path).section("logging"))
            except ConfigError:
                pass  # surfaced properly by the command handler below
            if args.verbose:
                settings["level"] = "DEBUG"
        setup_logging(settings, force=True)
    except OSError as exc:
        print(f"bolt: could not initialise logging: {exc}", file=sys.stderr)
        return 2

    handler: Callable[[argparse.Namespace], int] = args.func
    try:
        return handler(args)
    except ConfigError as exc:
        log.error("configuration error: %s", exc)
        return 2
    except CommandNotReady as exc:
        log.error("%s", exc)
        return 3
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 5
    except KeyboardInterrupt:
        log.warning("interrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
