"""CLI surface (BOLT_SPEC.md Section 10, Phase 0 acceptance).

Phase 0 accepts when ``python -m bolt.cli --help`` prints all commands. These
tests pin that surface so a later refactor cannot quietly drop or rename one.
"""

from __future__ import annotations

import pytest

from bolt.cli import MODEL_CHOICES, build_parser, main

EXPECTED_COMMANDS = {
    "ingest", "build", "train", "evaluate", "explain", "predict", "verify",
    "monitor", "serve",
}


def _subparsers(parser):
    for action in parser._actions:  # noqa: SLF001 - argparse exposes no public API for this
        if hasattr(action, "choices") and action.dest == "command":
            return action.choices
    raise AssertionError("parser has no subcommands")


def test_all_spec_commands_exist():
    assert set(_subparsers(build_parser())) == EXPECTED_COMMANDS


def test_help_lists_every_command(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for command in EXPECTED_COMMANDS:
        assert command in out


@pytest.mark.parametrize("command", sorted(EXPECTED_COMMANDS))
def test_each_subcommand_has_help(command, capsys):
    with pytest.raises(SystemExit) as exc:
        main([command, "--help"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip()


def test_model_choices_cover_the_seven_benchmarked_models():
    # G3: LSTM, GRU, XGBoost, RF, LogReg, MLP and the volatility rule.
    assert set(MODEL_CHOICES) == {"lstm", "gru", "xgb", "rf", "logreg", "mlp", "rule", "all"}


def test_spec_invocations_parse():
    parser = build_parser()
    parser.parse_args(["ingest", "--config", "config/default.yaml"])
    parser.parse_args(["build"])
    parser.parse_args(["train", "--model", "lstm"])
    parser.parse_args(["evaluate", "--all", "--report"])
    parser.parse_args(["explain", "--model", "lstm", "--consistency"])
    parser.parse_args(["predict", "--asset", "BTC", "--as-of", "2025-11-01", "--commit"])
    parser.parse_args(["verify", "--payload", "outputs/predictions/x.json", "--id", "x"])


def test_no_command_is_an_error(capsys):
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


def test_pending_phases_exit_three_not_zero():
    """Rule 12.2/12.3: an unbuilt phase must fail loudly, never fake success.

    Every command is implemented as of Phase 5, so this exercises the error path
    directly rather than invoking a command. Running a real command here would
    hit the network and rewrite the frozen dataset from inside the test suite.
    """
    import argparse

    from bolt.cli import CommandNotReady, _pending

    with pytest.raises(CommandNotReady, match="Phase 7"):
        _pending("future-thing", "Phase 7")

    def explode(_args):
        _pending("future-thing", "Phase 7")

    parser = build_parser()
    args = parser.parse_args(["build"])
    args.func = explode
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("bolt.cli.build_parser", lambda: _parser_returning(args))
        assert main(["build"]) == 3


def _parser_returning(args):
    """A stand-in parser that yields fixed args, for exit-code testing."""
    class _Stub:
        def parse_args(self, _argv=None):
            return args
    return _Stub()


def test_every_spec_command_is_implemented():
    """No command may still be a placeholder at the 50% review."""
    import bolt.commands as commands

    for name in ("do_ingest", "do_build", "do_train", "do_evaluate",
                 "do_explain", "do_predict", "do_verify", "do_monitor", "do_serve"):
        assert hasattr(commands, name), f"bolt.commands.{name} is missing"


def test_bad_config_exits_two(tmp_path):
    bad = tmp_path / "broken.yaml"
    bad.write_text("project: {seed: 1}\n", encoding="utf-8")
    assert main(["ingest", "--config", str(bad)]) == 2


def test_verify_does_not_require_the_pipeline_config():
    # BOLT_SPEC.md Section 9: verification must stand alone.
    parser = build_parser()
    args = parser.parse_args(["verify", "--payload", "p.json", "--id", "abc"])
    assert not hasattr(args, "config")
