"""CLI tests."""

from __future__ import annotations

import pytest

from neuroforge.cli import build_parser, main


def test_cli_info_returns_zero():
    assert main(["info"]) == 0


def test_cli_no_command_prints_help():
    # No subcommand -> prints help, exit 0.
    assert main([]) == 0


def test_build_parser_has_subcommands():
    parser = build_parser()
    # Parsing 'info' resolves the handler.
    args = parser.parse_args(["info"])
    assert callable(args.func)


@pytest.mark.slow
def test_cli_demo_slow(tmp_path):
    """End-to-end demo via the CLI (trains a tiny model; slow)."""
    import os

    report = os.path.join(tmp_path, "demo.html")
    rc = main(["demo", "--report", report])
    assert rc == 0
    assert os.path.exists(report)
