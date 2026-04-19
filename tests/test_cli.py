"""Tests for the Typer CLI entry-point."""

from typer.testing import CliRunner

from f1_commentary.cli import app

runner = CliRunner()


def test_cli_help() -> None:
    """CLI --help should exit cleanly and show the app name."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "F1 Commentary Generation System" in result.output


def test_replay_stub() -> None:
    """The replay command should print a Phase 1 stub message."""
    result = runner.invoke(app, ["replay"])
    assert result.exit_code == 0
    assert "Phase 1" in result.output


def test_replay_log_help() -> None:
    """The replay-log sub-app should show help with available commands."""
    result = runner.invoke(app, ["replay-log", "--help"])
    assert result.exit_code == 0
    assert "summary" in result.output
    assert "rebuild-test" in result.output


def test_build_index_no_chunks() -> None:
    """The build-index command should exit with error when no chunks file exists."""
    result = runner.invoke(app, ["build-index"])
    assert result.exit_code == 1


def test_evaluate_run_no_traces() -> None:
    """The evaluate-run command should error when no traces file exists."""
    result = runner.invoke(app, ["evaluate-run", "--traces", "/nonexistent/traces.jsonl"])
    assert result.exit_code != 0


def test_generate_artifacts_no_comparison() -> None:
    """The generate-artifacts command should error when no comparison file exists."""
    result = runner.invoke(app, ["generate-artifacts", "--comparison", "/nonexistent/comparison.json"])
    assert result.exit_code != 0
