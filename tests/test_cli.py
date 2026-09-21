"""CLI behaviour: analyze offline, JSON summary, show, and usage errors."""

from __future__ import annotations

import json

import pytest

from datajev.cli import main

GOAL = "How are alpha and beta related over time?"


def _analyze_args(tiny_csv, tmp_path, extra=None):
    args = [
        "analyze",
        tiny_csv,
        "--goal",
        GOAL,
        "--controller",
        "heuristic",
        "--analyst",
        "scripted",
        "--max-steps",
        "3",
        "--out",
        str(tmp_path / "runs"),
        "--no-color",
    ]
    return args + (extra or [])


def test_cli_analyze_json_summary(tiny_csv, tmp_path, capsys):
    code = main(_analyze_args(tiny_csv, tmp_path, ["--json"]))
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["controller"] == "heuristic"
    assert payload["analyst"] == "scripted"
    assert payload["steps"] >= 1
    assert payload["retained_insights"] or payload["final_answer"]

    from pathlib import Path

    assert Path(payload["trace_path"]).exists()
    assert Path(payload["run_dir"]).exists()


def test_cli_analyze_human_output(tiny_csv, tmp_path, capsys):
    code = main(_analyze_args(tiny_csv, tmp_path))
    captured = capsys.readouterr().out
    assert code == 0
    assert "DataJev" in captured
    assert "GOAL" in captured
    assert "NEXT ACTION" in captured
    assert "FINAL ANSWER" in captured
    assert "trace" in captured


def test_cli_quiet_only_prints_answer(tiny_csv, tmp_path, capsys):
    code = main(_analyze_args(tiny_csv, tmp_path, ["--quiet"]))
    captured = capsys.readouterr().out
    assert code == 0
    assert "# Final answer" in captured
    assert "NEXT ACTION" not in captured


def test_cli_show_replays_a_saved_trace(tiny_csv, tmp_path, capsys):
    assert main(_analyze_args(tiny_csv, tmp_path, ["--json"])) == 0
    run_dir = json.loads(capsys.readouterr().out)["run_dir"]

    assert main(["show", run_dir, "--no-color"]) == 0
    out = capsys.readouterr().out
    assert "goal" in out
    assert "RETAINED INSIGHTS" in out
    assert "FINAL ANSWER" in out

    assert main(["show", run_dir, "--steps", "--no-color"]) == 0
    detailed = capsys.readouterr().out
    assert "action probabilities" in detailed
    assert "code:" in detailed


def test_cli_missing_csv_is_a_usage_error(tmp_path, capsys):
    code = main(
        [
            "analyze",
            str(tmp_path / "nope.csv"),
            "--goal",
            "g",
            "--controller",
            "heuristic",
            "--analyst",
            "scripted",
            "--no-color",
        ]
    )
    assert code == 2
    assert "CSV not found" in capsys.readouterr().err


def test_cli_jev_without_key_errors(monkeypatch, tiny_csv, tmp_path, capsys):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    code = main(_analyze_args(tiny_csv, tmp_path, ["--controller", "jev"]))
    assert code == 1
    assert "TYPESAFE_API_KEY" in capsys.readouterr().err


def test_cli_no_command_prints_help(capsys):
    assert main([]) == 2
    assert "usage" in capsys.readouterr().out.lower()


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert "datajev" in capsys.readouterr().out
