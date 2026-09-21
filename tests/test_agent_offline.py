"""End-to-end offline runs: no API keys, no network.

The scripted analyst writes real pandas code and the heuristic controller makes
the decisions, so this exercises the full loop: decision -> plan -> execute ->
interpret -> retain -> decide again -> final answer -> trace.
"""

from __future__ import annotations

import json

import pytest

from datajev.agent import (
    DataJevAgent,
    RunConfig,
    build_agent,
    resolve_analyst_name,
    resolve_controller_name,
)
from datajev.analyst.llm import AnalystError, LLMAnalyst
from datajev.analyst.scripted import ScriptedAnalyst
from datajev.controllers.base import Controller, ControllerError
from datajev.llm import resolve_llm_config
from datajev.state import AnalysisState
from datajev.types import ControllerDecision

GOAL = "How are alpha and beta related over time, and is one global description enough?"


def _config(tiny_csv: str, tmp_path, **overrides) -> RunConfig:
    values = {
        "csv_path": tiny_csv,
        "goal": GOAL,
        "out_dir": str(tmp_path / "runs"),
        "controller": "heuristic",
        "analyst": "scripted",
        "max_steps": 6,
    }
    values.update(overrides)
    return RunConfig(**values)


def test_offline_agent_runs_full_loop(tiny_csv, tmp_path):
    result = build_agent(_config(tiny_csv, tmp_path)).run()

    assert result.controller == "heuristic"
    assert result.analyst == "scripted"
    assert result.stop_reason == "controller_stop"
    assert result.steps >= 3

    trace = json.loads(result.trace_path.read_text())
    actions = [step["action"] for step in trace["steps"]]
    assert "continue" in actions
    assert "switch" in actions, "the loop must be able to change direction"
    assert "verify" in actions, "the loop must be able to verify a finding"

    # One decision per step, plus the final stop decision.
    assert len(trace["decisions"]) == result.steps + 1
    for decision in trace["decisions"]:
        assert decision["action_probabilities"]
        assert sum(decision["action_probabilities"].values()) == pytest.approx(1.0)
        assert decision["tool_probabilities"]
        assert 0.0 <= decision["retain_insight"] <= 1.0
        assert 0.0 <= decision["completeness"] <= 3.0

    # Every proposed insight carries the controller's retention judgment.
    assert trace["insights"], "the analysis produced no insights"
    assert all(i["retain_probability"] is not None for i in trace["insights"])
    assert any(i["retained"] for i in trace["insights"])
    assert trace["retention_fallback"] is False

    # Evidence was written to disk.
    assert result.steps >= 2
    artifacts = trace["artifacts"]
    assert any(a.endswith(".png") for a in artifacts)
    for artifact in artifacts:
        assert (result.run_dir / artifact).exists()

    assert (result.run_dir / "trace.json").exists()
    assert (result.run_dir / "final_answer.md").exists()
    assert "# Final answer" in result.final_answer.text
    assert "MEANGAM" not in result.final_answer.text  # tiny fixture uses alpha/beta


def test_offline_run_writes_reproducible_step_scripts(tiny_csv, tmp_path):
    result = build_agent(_config(tiny_csv, tmp_path, max_steps=3)).run()
    scripts = sorted((result.run_dir / "steps").glob("step_*.py"))
    assert len(scripts) == result.steps
    first = scripts[0].read_text()
    assert "DATAJEV_CSV" in first
    assert "pd.read_csv" in first
    # The analyst's code is appended after the preamble and is inspectable.
    assert "INSIGHT:" in first


def test_max_steps_is_respected_and_last_insights_are_kept(tiny_csv, tmp_path):
    result = build_agent(_config(tiny_csv, tmp_path, max_steps=2)).run()
    trace = json.loads(result.trace_path.read_text())
    assert result.steps == 2
    assert result.stop_reason == "max_steps"
    assert trace["retention_fallback"] is True

    first = [i for i in trace["insights"] if i["step_index"] == 1]
    last = [i for i in trace["insights"] if i["step_index"] == 2]
    assert first and last
    assert all(i["retain_probability"] is not None for i in first)
    assert all(i["retain_probability"] is None for i in last)
    assert all(i["retained"] for i in last)


def test_auto_resolution_prefers_jev_and_llm_when_keys_exist(monkeypatch, tiny_csv, tmp_path):
    monkeypatch.setenv("TYPESAFE_API_KEY", "x")
    monkeypatch.setenv("OPENAI_API_KEY", "y")
    monkeypatch.delenv("deepseek_api", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DATAJEV_LLM_API_KEY", raising=False)
    config = _config(tiny_csv, tmp_path, controller="auto", analyst="auto")
    assert resolve_controller_name(config) == "jev"
    assert resolve_analyst_name(config) == "llm"


def test_auto_resolution_falls_back_offline(monkeypatch, tiny_csv, tmp_path):
    for name in (
        "TYPESAFE_API_KEY",
        "OPENAI_API_KEY",
        "DATAJEV_LLM_API_KEY",
        "DEEPSEEK_API_KEY",
        "deepseek_api",
    ):
        monkeypatch.delenv(name, raising=False)
    config = _config(tiny_csv, tmp_path, controller="auto", analyst="auto")
    assert resolve_controller_name(config) == "heuristic"
    assert resolve_analyst_name(config) == "scripted"
    assert resolve_llm_config().available is False
    with pytest.raises(AnalystError):
        LLMAnalyst()


def test_scripted_analyst_code_is_schema_driven(tiny_csv, tmp_path):
    from datajev.dataset import profile_csv
    from datajev.types import ControllerDecision

    agent = build_agent(_config(tiny_csv, tmp_path))
    state = AnalysisState(goal=GOAL, dataset=profile_csv(tiny_csv))
    plan = agent.analyst.plan(
        state, ControllerDecision(source="heuristic", action="continue", tool="python")
    )
    assert "alpha" not in plan.code
    assert "MEANGAM" not in plan.code
    assert "select_dtypes" in plan.code
    assert "pd.read_csv" not in plan.code  # df is pre-loaded by the executor


# --------------------------------------------------------------------------- #
# Controller failure must be visible and must stop the run cleanly.
# --------------------------------------------------------------------------- #


class _DeadController(Controller):
    name = "dead"

    def decide(self, state):
        raise ControllerError("TypeSafe unreachable")


class _FlakyController(Controller):
    name = "flaky"

    def __init__(self) -> None:
        self.calls = 0

    def decide(self, state):
        self.calls += 1
        if self.calls > 1:
            raise ControllerError("Jev died mid-run")
        return ControllerDecision(
            source="flaky",
            action="continue",
            action_probabilities={"continue": 1.0},
            tool="python",
            tool_probabilities={"python": 1.0},
            retain_insight=0.9,
            completeness=0.5,
        )


def test_controller_failure_before_the_first_step(tiny_csv, tmp_path):
    config = _config(tiny_csv, tmp_path, controller="dead", analyst="scripted", max_steps=4)
    result = DataJevAgent(config, _DeadController(), ScriptedAnalyst()).run()

    assert result.steps == 0
    assert result.stop_reason == "controller_error"
    trace = json.loads(result.trace_path.read_text())
    assert trace["errors"] and "TypeSafe unreachable" in trace["errors"][0]
    assert trace["decisions"][0]["action"] == "stop"
    assert "# Final answer" in result.final_answer.text
    assert (result.run_dir / "trace.json").exists()


def test_controller_failure_mid_run_keeps_completed_steps(tiny_csv, tmp_path):
    config = _config(tiny_csv, tmp_path, controller="flaky", analyst="scripted", max_steps=4)
    result = DataJevAgent(config, _FlakyController(), ScriptedAnalyst()).run()

    assert result.steps == 1
    assert result.stop_reason == "controller_error"
    trace = json.loads(result.trace_path.read_text())
    assert trace["steps"][0]["ok"] is True
    assert "Jev died mid-run" in trace["errors"][0]
    assert trace["decisions"][-1]["action"] == "stop"
    assert any(i["retained"] for i in trace["insights"])
