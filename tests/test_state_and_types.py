"""Typed state, insight retention and decision objects."""

from __future__ import annotations

import pytest

from datajev.state import AnalysisState, clip
from datajev.types import (
    ControllerDecision,
    Interpretation,
    Insight,
    Step,
    argmax_option,
    confidence_from_probabilities,
    normalize_probabilities,
    to_dict,
)


def test_clip_shortens_long_text():
    assert clip("abcdef", 4) == "abc…"
    assert clip("abc", 10) == "abc"
    assert clip(None, 4) == ""


def test_to_dict_is_json_ready(tiny_state):
    decision = ControllerDecision(
        source="jev",
        action="switch",
        action_probabilities={"switch": 0.6, "continue": 0.4},
        completeness_probabilities={"1": 0.5, "2": 0.5},
    )
    payload = to_dict(decision)
    assert payload["source"] == "jev"
    assert payload["action_probabilities"]["switch"] == 0.6
    assert isinstance(payload["notes"], list)


def test_normalize_and_argmax_handle_degenerate_input():
    assert normalize_probabilities({}) == {}
    assert normalize_probabilities({"a": 0.0, "b": 0.0}) == {"a": 0.5, "b": 0.5}
    assert argmax_option({"a": 0.2, "b": 0.8}) == "b"
    assert argmax_option({}, "fallback") == "fallback"
    assert confidence_from_probabilities({"a": 0.3, "b": 0.7}) == 0.7
    assert confidence_from_probabilities({}) is None


def test_state_jev_view_stays_compact_and_excludes_rows(tiny_state):
    state = tiny_state
    step = Step(
        index=1,
        question="global relationship",
        action="continue",
        tool="python",
        code="print(1)",
        stdout="x" * 20_000,
        stderr="",
        ok=True,
        duration_s=0.1,
        artifacts=["step_01.png"],
        decision=ControllerDecision(source="jev", action="continue"),
        interpretation=Interpretation(summary="found a strong global relationship", insights=["i"]),
    )
    state.add_step(step, [Insight(id="i1", text="alpha and beta are related", step_index=1, retained=True)])

    payload = state.to_jev_state()
    assert payload["goal"] == state.goal
    assert payload["retained_insights"] == ["alpha and beta are related"]
    assert payload["latest_action"] == "continue"
    assert len(payload["latest_result"]) <= 1800
    assert payload["dataset"]["rows"] == 120
    assert "alpha" in payload["dataset"]["numeric_columns"]
    assert payload["analysis_completeness_so_far"]["steps_run"] == 1

    text = state.render_text()
    assert "GOAL" in text and "RETAINED INSIGHTS" in text
    assert "alpha and beta are related" in text
    assert "dtypes" in text and "day" in text


def test_step_limit_is_included_in_state(tiny_state):
    state = AnalysisState(goal="g", dataset=tiny_state.dataset, max_steps=3)
    payload = state.to_jev_state()
    assert payload["analysis_completeness_so_far"]["max_steps"] == 3
    assert payload["analysis_completeness_so_far"]["steps_run"] == 0


def test_dataset_profile_detects_time_and_categories(tiny_state):
    profile = tiny_state.dataset
    assert profile.rows == 120
    assert "alpha" in profile.numeric_columns
    assert "beta" in profile.numeric_columns
    assert profile.datetime_columns == ["day"]
    assert "region" in profile.categorical_columns
    assert profile.dtypes["day"] in {"object", "string", "str"}


@pytest.mark.parametrize("value,expected", [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)])
def test_decision_defaults(value, expected):
    decision = ControllerDecision(source="heuristic", action="continue", retain_insight=value)
    assert decision.retain_insight == expected
    assert decision.should_run_step is True
    assert ControllerDecision(source="jev", action="stop").should_run_step is False
