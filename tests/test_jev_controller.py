"""JevController: request shape, answer parsing, retries, trace fidelity.

Everything here runs against an httpx MockTransport - no network, no API key.
"""

from __future__ import annotations

import json

import httpx
import pytest

from datajev.controllers.base import ControllerError
from datajev.controllers.jev import JevController, build_questions
from datajev.state import AnalysisState
from datajev.trace import Trace
from datajev.types import Insight, Step

CANNED_RESPONSE = {
    "model": "jev-1.13.0",
    "answers": {
        "action": {
            "type": "choice",
            "choice": "switch",
            "probabilities": {"continue": 0.18, "switch": 0.62, "verify": 0.14, "stop": 0.04, "other": 0.02},
            "confidence": 0.55,
        },
        "tool": {
            "type": "choice",
            "choice": "python",
            "probabilities": {"python": 0.68, "sql": 0.02, "none": 0.04, "other": 0.26},
            "confidence": 0.51,
        },
        "retain_insight": {"type": "noul", "noul": 0.91},
        "needs_verification": {"type": "noul", "noul": 0.73},
        "completeness": {
            "type": "score",
            "score": 1.42,
            "legend": {"0": "unanswered", "1": "gaps", "2": "covered", "3": "comprehensive"},
            "probabilities": {"0": 0.05, "1": 0.53, "2": 0.37, "3": 0.05},
            "confidence": 0.42,
        },
    },
    "usage": {"input_tokens": 412, "output_tokens": 58},
}


def _controller(handler, **kwargs) -> JevController:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return JevController("test-key", client=client, **kwargs)


def test_build_questions_matches_plan():
    questions = build_questions()
    assert set(questions) == {
        "action",
        "tool",
        "retain_insight",
        "needs_verification",
        "completeness",
    }
    assert questions["action"]["type"] == "choice"
    assert questions["tool"]["type"] == "choice"
    assert "visualization" not in questions["tool"]["criteria"]
    assert questions["retain_insight"]["type"] == "noul"
    assert questions["needs_verification"]["type"] == "noul"
    assert questions["completeness"]["type"] == "score"
    assert len(questions["completeness"]["criteria"]) == 4
    assert len(questions["action"]["criteria"]) == 5


def test_decide_parses_and_preserves_distributions(tiny_state):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json=CANNED_RESPONSE)

    controller = _controller(handler)
    decision = controller.decide(tiny_state)

    assert captured["auth"] == "Bearer test-key"
    payload = captured["payload"]
    assert payload["model"] == "jev-latest"
    assert payload["state"]["goal"] == tiny_state.goal
    # The raw CSV must never be part of the state.
    assert "df" not in json.dumps(payload["state"])
    assert len(json.dumps(payload["state"])) < 6000

    assert decision.source == "jev"
    assert decision.action == "switch"
    assert decision.action_probabilities == pytest.approx(
        {"continue": 0.18, "switch": 0.62, "verify": 0.14, "stop": 0.04, "other": 0.02}
    )
    assert decision.action_confidence == pytest.approx(0.55)
    assert decision.tool == "python"
    assert decision.tool_probabilities["python"] == pytest.approx(0.68)
    assert decision.retain_insight == pytest.approx(0.91)
    assert decision.needs_verification == pytest.approx(0.73)
    assert decision.completeness == pytest.approx(1.42)
    assert decision.completeness_probabilities["1"] == pytest.approx(0.53)
    assert decision.completeness_legend["2"] == "covered"
    assert decision.model == "jev-1.13.0"
    assert decision.usage == {"input_tokens": 412, "output_tokens": 58}
    assert decision.raw["action"]["choice"] == "switch"


def test_decide_normalizes_unnormalized_probabilities(tiny_state):
    payload = json.loads(json.dumps(CANNED_RESPONSE))
    payload["answers"]["action"]["probabilities"] = {
        "continue": 2.0,
        "switch": 6.0,
        "verify": 1.0,
        "stop": 0.0,
        "other": 1.0,
    }

    controller = _controller(lambda request: httpx.Response(200, json=payload))
    decision = controller.decide(tiny_state)
    assert sum(decision.action_probabilities.values()) == pytest.approx(1.0)
    assert decision.action == "switch"


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(ControllerError):
        JevController()


def test_http_error_is_not_retried(tiny_state):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(401, json={"error": "invalid key"})

    controller = _controller(handler, max_retries=3)
    with pytest.raises(ControllerError):
        controller.decide(tiny_state)
    assert calls["n"] == 1


def test_rate_limit_is_retried(tiny_state):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, json={"error": "slow down"})
        return httpx.Response(200, json=CANNED_RESPONSE)

    controller = _controller(handler, max_retries=2)
    decision = controller.decide(tiny_state)
    assert calls["n"] == 2
    assert decision.action == "switch"


def test_trace_preserves_jev_distributions(tiny_state, tmp_path):
    controller = _controller(lambda request: httpx.Response(200, json=CANNED_RESPONSE))
    decision = controller.decide(tiny_state)

    trace = Trace.create(
        run_dir=tmp_path,
        run_id="test-run",
        goal=tiny_state.goal,
        config={"controller": "jev"},
        dataset=tiny_state.dataset.as_dict(),
    )
    step = Step(
        index=1,
        question="q",
        action="switch",
        tool="python",
        code="print(1)",
        stdout="1",
        stderr="",
        ok=True,
        duration_s=0.01,
        artifacts=[],
        decision=decision,
    )
    trace.record_decision(1, decision)
    trace.record_step(
        step,
        [Insight(id="i1", text="t", step_index=1, retained=True, retain_probability=0.91)],
    )
    trace.set_final("answer")
    trace.finish("controller_stop")
    path = trace.save()

    reloaded = json.loads(path.read_text())
    stored = reloaded["decisions"][0]
    assert stored["source"] == "jev"
    assert stored["action_probabilities"]["switch"] == pytest.approx(0.62)
    assert stored["tool_probabilities"]["python"] == pytest.approx(0.68)
    assert stored["completeness_probabilities"]["1"] == pytest.approx(0.53)
    assert stored["completeness_legend"]["2"] == "covered"
    assert stored["retain_insight"] == pytest.approx(0.91)
    assert stored["needs_verification"] == pytest.approx(0.73)
    assert stored["raw"]["completeness"]["score"] == pytest.approx(1.42)
    assert reloaded["insights"][0]["retain_probability"] == pytest.approx(0.91)
    assert reloaded["final_answer"] == "answer"
