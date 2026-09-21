"""Controller contracts: the same typed decision shape for every policy."""

from __future__ import annotations

import pytest

from datajev.controllers.heuristic import HeuristicController
from datajev.controllers.llm import LLMController
from datajev.llm import LLMClient, LLMConfig, LLMResponse
from datajev.types import ACTIONS, TOOLS, ControllerDecision, Interpretation, Insight, Step

LLM_REPLY = {
    "action": {"continue": 0.1, "switch": 0.2, "verify": 0.15, "stop": 0.5, "other": 0.05},
    "tool": {"python": 0.65, "sql": 0.05, "none": 0.05, "other": 0.25},
    "retain_insight": 0.62,
    "needs_verification": 0.44,
    "completeness": {"0": 0.0, "1": 0.2, "2": 0.3, "3": 0.5},
}


class _StubLLMClient(LLMClient):
    def __init__(self) -> None:
        super().__init__(LLMConfig(model="stub", api_key="stub"))
        self.calls = 0

    def json(self, system, user, *, max_tokens=2048, temperature=0.0):  # type: ignore[override]
        self.calls += 1
        return LLMResponse(data=LLM_REPLY, text="{}", usage={"prompt_tokens": 10, "completion_tokens": 5})


def _add_step(state, action="continue", insight_text="alpha and beta correlate"):
    decision = ControllerDecision(source="heuristic", action=action)
    step = Step(
        index=state.step_count + 1,
        question="q",
        action=action,
        tool="python",
        code="print(1)",
        stdout="r=0.9",
        stderr="",
        ok=True,
        duration_s=0.01,
        artifacts=[],
        decision=decision,
        interpretation=Interpretation(summary="a strong global relationship", insights=[insight_text]),
    )
    state.add_step(
        step,
        [Insight(id=f"s{step.index}", text=insight_text, step_index=step.index, retained=True)],
    )


def test_controller_decision_shape_is_uniform(tiny_state):
    decision = HeuristicController().decide(tiny_state)
    assert decision.action in ACTIONS
    assert decision.tool in TOOLS
    assert sum(decision.action_probabilities.values()) == pytest.approx(1.0)
    assert sum(decision.tool_probabilities.values()) == pytest.approx(1.0)
    assert sum(decision.completeness_probabilities.values()) == pytest.approx(1.0)
    assert 0.0 <= decision.retain_insight <= 1.0
    assert 0.0 <= decision.needs_verification <= 1.0
    assert 0.0 <= decision.completeness <= 3.0


def test_heuristic_first_step_continues_then_switches(tiny_state):
    controller = HeuristicController()
    first = controller.decide(tiny_state)
    assert first.action == "continue"
    assert first.tool == "python"

    _add_step(tiny_state)
    second = controller.decide(tiny_state)
    assert second.action == "switch", "a single global pass should not be treated as sufficient"
    assert second.tool == "python"


def test_heuristic_stops_only_after_verification(tiny_state):
    controller = HeuristicController()
    _add_step(tiny_state, "continue")
    _add_step(tiny_state, "switch")
    before_verify = controller.decide(tiny_state)
    assert before_verify.action == "verify"
    assert before_verify.needs_verification > 0.5

    tiny_state.verified = True
    tiny_state.verifications = 1
    _add_step(tiny_state, "verify")
    after_verify = controller.decide(tiny_state)
    assert after_verify.action == "stop"
    assert after_verify.completeness > 1.5
    assert after_verify.completeness_probabilities


def test_llm_controller_parses_generative_decisions(tiny_state):
    client = _StubLLMClient()
    decision = LLMController(client=client).decide(tiny_state)
    assert client.calls == 1
    assert decision.source == "llm"
    assert decision.action == "stop"
    assert decision.action_probabilities["stop"] == pytest.approx(0.5)
    assert decision.tool == "python"
    assert decision.retain_insight == pytest.approx(0.62)
    assert decision.needs_verification == pytest.approx(0.44)
    assert decision.completeness == pytest.approx(0.0 * 0 + 0.2 * 1 + 0.3 * 2 + 0.5 * 3)
    assert decision.raw == LLM_REPLY


def test_llm_controller_clamps_out_of_range_nouls(tiny_state):
    client = _StubLLMClient()

    def bad_json(system, user, *, max_tokens=2048, temperature=0.0):  # type: ignore[override]
        return LLMResponse(
            data={"action": {"continue": 1.0}, "retain_insight": 5.0, "needs_verification": -2},
            text="{}",
        )

    client.json = bad_json  # type: ignore[assignment]
    decision = LLMController(client=client).decide(tiny_state)
    assert decision.retain_insight == 1.0
    assert decision.needs_verification == 0.0
    assert decision.action == "continue"
    assert decision.completeness_probabilities == {} or decision.completeness == 0.0
