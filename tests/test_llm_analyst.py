"""The LLM analyst path, exercised end-to-end with a stubbed model.

No network and no API key: only the model call is stubbed, everything else -
code generation, subprocess execution, interpretation, retention, final answer,
trace writing - is the real implementation.
"""

from __future__ import annotations

import json

import pytest

from datajev.agent import DataJevAgent, RunConfig
from datajev.analyst.llm import AnalystError, LLMAnalyst, _strip_fence
from datajev.controllers.heuristic import HeuristicController
from datajev.llm import LLMClient, LLMConfig, LLMResponse, _parse_json
from datajev.types import ControllerDecision

PLAN_CODE = """
print("RESULT: pearson r=0.81 between alpha and beta")
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [3, 2, 1])
plt.savefig("stub_plot.png", dpi=60)
plt.close()
print("RESULT: wrote stub_plot.png")
"""


class StubModel(LLMClient):
    """Answers each of the analyst's three calls with canned JSON."""

    def __init__(self) -> None:
        super().__init__(LLMConfig(model="stub-model", api_key="stub"))
        self.plan_calls = 0
        self.interpret_calls = 0
        self.final_calls = 0

    def json(self, system, user, *, max_tokens=2048, temperature=0.0):  # type: ignore[override]
        if "senior data analyst" in system:
            self.plan_calls += 1
            data = {
                "question": f"Step {self.plan_calls}: how strong is the relationship?",
                "code": PLAN_CODE,
                "rationale": "stub rationale",
            }
        elif "executed analysis step" in system:
            self.interpret_calls += 1
            data = {
                "summary": f"Stub reading {self.interpret_calls}: r=0.81.",
                "insights": [f"Stub insight {self.interpret_calls}: alpha and beta correlate at r=0.81."],
                "finding": "stub",
            }
        elif "final answer" in system:
            self.final_calls += 1
            data = {"answer": "# Final answer\n\nStub answer grounded in the retained insights."}
        else:  # pragma: no cover - guard against prompt drift
            raise AssertionError(f"unexpected system prompt: {system[:60]}")
        usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        self._accumulate(usage)
        return LLMResponse(data=data, text=json.dumps(data), usage=usage)


class TruncatedPlanModel(LLMClient):
    """First plan response is truncated JSON; the repair response is valid."""

    def __init__(self) -> None:
        super().__init__(LLMConfig(model="stub-model", api_key="stub"))
        self.plan_calls = 0
        self.users: list[str] = []

    def json(self, system, user, *, max_tokens=2048, temperature=0.0):  # type: ignore[override]
        if "senior data analyst" not in system:
            raise AssertionError("the regression test only exercises plan parsing")
        self.plan_calls += 1
        self.users.append(user)
        usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        if self.plan_calls == 1:
            raw = '{"question":"verify the finding","code":"import numpy as np\nprint('
            data, parse_error = _parse_json(raw)
            return LLMResponse(
                data=data,
                text=raw,
                usage=usage,
                parse_error=parse_error,
                finish_reason="length",
            )
        data = {"question": "verify the finding", "code": "print(1)", "rationale": "repair"}
        return LLMResponse(data=data, text=json.dumps(data), usage=usage)


class UnparseablePlanModel(TruncatedPlanModel):
    """Both plan attempts fail so the agent must persist the raw response."""

    def json(self, system, user, *, max_tokens=2048, temperature=0.0):  # type: ignore[override]
        if "senior data analyst" in system:
            self.plan_calls += 1
            raw = '{"question":"verify","code":"import numpy as np\nprint('
            data, parse_error = _parse_json(raw)
            return LLMResponse(
                data=data,
                text=raw,
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                parse_error=parse_error,
                finish_reason="length",
            )
        data = {"answer": "final fallback"}
        return LLMResponse(data=data, text=json.dumps(data), usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})


class TruncatedFinalModel(LLMClient):
    """First final response is truncated JSON; the repair response is valid."""

    def __init__(self) -> None:
        super().__init__(LLMConfig(model="stub-model", api_key="stub"))
        self.final_calls = 0

    def json(self, system, user, *, max_tokens=2048, temperature=0.0):  # type: ignore[override]
        if "final answer" not in system:
            raise AssertionError("the regression test only exercises final parsing")
        self.final_calls += 1
        usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        if self.final_calls == 1:
            raw = '{"answer":"## Final answer\\n\\n| Period | Pearson r |\\n|---|---:|\\n| Third 1 |'
            data, parse_error = _parse_json(raw)
            return LLMResponse(
                data=data,
                text=raw,
                usage=usage,
                parse_error=parse_error,
                finish_reason="length",
            )
        data = {"answer": "# Final answer\n\nRecovered complete answer."}
        return LLMResponse(data=data, text=json.dumps(data), usage=usage)


def test_strip_fence():
    assert _strip_fence("```python\nprint(1)\n```") == "print(1)"
    assert _strip_fence("print(1)") == "print(1)"
    assert _strip_fence("") == ""


def test_llm_analyst_repairs_truncated_plan_json(tiny_state):
    model = TruncatedPlanModel()
    analyst = LLMAnalyst(client=model)
    plan = analyst.plan(
        tiny_state,
        ControllerDecision(source="heuristic", action="verify", tool="python"),
    )

    assert plan.code == "print(1)"
    assert model.plan_calls == 2
    assert "PREVIOUS RAW RESPONSE" in model.users[1]
    assert "import numpy as np" in model.users[1]


def test_unparseable_plan_is_persisted_in_trace(tiny_csv, tmp_path):
    model = UnparseablePlanModel()
    analyst = LLMAnalyst(client=model)
    config = RunConfig(
        csv_path=tiny_csv,
        goal="g",
        out_dir=str(tmp_path / "runs"),
        controller="heuristic",
        analyst="llm",
        max_steps=1,
    )
    result = DataJevAgent(config, HeuristicController(), analyst).run()
    trace = json.loads(result.trace_path.read_text())

    assert result.stop_reason == "analyst_error"
    assert trace["errors"]
    assert "structured output could not be parsed" in trace["errors"][0]
    assert "import numpy as np" in trace["errors"][0]
    assert "Analyst returned no code" not in trace["errors"][0]


def test_llm_analyst_repairs_truncated_final_json(tiny_state):
    model = TruncatedFinalModel()
    analyst = LLMAnalyst(client=model)

    answer = analyst.finalize(tiny_state, "controller_stop")

    assert answer.source == "llm"
    assert "Recovered complete answer" in answer.text
    assert model.final_calls == 2


def test_llm_analyst_requires_a_key(monkeypatch):
    for name in ("DATAJEV_LLM_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY", "deepseek_api"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(AnalystError):
        LLMAnalyst()


def test_llm_analyst_drives_the_loop(tiny_csv, tmp_path):
    model = StubModel()
    analyst = LLMAnalyst(client=model)
    config = RunConfig(
        csv_path=tiny_csv,
        goal="How are alpha and beta related?",
        out_dir=str(tmp_path / "runs"),
        controller="heuristic",
        analyst="llm",
        model="stub-model",
        max_steps=3,
    )
    result = DataJevAgent(config, HeuristicController(), analyst).run()

    assert result.steps == 3
    assert model.plan_calls == 3
    assert model.interpret_calls == 3
    assert model.final_calls == 1

    trace = json.loads(result.trace_path.read_text())
    assert trace["config"]["analyst"] == "llm"
    assert trace["insights"], "stubbed insights should reach the trace"
    # max_steps=3 exhausts the budget, so the final step's findings are kept by
    # the explicit fallback rather than by a controller judgment.
    assert trace["retention_fallback"] is True
    judged = [i for i in trace["insights"] if i["retain_probability"] is not None]
    assert judged, "insights from completed steps must carry the controller's judgment"
    assert all(i["retained"] for i in trace["insights"])
    assert len({i["text"] for i in trace["insights"]}) == len(trace["insights"]), "duplicates dropped"
    assert "Stub answer" in result.final_answer.text

    # The generated code really ran in a subprocess and produced a real file.
    assert "artifacts/stub_plot.png" in trace["artifacts"]
    assert (result.run_dir / "artifacts" / "stub_plot.png").exists()
    assert "pearson r=0.81" in trace["steps"][0]["stdout"]
    assert trace["usage"]["analyst"]["calls"] == 7
    assert trace["usage"]["analyst"]["total_tokens"] == 7 * 150


def test_llm_analyst_survives_a_failed_step(tiny_csv, tmp_path):
    class BreakingModel(StubModel):
        def json(self, system, user, *, max_tokens=2048, temperature=0.0):  # type: ignore[override]
            if "senior data analyst" in system:
                self.plan_calls += 1
                return LLMResponse(
                    data={"question": "q", "code": "raise RuntimeError('nope')", "rationale": ""},
                    text="{}",
                )
            return super().json(system, user, max_tokens=max_tokens, temperature=temperature)

    model = BreakingModel()
    analyst = LLMAnalyst(client=model)
    config = RunConfig(
        csv_path=tiny_csv,
        goal="g",
        out_dir=str(tmp_path / "runs"),
        controller="heuristic",
        analyst="llm",
        max_steps=2,
    )
    result = DataJevAgent(config, HeuristicController(), analyst).run()

    trace = json.loads(result.trace_path.read_text())
    assert result.steps == 2
    assert all(step["ok"] is False for step in trace["steps"])
    assert all("nope" in step["stderr"] for step in trace["steps"])
    assert trace["insights"] == []
    assert "# Final answer" in result.final_answer.text
