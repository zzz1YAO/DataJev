"""The DataJev agent loop.

    CSV + goal -> analyst writes a step -> code executes -> analyst reads the
    result -> controller decides what to do next -> ... -> final answer

The controller only ever sees compressed analytical state; the analyst never
decides what to do next. Both halves are swappable.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from datajev.analyst.base import Analyst
from datajev.analyst.executor import StepExecutor
from datajev.analyst.llm import LLMAnalyst
from datajev.analyst.scripted import ScriptedAnalyst
from datajev.controllers.base import Controller, ControllerError
from datajev.controllers.heuristic import HeuristicController
from datajev.controllers.jev import JevController
from datajev.controllers.llm import LLMController
from datajev.dataset import DatasetProfile, profile_csv
from datajev.llm import resolve_llm_config
from datajev.state import AnalysisState
from datajev.trace import Trace
from datajev.types import (
    ControllerDecision,
    FinalAnswer,
    Insight,
    Interpretation,
    Step,
)

EventCallback = Callable[[str, dict[str, Any]], None]


@dataclass
class RunConfig:
    """Everything needed for one run."""

    csv_path: str
    goal: str
    out_dir: str = "runs"
    controller: str = "auto"  # auto | jev | llm | heuristic
    analyst: str = "auto"  # auto | llm | scripted
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    typesafe_api_key: str | None = None
    jev_model: str = "jev-latest"
    max_steps: int = 6
    min_steps: int = 1
    timeout_s: float = 90.0
    run_name: str | None = None
    seed: int | None = None


@dataclass
class RunResult:
    run_dir: Path
    trace_path: Path
    final_answer: FinalAnswer
    stop_reason: str
    controller: str
    analyst: str
    steps: int
    insights: list[Insight] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


def resolve_controller_name(config: RunConfig) -> str:
    if config.controller != "auto":
        return config.controller
    import os

    if config.typesafe_api_key or os.environ.get("TYPESAFE_API_KEY"):
        return "jev"
    return "heuristic"


def resolve_analyst_name(config: RunConfig) -> str:
    if config.analyst != "auto":
        return config.analyst
    llm_config = resolve_llm_config(config.model, config.api_key, config.base_url)
    return "llm" if llm_config.available else "scripted"


def build_controller(name: str, config: RunConfig) -> Controller:
    if name == "jev":
        return JevController(
            config.typesafe_api_key,
            model=config.jev_model,
        )
    if name == "llm":
        return LLMController(
            model=config.model, api_key=config.api_key, base_url=config.base_url
        )
    if name == "heuristic":
        return HeuristicController()
    raise ValueError(f"Unknown controller: {name!r} (expected jev, llm or heuristic)")


def build_analyst(name: str, config: RunConfig) -> Analyst:
    if name == "llm":
        return LLMAnalyst(model=config.model, api_key=config.api_key, base_url=config.base_url)
    if name == "scripted":
        return ScriptedAnalyst()
    raise ValueError(f"Unknown analyst: {name!r} (expected llm or scripted)")


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #


class DataJevAgent:
    """Runs the analysis loop for one CSV and one goal."""

    def __init__(
        self,
        config: RunConfig,
        controller: Controller,
        analyst: Analyst,
        *,
        executor: Any | None = None,
    ) -> None:
        self.config = config
        self.controller = controller
        self.analyst = analyst
        self._executor_override = executor
        self.controller_name = getattr(controller, "name", "controller")
        self.analyst_name = getattr(analyst, "name", "analyst")

    # ------------------------------------------------------------------ #

    def run(self, on_event: EventCallback | None = None) -> RunResult:
        emit = on_event or (lambda name, payload: None)

        profile = profile_csv(self.config.csv_path)
        run_dir = self._prepare_run_dir(profile)
        run_id = run_dir.name
        trace = Trace.create(
            run_dir=run_dir,
            run_id=run_id,
            goal=self.config.goal,
            config={
                "controller": self.controller_name,
                "controller_description": getattr(self.controller, "description", ""),
                "analyst": self.analyst_name,
                "analyst_description": getattr(self.analyst, "description", ""),
                "llm_model": resolve_llm_config(
                    self.config.model, self.config.api_key, self.config.base_url
                ).model,
                "jev_model": self.config.jev_model if self.controller_name == "jev" else None,
                "max_steps": self.config.max_steps,
                "min_steps": self.config.min_steps,
                "timeout_s": self.config.timeout_s,
            },
            dataset=profile.as_dict(),
        )

        state = AnalysisState(
            goal=self.config.goal,
            dataset=profile,
            max_steps=self.config.max_steps,
        )
        executor = self._executor_override or StepExecutor(
            csv_path=self.config.csv_path,
            run_dir=run_dir,
            timeout_s=self.config.timeout_s,
        )

        emit(
            "run_start",
            {
                "run_dir": str(run_dir),
                "dataset": profile,
                "goal": self.config.goal,
                "controller": self.controller_name,
                "analyst": self.analyst_name,
                "config": self.config,
            },
        )

        stop_reason = "max_steps"
        started = time.perf_counter()
        try:
            for step_index in range(1, self.config.max_steps + 1):
                decision, failure = self._decide(state, trace, emit)
                step_index = len(state.steps) + 1
                trace.record_decision(step_index, decision)
                emit("decision", {"step_index": step_index, "decision": decision})

                self._apply_retention(state, decision, trace)

                if failure is not None:
                    # Control is unavailable: stop instead of running analyses
                    # with no controller policy behind them.
                    stop_reason = "controller_error"
                    break

                if decision.action == "stop":
                    if state.step_count < self.config.min_steps:
                        decision.notes.append(
                            f"stop suppressed: at least {self.config.min_steps} analysis step(s) required"
                        )
                        decision.action = "continue"
                    else:
                        stop_reason = "controller_stop"
                        break

                try:
                    plan = self.analyst.plan(state, decision)
                except Exception as exc:
                    message = f"analyst error before step {step_index}: {exc}"
                    if trace.errors is not None:
                        trace.errors.append(message)
                    emit("analyst_error", {"message": message, "step_index": step_index})
                    stop_reason = "analyst_error"
                    break
                execution = executor.run(plan.code, step_index)
                interpretation = self.analyst.interpret(state, plan, execution)
                insights = self._make_insights(
                    state, step_index, decision, interpretation, execution.ok
                )

                step = Step(
                    index=step_index,
                    question=plan.question,
                    action=decision.action,
                    tool=decision.tool,
                    code=plan.code,
                    stdout=execution.stdout,
                    stderr=execution.stderr,
                    ok=execution.ok,
                    duration_s=execution.duration_s,
                    artifacts=execution.artifacts,
                    decision=decision,
                    interpretation=interpretation,
                    rationale=plan.rationale,
                    timed_out=execution.timed_out,
                    error=None if execution.ok else _last_line(execution.stderr),
                    script_path=execution.script_path,
                )
                state.add_step(step, insights)
                trace.record_step(step, insights)
                emit(
                    "step",
                    {"step": step, "insights": insights, "analyst": self.analyst_name},
                )
            else:
                if state.steps:
                    trace.mark_retention_fallback()
                    self._final_retention_fallback(state)
        finally:
            self.controller.close()
            self.analyst.close()

        try:
            final = self.analyst.finalize(state, stop_reason)
        except Exception as exc:  # the trace must survive a failing final answer
            trace.errors.append(f"final answer generation failed: {exc}")
            final = FinalAnswer(
                text=f"# Final answer\n\nFinal answer generation failed: {exc}",
                source="fallback",
            )
        trace.set_final(final.text)
        trace.finish(
            stop_reason=stop_reason,
            controller_usage=_usage_of(self.controller),
            analyst_usage=self.analyst.usage(),
        )
        trace_path = trace.save()
        emit(
            "run_finish",
            {
                "run_dir": str(run_dir),
                "trace_path": str(trace_path),
                "final_answer": final.text,
                "stop_reason": stop_reason,
                "duration_s": round(time.perf_counter() - started, 2),
                "insights": state.insights,
            },
        )

        return RunResult(
            run_dir=run_dir,
            trace_path=trace_path,
            final_answer=final,
            stop_reason=stop_reason,
            controller=self.controller_name,
            analyst=self.analyst_name,
            steps=state.step_count,
            insights=state.insights,
            trace=trace.to_dict(),
        )

    # ------------------------------------------------------------------ #

    def _decide(
        self, state: AnalysisState, trace: Trace, emit: EventCallback
    ) -> tuple[ControllerDecision, str | None]:
        """Ask the controller for the next decision.

        Returns the decision and, when the controller backend failed, the error
        message. A controller failure is recorded and stops the run: continuing
        to analyze without a control policy would defeat the architecture.
        """
        try:
            return self.controller.decide(state), None
        except ControllerError as exc:
            message = f"controller error before step {state.step_count + 1}: {exc}"
            if trace.errors is not None:
                trace.errors.append(message)
            emit("controller_error", {"message": message})
            decision = Controller._fallback_decision(message, self.controller_name, action="stop")
            return decision, message

    @staticmethod
    def _apply_retention(state: AnalysisState, decision: ControllerDecision, trace: Trace) -> None:
        """Judge the most recent step's insights with the controller's Noul."""
        for insight in state.insights:
            if insight.retain_probability is not None:
                continue
            insight.retain_probability = decision.retain_insight
            insight.needs_verification = decision.needs_verification
            insight.retained = decision.retain_insight >= 0.5 or insight.kind == "verification"

    @staticmethod
    def _final_retention_fallback(state: AnalysisState) -> None:
        """Step budget ran out: keep the last findings rather than dropping them."""
        for insight in state.insights:
            if insight.retain_probability is None:
                insight.retained = True

    @staticmethod
    def _make_insights(
        state: AnalysisState,
        step_index: int,
        decision: ControllerDecision,
        interpretation: Interpretation,
        ok: bool,
    ) -> list[Insight]:
        """Turn the analyst's reading into insights, dropping exact repeats.

        A failed step provides no evidence, so it produces no insights (its
        reading stays on the step itself). Analysts - LLMs in particular - also
        restate earlier findings: the controller judges *redundancy* with its
        retain_insight Noul, and the loop additionally refuses to store
        byte-identical repeats.
        """
        if not ok:
            return []
        kind = "verification" if decision.action == "verify" else "finding"
        seen = {_normalize_insight(insight.text) for insight in state.insights}
        insights: list[Insight] = []
        for text in interpretation.insights:
            key = _normalize_insight(text)
            if not key or key in seen:
                continue
            seen.add(key)
            insights.append(
                Insight(
                    id=f"step{step_index:02d}-i{len(insights) + 1}",
                    text=text,
                    step_index=step_index,
                    retained=False,
                    retain_probability=None,
                    needs_verification=None,
                    kind=kind,
                )
            )
        return insights

    def _prepare_run_dir(self, profile: DatasetProfile) -> Path:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        slug = self.config.run_name or re.sub(r"[^a-z0-9]+", "-", Path(profile.path).stem.lower()).strip("-")
        slug = slug or "run"
        run_dir = Path(self.config.out_dir) / f"{stamp}-{slug}"
        counter = 2
        while run_dir.exists():
            run_dir = Path(self.config.out_dir) / f"{stamp}-{slug}-{counter}"
            counter += 1
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir


def build_agent(config: RunConfig, on_event: EventCallback | None = None) -> DataJevAgent:
    """Build an agent from a run config, resolving auto choices."""
    controller_name = resolve_controller_name(config)
    analyst_name = resolve_analyst_name(config)
    controller = build_controller(controller_name, config)
    analyst = build_analyst(analyst_name, config)
    return DataJevAgent(config, controller, analyst)


def _usage_of(controller: Controller) -> dict[str, Any] | None:
    usage = getattr(controller, "total_usage", None)
    if usage:
        return dict(usage)
    client = getattr(controller, "client", None)
    if client is not None and getattr(client, "total_usage", None):
        return dict(client.total_usage)
    return None


def _last_line(text: str) -> str:
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _normalize_insight(text: str) -> str:
    """Whitespace/case/punctuation-insensitive key for duplicate detection."""
    return re.sub(r"\s+", " ", str(text).strip().lower()).rstrip(".")
