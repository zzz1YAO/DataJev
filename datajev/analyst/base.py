"""The Analyst: the System-2 half of DataJev.

The analyst reasons, writes code, reads results and produces insights. It never
decides what to do next - that is the controller's job.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from datajev.state import AnalysisState
from datajev.types import AnalysisPlan, ControllerDecision, ExecutionResult, FinalAnswer, Interpretation


class Analyst(ABC):
    """Base class for analysis policies."""

    name: str = "analyst"
    description: str = ""

    @abstractmethod
    def plan(self, state: AnalysisState, decision: ControllerDecision) -> AnalysisPlan:
        """Turn a control decision into a concrete analysis step."""

    @abstractmethod
    def interpret(
        self, state: AnalysisState, plan: AnalysisPlan, execution: ExecutionResult
    ) -> Interpretation:
        """Read the execution result and extract findings."""

    @abstractmethod
    def finalize(self, state: AnalysisState, stop_reason: str) -> FinalAnswer:
        """Write the closing answer for the goal."""

    def close(self) -> None:
        """Release resources."""

    # ------------------------------------------------------------------ #

    def usage(self) -> dict | None:
        return None
