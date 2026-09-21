"""DataJev - System-1 control for System-2 data agents.

Let LLMs analyze. Let Jev decide what to analyze next.

Public surface kept intentionally small for the first vertical slice:
CSV + goal -> LLM analyst -> analysis step -> controller -> next decision.
"""

from __future__ import annotations

from datajev.agent import DataJevAgent, RunConfig, RunResult, build_agent
from datajev.state import AnalysisState
from datajev.types import (
    AnalysisPlan,
    ControllerDecision,
    ExecutionResult,
    FinalAnswer,
    Insight,
    Interpretation,
    Step,
)
from datajev.version import __version__

__all__ = [
    "__version__",
    "AnalysisState",
    "AnalysisPlan",
    "ControllerDecision",
    "ExecutionResult",
    "FinalAnswer",
    "Insight",
    "Interpretation",
    "Step",
    "DataJevAgent",
    "RunConfig",
    "RunResult",
    "build_agent",
]
