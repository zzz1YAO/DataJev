"""Controller abstraction.

A Controller looks at the *compressed analytical state* and returns typed
decisions. It never sees the raw CSV and never runs analysis. Swapping the
controller is what changes the control policy of the agent.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from datajev.state import AnalysisState
from datajev.types import ControllerDecision


class ControllerError(RuntimeError):
    """Raised when a controller cannot produce a decision."""


class Controller(ABC):
    """Base class for all control policies."""

    name: str = "controller"
    #: Description used in traces/UI.
    description: str = ""

    @abstractmethod
    def decide(self, state: AnalysisState) -> ControllerDecision:
        """Return the next typed decision for this state."""

    def close(self) -> None:
        """Release resources (HTTP clients, etc.)."""

    # ------------------------------------------------------------------ #
    # Shared helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _fallback_decision(note: str, source: str, action: str = "continue") -> ControllerDecision:
        """A safe decision used when a backend fails mid-run."""
        return ControllerDecision(
            source=source,
            action=action,
            action_probabilities={action: 1.0},
            action_confidence=1.0,
            tool="python",
            tool_probabilities={"python": 1.0},
            tool_confidence=1.0,
            retain_insight=0.5,
            needs_verification=0.0,
            completeness=0.0,
            completeness_probabilities={"0": 1.0},
            completeness_legend={"0": "unknown"},
            completeness_confidence=1.0,
            notes=[note],
        )
