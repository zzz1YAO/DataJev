"""Typed objects passed around the DataJev loop.

Everything a controller decides and everything an analyst produces is typed, so
the trace is lossless and a future UI can render Jev's probability
distributions directly from the saved run.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any


# --------------------------------------------------------------------------- #
# Actions / tools
# --------------------------------------------------------------------------- #

ACTIONS = ("continue", "switch", "verify", "stop", "other")
TOOLS = ("python", "sql", "none", "other")

#: Actions that actually run an analysis step.
RUN_ACTIONS = ("continue", "switch", "verify", "other")


@dataclass
class ExecutionResult:
    """Outcome of running one analysis step's code."""

    ok: bool
    stdout: str
    stderr: str
    artifacts: list[str] = field(default_factory=list)
    duration_s: float = 0.0
    timed_out: bool = False
    script_path: str | None = None

    @property
    def output(self) -> str:
        """Combined output, stderr last (what the analyst should read)."""
        parts = [self.stdout.strip()]
        if self.stderr.strip():
            parts.append("[stderr]\n" + self.stderr.strip())
        return "\n".join(p for p in parts if p)


@dataclass
class AnalysisPlan:
    """What the analyst intends to run next."""

    question: str
    code: str
    rationale: str = ""
    tool: str = "python"


@dataclass
class Interpretation:
    """The analyst's reading of an execution result."""

    summary: str
    insights: list[str] = field(default_factory=list)
    finding: str = ""
    usage: dict[str, Any] | None = None


@dataclass
class Insight:
    """One analytical finding, with the controller's retention judgment."""

    id: str
    text: str
    step_index: int
    retained: bool
    retain_probability: float | None = None
    needs_verification: float | None = None
    kind: str = "finding"  # finding | verification | context


@dataclass
class ControllerDecision:
    """A typed control decision made by a Controller.

    ``probabilities`` maps mirror Jev's answer shape (Choice -> distribution
    over options, Score -> distribution over levels) and are always kept, even
    when they are one-hot, so traces from every controller render the same way.
    """

    source: str  # "jev" | "llm" | "heuristic"
    action: str
    action_probabilities: dict[str, float] = field(default_factory=dict)
    action_confidence: float | None = None

    tool: str = "python"
    tool_probabilities: dict[str, float] = field(default_factory=dict)
    tool_confidence: float | None = None

    retain_insight: float = 0.5  # P(latest result is worth keeping)
    needs_verification: float = 0.0  # P(an important finding needs checking)

    completeness: float = 0.0  # probability-weighted position on the score
    completeness_probabilities: dict[str, float] = field(default_factory=dict)
    completeness_legend: dict[str, str] = field(default_factory=dict)
    completeness_confidence: float | None = None

    latency_s: float = 0.0
    model: str | None = None
    usage: dict[str, Any] | None = None
    raw: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def should_run_step(self) -> bool:
        return self.action != "stop"

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class Step:
    """One completed analysis step: decision -> plan -> execution -> reading."""

    index: int
    question: str
    action: str
    tool: str
    code: str
    stdout: str
    stderr: str
    ok: bool
    duration_s: float
    artifacts: list[str]
    decision: ControllerDecision
    interpretation: Interpretation | None = None
    rationale: str = ""
    timed_out: bool = False
    error: str | None = None
    script_path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class FinalAnswer:
    """The analyst's closing answer for the goal."""

    text: str
    usage: dict[str, Any] | None = None
    source: str = "llm"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def to_dict(value: Any) -> Any:
    """Recursively convert dataclasses into plain JSON-serializable values."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {k: to_dict(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): to_dict(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_dict(v) for v in value]
    return value


def argmax_option(probabilities: dict[str, float], default: str = "other") -> str:
    """Highest-probability key, tolerating empty/None input."""
    if not probabilities:
        return default
    return max(probabilities.items(), key=lambda kv: (kv[1], str(kv[0])))[0]


def normalize_probabilities(probabilities: dict[str, float]) -> dict[str, float]:
    """Make a distribution sum to 1 without dropping keys."""
    clean: dict[str, float] = {}
    for key, value in probabilities.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = 0.0
        clean[str(key)] = max(0.0, numeric)
    total = sum(clean.values())
    if total <= 0:
        if not clean:
            return {}
        share = 1.0 / len(clean)
        return {k: share for k in clean}
    return {k: v / total for k, v in clean.items()}


def confidence_from_probabilities(probabilities: dict[str, float]) -> float | None:
    """Peakedness of a distribution, used when the backend omits confidence."""
    if not probabilities:
        return None
    return round(max(probabilities.values()), 6)
