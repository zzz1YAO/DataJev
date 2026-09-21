"""The analytical state handed to the controller.

This is a *compressed* view of the analysis: goal, schema, current direction,
latest result, retained insights and the recent trajectory. It deliberately
does not contain the dataset itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from datajev.dataset import DatasetProfile
from datajev.types import Insight, Step


def clip(text: Any, limit: int) -> str:
    """Trim long text so the state stays inside a System One model budget."""
    value = "" if text is None else str(text)
    value = value.strip()
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "…"


@dataclass
class AnalysisState:
    goal: str
    dataset: DatasetProfile
    available_tools: list[str] = field(default_factory=lambda: ["python"])
    max_steps: int = 6
    steps: list[Step] = field(default_factory=list)
    insights: list[Insight] = field(default_factory=list)
    current_direction: str = ""
    current_question: str = ""
    latest_action: str = ""
    latest_result: str = ""
    latest_summary: str = ""
    last_tool: str = ""
    verified: bool = False
    verifications: int = 0

    # ------------------------------------------------------------------ #
    # Mutation
    # ------------------------------------------------------------------ #

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def action_history(self) -> list[str]:
        return [step.action for step in self.steps]

    def add_step(self, step: Step, insights: list[Insight]) -> None:
        self.steps.append(step)
        self.insights.extend(insights)
        self.current_question = step.question
        self.current_direction = step.question
        self.latest_action = step.action
        self.last_tool = step.tool
        summary = step.interpretation.summary if step.interpretation else ""
        self.latest_summary = clip(summary, 1200)
        self.latest_result = clip(step.stdout, 1800)
        if step.action == "verify":
            self.verifications += 1
            self.verified = True

    def retained_insights(self) -> list[Insight]:
        return [insight for insight in self.insights if insight.retained]

    # ------------------------------------------------------------------ #
    # Views for controllers / analysts
    # ------------------------------------------------------------------ #

    def recent_trajectory(self, limit: int = 6) -> list[str]:
        lines: list[str] = []
        for step in self.steps[-limit:]:
            headline = ""
            if step.interpretation and step.interpretation.summary:
                headline = clip(step.interpretation.summary, 160)
            lines.append(f"Step {step.index} [{step.action}/{step.tool}]: {headline}")
        return lines

    def artifact_names(self) -> list[str]:
        names: list[str] = []
        for step in self.steps:
            names.extend(step.artifacts)
        return names

    def to_jev_state(self) -> dict[str, Any]:
        """Structured state for the TypeSafe API (object, named fields)."""
        dataset = self.dataset
        return {
            "goal": self.goal,
            "dataset": {
                "rows": dataset.rows,
                "columns": dataset.columns,
                "numeric_columns": dataset.numeric_columns,
                "datetime_columns": dataset.datetime_columns,
                "categorical_columns": dataset.categorical_columns,
                "missing": dataset.missing,
                "head": clip(dataset.head, 600),
            },
            "current_direction": clip(self.current_direction, 300) or "not started",
            "latest_action": self.latest_action or "not started",
            "latest_result": clip(self.latest_result, 1500) or "no step has run yet",
            "latest_summary": self.latest_summary or "no step has run yet",
            "retained_insights": [clip(i.text, 220) for i in self.retained_insights()][-12:],
            "discarded_insights": [
                clip(i.text, 160)
                for i in self.insights
                if not i.retained
            ][-6:],
            "recent_trajectory": self.recent_trajectory(),
            "available_tools": self.available_tools,
            "analysis_completeness_so_far": {
                "steps_run": self.step_count,
                "max_steps": self.max_steps,
                "verification_steps": self.verifications,
                "artifacts": self.artifact_names()[-6:],
            },
        }

    def render_text(self) -> str:
        """Human/LLM readable version of the same state."""
        lines: list[str] = []
        dataset = self.dataset
        lines.append("GOAL")
        lines.append(self.goal)
        lines.append("")
        lines.append("DATA")
        lines.append(f"{dataset.rows} rows, {len(dataset.columns)} columns")
        lines.append(f"columns: {', '.join(dataset.columns)}")
        if dataset.dtypes:
            lines.append(f"dtypes: {dataset.dtypes}")
        if dataset.numeric_columns:
            lines.append(f"numeric: {', '.join(dataset.numeric_columns)}")
        if dataset.datetime_columns:
            lines.append(f"time-like: {', '.join(dataset.datetime_columns)}")
        if dataset.missing:
            lines.append(f"missing values: {dataset.missing}")
        lines.append("")
        lines.append("CURRENT DIRECTION")
        lines.append(self.current_direction or "(not started)")
        lines.append("")
        lines.append("LATEST RESULT")
        lines.append(clip(self.latest_result, 1500) or "(no step has run yet)")
        lines.append("")
        lines.append("LATEST READING")
        lines.append(self.latest_summary or "(none)")
        lines.append("")
        lines.append("RETAINED INSIGHTS")
        retained = self.retained_insights()
        if retained:
            for index, insight in enumerate(retained, start=1):
                lines.append(f"{index}. {insight.text}")
        else:
            lines.append("(none retained yet)")
        lines.append("")
        lines.append("RECENT TRAJECTORY")
        trajectory = self.recent_trajectory()
        lines.extend(trajectory or ["(no steps yet)"])
        return "\n".join(lines)
